import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

r = redis.Redis(
  host=os.getenv('REDIS_HOST'),
  port=13032,
  password=os.getenv('REDIS_PASSWORD'))

hfkey = os.getenv('HF_KEY')

INDEX_NAME = "index"                              # Vector Index Name
DOC_PREFIX = "doc:"                               # RediSearch Key Prefix for the Index
model = SentenceTransformer('all-MiniLM-L6-v2')   # hugging face embeddings model

def create_index(vector_dimensions: int):
    try:
        # check to see if index exists
        r.ft(INDEX_NAME).info()
        print("Index already exists!")
    except:
        # schema
        schema = (
            TagField("tag"),                       # Tag Field Name
            VectorField("vector",                  # Vector Field Name
                "FLAT", {                          # Vector Index Type: FLAT or HNSW
                    "TYPE": "FLOAT32",             # FLOAT32 or FLOAT64
                    "DIM": vector_dimensions,      # Number of Vector Dimensions
                    "DISTANCE_METRIC": "COSINE",   # Vector Search Distance Metric
                }
            ),
        )

        # index Definition
        definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)

        # create Index
        r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)

# define vector dimensions
VECTOR_DIMENSIONS = 384

# delete index
r.ft(INDEX_NAME).dropindex(delete_documents=True)

# create the index
create_index(vector_dimensions=VECTOR_DIMENSIONS)

# create embeddings with huggingface sentence transformer
texts = [
    "Today is a really great day!",
    "The dog next door barks really loudly.",
    "My cat escaped and got out before I could close the door.",
    "It's supposed to rain and thunder tomorrow."
]
response = model.encode(texts)
embeddings = np.array([r for r in response], dtype=np.float32)
print("EMBEDDINGS")
print(embeddings)

# Write to Redis
pipe = r.pipeline()
for i, embedding in enumerate(embeddings):
    pipe.hset(f"doc:{i}", mapping = {
        "vector": embedding.astype(np.float32).tobytes(),
        "content": texts[i],
        "tag": "huggingface"
    })
res = pipe.execute()

#embed query text
text = "can you find me animals"
response = model.encode(text)
query_embedding = np.array([r for r in response], dtype=np.float32)

# query for similar documents that have the openai tag
query = (
    Query("(@tag:{ huggingface })=>[KNN 2 @vector $vec as score]")
     .sort_by("score")
     .return_fields("content", "tag", "score")
     .paging(0, 2)
     .dialect(2)
)
query_params = {"vec": query_embedding.tobytes()}
results = r.ft(INDEX_NAME).search(query, query_params).docs

# the two pieces of content related to animals are returned
print("RESULTS")
print(results)
