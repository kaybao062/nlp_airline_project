! pip install pinecone-clientimport logging
import sys
import os
import pinecone

from dotenv import load_dotenv
from pinecone import (
    Pinecone,
)
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def load_vector_index(index_name: str = "langchain-retrieval-agent2"):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        text_key="Review Content",
    )
    return VectorStoreIndex.from_vector_store(vector_store)
 
