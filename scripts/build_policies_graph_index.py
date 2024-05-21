# hack to import modules from parent directories:
import os
import sys

sys.path.insert(0, f"{os.path.dirname(os.path.abspath(__file__))}/../")

import logging

import openai
from dotenv import load_dotenv
from llama_index.core.settings import Settings
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.llms.openai import OpenAI

from config import load_graph_config
from util.dataset import load_policies
from util.graph import build_graph_index
from util.multi_tenant import build_multi_tenant_index

# prepare environment:
load_dotenv()

# prepare logger:
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
)

# prepare model:
openai.api_key = os.environ["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.chunk_size = 256
Settings.node_parser = SentenceSplitter(
    chunk_size=200,
    chunk_overlap=20,
    paragraph_separator="\n\n"
)
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
# Settings.num_output = 512
# Settings.context_window = 3900


if __name__ == "__main__":
    config = load_graph_config(
        persist_dir="./data/index/policies",
    )
    documents = load_policies()
    build_multi_tenant_index(
        documents,
        tenant_key="airline",
        config=config,
        build_index_fn=build_graph_index,
    )
    print("Done!")
