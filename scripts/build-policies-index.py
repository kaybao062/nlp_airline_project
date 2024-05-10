# hack to import modules from parent directories:
import os
import logging
import sys

sys.path.insert(0, f"{os.path.dirname(os.path.abspath(__file__))}/../")

import openai
from dotenv import load_dotenv
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.settings import Settings
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.llms.openai import OpenAI

from util.dataset import load_policies
from util.index.policies import (
    load_config,
    load_store_context,
)
from util.nebula import create_space

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


def build_index(name: str):
    # load store:
    policy_config = load_config(name)
    storage_context = load_store_context(policy_config)

    # initialize empty index:
    index: KnowledgeGraphIndex = None

    # load airlines:
    policies = load_policies(as_dataframe=True, limit=1000)
    airlines = policies["Airline"].unique()
    print(f"Found airlines: {airlines}")

    # iterate through airlines, filter policies grouped by airlines,
    # and insert policy documents into index in batches
    for airline in airlines:
        print(f"Loading policies: {airline}")
        airline_policies = load_policies(airline=airline)

        # insert documents to index
        if index is None:
            index = KnowledgeGraphIndex.from_documents(
                airline_policies,
                storage_context=storage_context,
                max_triplets_per_chunk=20,
                space_name=policy_config.space_name,
                edge_types=policy_config.edge_types,
                rel_prop_names=policy_config.rel_prop_names,
                tags=policy_config.tags,
                include_embeddings=True,
                # kg_triplet_extract_fn=extract_triplets,
                verbose=True,
            )
        else:
            index.insert(airline_policies, verbose=True)
        
        print(f"Persisting policies: {policy_config.persist_dir}")
        index.storage_context.persist(persist_dir=policy_config.persist_dir)

    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()

    create_space(args.name)
    build_index(args.name)
