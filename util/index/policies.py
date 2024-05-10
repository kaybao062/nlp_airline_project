import os
from pathlib import Path

from pydantic import BaseModel
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.core import (
    load_index_from_storage,
    KnowledgeGraphIndex,
    StorageContext,
)

INDEX_PATH = os.getenv("INDEX_PATH", "./data/index/")


class GraphConfig(BaseModel):
    """Configuration for Policies Index."""
    space_name: str = "policies_aa"
    edge_types: list[str] = ["relationship"]
    # default, could be omit if create from an empty kg
    rel_prop_names: list[str] = ["name"]
    # default, could be omit if create from an empty kg
    tags: list[str] = ["entity"]
    persist_dir: Path = None


def load_config(space_name: str, persist_dir: Path = None):
    """Load graph configuration."""
    config = GraphConfig(
        space_name=space_name,
        persist_dir=persist_dir or Path(INDEX_PATH) / space_name,
    )
    return config


def load_store_context(config: GraphConfig):
    """Load Graph Store."""
    graph_store = NebulaGraphStore(
        space_name=config.space_name,
        edge_types=config.edge_types,
        rel_prop_names=config.rel_prop_names,
        tags=config.tags,
    )
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=config.persist_dir,
            graph_store=graph_store
        )
    except FileNotFoundError:
        storage_context = StorageContext.from_defaults(
            graph_store=graph_store
        )
    return storage_context


def load_index(config: GraphConfig):
    storage_context = load_store_context(config.space_name)
    index = load_index_from_storage(
        storage_context=storage_context,
        max_triplets_per_chunk=10,
        space_name=config.space_name,
        edge_types=config.edge_types,
        rel_prop_names=config.rel_prop_names,
        tags=config.tags,
        verbose=True,
    )
    return index
