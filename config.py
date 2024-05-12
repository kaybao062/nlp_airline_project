from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    pass


class GraphConfig(Config):
    """Configuration for Graph Index."""
    index_name: str = "space"
    edge_types: list[str] = ["relationship"]
    # default, could be omit if create from an empty kg
    rel_prop_names: list[str] = ["name"]
    # default, could be omit if create from an empty kg
    tags: list[str] = ["entity"]
    # storage settings:
    nebula: bool = False
    persist_dir: Path = None
    # knowledge graph index settings:
    max_triplets_per_chunk: int = 20
    include_embeddings: bool = True
    verbose: bool = True
    # query engine settings:
    graph_store_query_depth: int = 4
    response_mode: str = "tree_summarize"
    retriever_mode: str = "hybrid"


def load_graph_config(**kwargs):
    """Load graph configuration."""
    return GraphConfig(**kwargs)
