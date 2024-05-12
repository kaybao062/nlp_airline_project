from llama_index.core import (
    load_index_from_storage,
    KnowledgeGraphIndex,
    StorageContext,
)
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.graph_stores.nebula import NebulaGraphStore


from config import GraphConfig
from util.nebula import create_space


def load_graph_store_context(config: GraphConfig):
    """Load Graph Store."""
    # without nebula, this is just a similar text retriever
    # it cant make graph queries...
    # maybe we try simple graph store instead?
    if not config.nebula:
        return StorageContext.from_defaults(
            persist_dir=str(config.persist_dir),
        )
    # graph_store = NebulaGraphStore(
    graph_store = SimpleGraphStore(
        space_name=config.index_name,
        edge_types=config.edge_types,
        rel_prop_names=config.rel_prop_names,
        tags=config.tags,
    )
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=str(config.persist_dir),
            graph_store=graph_store
        )
    except FileNotFoundError:
        storage_context = StorageContext.from_defaults(
            graph_store=graph_store
        )
    return storage_context


def load_graph_index_from_config(config: GraphConfig):
    """Load Graph Index from config."""
    storage_context = load_graph_store_context(config)
    index = load_index_from_storage(
        # index_id=config.index_name,
        storage_context=storage_context,
        max_triplets_per_chunk=config.max_triplets_per_chunk,
        space_name=config.index_name,
        edge_types=config.edge_types,
        rel_prop_names=config.rel_prop_names,
        tags=config.tags,
        include_embeddings=config.include_embeddings,
        verbose=config.verbose,
    )
    return index


def load_graph_query_engine(config: GraphConfig):
    """Load query engine from graph index."""
    index = load_graph_index_from_config(config)
    query_engine = index.as_query_engine(
        retriever_mode=config.retriever_mode,
        verbose=config.verbose,
        response_mode=config.response_mode,
        graph_store_query_depth=config.graph_store_query_depth,
    )
    return query_engine


def build_graph_index(
    documents: list,
    config: GraphConfig = None,
):
    """Build graph index."""
    # check if already created:
    if config.persist_dir is not None:
        if config.persist_dir.exists():
            print(f"Skip: already exists: {config.persist_dir}")
            return
    # create space:
    print(f"Creating space: {config.index_name}")
    create_space(config.index_name)
    # load storage context:
    storage_context = load_graph_store_context(config)
    # build index:
    print(f"Building: {config.index_name}")
    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=config.max_triplets_per_chunk,
        space_name=config.index_name,
        edge_types=config.edge_types,
        rel_prop_names=config.rel_prop_names,
        tags=config.tags,
        include_embeddings=config.include_embeddings,
        # kg_triplet_extract_fn=extract_triplets,
        verbose=config.verbose,
    )
    # persist index:
    # index.set_index_id(config.index_name)
    if config.persist_dir is not None:
        config.persist_dir.mkdir(exist_ok=True, parents=True)
        print(f"Persisting: {config.persist_dir}")
        index.storage_context.persist(persist_dir=config.persist_dir)
    return index
