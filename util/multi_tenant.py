from pathlib import Path
from typing import Callable
from slugify import slugify

from config import Config


def extract_tenants(
    documents: list,
    tenant_key: str
) -> list:
    tenants = [doc.metadata[tenant_key] for doc in documents]
    tenants = list(set(tenants))
    return tenants


def build_multi_tenant_index(
    documents: list,
    tenant_key: str,
    config: Config = None,
    build_index_fn: Callable = None,
    skip_tenant: list = [],

):
    tenants = extract_tenants(documents, tenant_key=tenant_key)
    for tenant in tenants:
        tenant_index_name = slugify(tenant, separator="_", lowercase=True)
        if tenant_index_name in skip_tenant or tenant in skip_tenant:
            print(f"Skip: {tenant}")
            continue
        # scope tenant documents:
        tenant_documents = [doc for doc in documents if doc.metadata[tenant_key] == tenant]
        # update config:
        tenant_persist_dir = Path(config.persist_dir) / tenant_index_name if config.persist_dir else None
        tenant_config = config.model_copy(update={
            "index_name": tenant_index_name,
            "persist_dir": tenant_persist_dir,
        })
        # build tenant index:
        build_index_fn(
            tenant_documents,
            config=tenant_config,
        )
