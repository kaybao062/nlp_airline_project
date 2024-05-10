import os
from functools import cache

from dotenv import load_dotenv
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.common import *
from pydantic import BaseModel, Field

load_dotenv()

class NebulaConfig(BaseModel):
    address: str = "127.0.0.1:9669"
    user: str = "root"
    password: str = "nebula"


@cache
def load_client(config: NebulaConfig = None):
    if config is None:
        config = NebulaConfig()
        host, port = os.getenv("NEBULA_ADDRESS", config.address).split(":")
        user = os.getenv("NEBULA_USER", config.user)
        password = os.getenv("NEBULA_PASSWORD", config.password)
    
    config = Config()
    config.max_connection_pool_size = 2

    # create connection pool:
    connection_pool = ConnectionPool()
    connection_pool.init([(host, int(port))], config)

    # get session from the pool:
    client = connection_pool.get_session(user, password)
    return client


def query(query: str):
    client = load_client()
    client.execute(query)


def create_space(name: str):
    return query(
        f"""
        CREATE SPACE {name}(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);
        USE {name};
        CREATE TAG entity(name string);
        CREATE EDGE relationship(name string);
        CREATE TAG INDEX entity_index ON entity(name(256));
        """
    )
