'''
This file defines functions for reading and writing document embeddings to a postgres database's vector store.
'''
from typing import Any

# 导入数据库配置
from config import config, DB_CONFIG, VECTOR_STORE_CONFIG

# vector store
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, Document

# postgresql support
import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore

# Node
from llama_index.core.schema import TextNode

# other dependencies
import json
import uuid

def connect_db() -> Any:
    conn = psycopg2.connect(
        host=DB_CONFIG["host"],
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    conn.autocommit = True

    return conn

def write_vector_store_from_chunks(chunks, embedder, db_table_name: str):
    vector_store = PGVectorStore.from_params(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        table_name=db_table_name,
        embed_dim=VECTOR_STORE_CONFIG["embed_dim"]
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    nodes = [TextNode(text=t, id_=str(uuid.uuid1())) for t in chunks]
    
    # build index
    index = VectorStoreIndex(nodes=nodes, embed_model=embedder, storage_context=storage_context, show_progress=True)
    return index

def get_vector_store(embedder, db_table_name):
    vector_store = PGVectorStore.from_params(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        table_name=db_table_name,
        embed_dim=VECTOR_STORE_CONFIG["embed_dim"]
    )

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embedder)

    return index