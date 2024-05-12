import logging
import os
import sys

import openai
import streamlit as st
from dotenv import load_dotenv
from llama_index.core.settings import Settings
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.llms.openai import OpenAI

from config import load_graph_config
from util.graph import load_graph_index_from_config
from util.graph import load_graph_query_engine

# prepare environment:
load_dotenv()

# prepare logger:
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
)

# prepare model:
openai.api_key = os.environ["OPENAI_API_KEY"]

# load config:
config = load_graph_config(
    index_name="policies_aa",
    persist_dir="./data/index/policies/american_airlines/",
)

# @st.cache_resource
def load_index(name: str):
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
    # config = load_graph_config(
        # index_name=name,
        # persist_dir=f"./data/index/policies/{name}/",
        # index_name="policies_aa",
        # persist_dir="./data/index/policies/american_airlines/",
    # )
    return load_graph_index_from_config(config)


# initialize page:
st.header("Chat with the Airlines 💬 ✈️")

# initialize message history:
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Ask me a question about Airline travel!"
    }]

# load index: 
index = load_index("american_airlines")

# 3.4. Create the chat engine
# condense question mode because it always queries the knowledge base when generating a response. 
# this mode is optimal because you want the model to keep its answers specific to the knowledge base.
# from: https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/
chat_engine = index.as_chat_engine(
    # chat_mode="condense_question",
    retriever_mode=config.retriever_mode,
    verbose=config.verbose,
    response_mode=config.response_mode,
    graph_store_query_depth=config.graph_store_query_depth,
)

# 3.5. Prompt for user input and display message history
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"], avatar="🧑‍✈️"):
        st.write(message["content"])

# 3.6. Pass query to chat engine and display response:
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="🧑‍✈️"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.markdown(response.response.replace("$", "\$"))
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
