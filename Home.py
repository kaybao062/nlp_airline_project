import logging
import os
import sys

import openai
import streamlit as st
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.settings import Settings
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.tools import (
    QueryEngineTool,
    ToolMetadata,
)
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.llms.openai import OpenAI



from config import load_graph_config
from util.graph import load_graph_index_from_config
from util.graph import load_graph_query_engine
from util.vector import load_vector_index

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

# @st.cache_resource
def load_graph_index(name: str):
    config = load_graph_config(
        index_name=name,
        persist_dir=f"./data/index/policies/{name}/",
    )
    return load_graph_index_from_config(config)


def load_agent():
    tools = []

    # load policy tools:
    tools += [
        QueryEngineTool(
            query_engine=load_graph_index("american_airlines").as_query_engine(),
            metadata=ToolMetadata(
                name="american_airlines_policies",
                description=(
                    "Provides information about American Airlines policies. "
                    "Useful for answering questions about restricted items, checked bags, mishandled bags, "
                    "special notices, delayed bags, delayed or cancelled trips, refunds, what you can fly with, excess baggage guidelines, "
                    "terms and conditions, oversized or bulky baggage, partner airlines, receipts, and all other policy questions related to American Airlines. "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),
    ]

    # load review tools:
    review_index = load_vector_index()
    tools += [
        QueryEngineTool(
            query_engine=review_index.as_query_engine(
                filter=MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="Airline",
                            value="American Airlines",
                        )
                    ]
                )
            ),
            metadata=ToolMetadata(
                name="american_airlines_reviews",
                description=(
                    "Provides summaries about consumer reports, customer reviews and sentiment toward American Airlines. "
                    "Useful for answering questions about seat comfort, crew staff service, food and beverage, inflight entertainment, "
                    "value for money, and overall ratings. "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        )
    ]

    # give agent tools:
    agent = OpenAIAgent.from_tools(
        tools=tools,
        verbose=True,
        system_prompt="Be as thorough as possible in your answer.",
    )
    return agent



# initialize page:
st.header("Chat with the Airlines 💬 ✈️")

# initialize message history:
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Ask me a question about Airline travel!"
    }]


# 3.4. Create the chat engine
# condense question mode because it always queries the knowledge base when generating a response. 
# this mode is optimal because you want the model to keep its answers specific to the knowledge base.
# from: https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/
# chat_engine = index.as_chat_engine(
#    # chat_mode="condense_question",
#    retriever_mode=config.retriever_mode,
#    verbose=config.verbose,
#    response_mode=config.response_mode,
#    graph_store_query_depth=config.graph_store_query_depth,
#)
agent = load_agent()

# 3.5. Prompt for user input and display message history
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"], avatar="🧑‍✈️"):
        st.write(message["content"].replace("$", "\$"))

# 3.6. Pass query to chat engine and display response:
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="🧑‍✈️"):
        with st.spinner("Thinking..."):
            response = agent.chat(prompt)
            logging.info(response)
            st.markdown(response.response.replace("$", "\$"))
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
