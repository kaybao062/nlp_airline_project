import logging
import os
import sys
from enum import Enum

import openai
import altair as alt
import streamlit as st
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.settings import Settings
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.tools import (
    FunctionTool,
    QueryEngineTool,
    ToolMetadata,
)
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.llms.openai import OpenAI
from pydantic import Field
from slugify import slugify

from config import load_graph_config
from util.dataset import load_policies, load_trend_data, load_rate_data
from util.graph import load_graph_index_from_config
from util.multi_tenant import extract_tenants
from util.vector import load_vector_index

# prepare environment:
load_dotenv()

# prepare logger:
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
)

# load datasets:
trend_df = load_trend_data()
rate_df = load_rate_data()


class AspectEnum(Enum):
    SEAT_COMFORT = 'Seat Comfort'
    STAFF_SERVICE = 'Staff Service'
    FOOD_BEVERAGE = 'Food & Beverages'
    INFLIGHT_ENTERTAINMENT = 'Inflight Entertainment'
    VALUE_FOR_MONEY = 'Value For Money'
    OVERALL_RATINGS = 'Overall Rating'    


@st.cache_resource
def load_graph_index(name: str):
    config = load_graph_config(
        index_name=name,
        # persist_dir=f"./data/index/policies_aa/policies-sgs-2/{name}/",
        persist_dir=f"./data/index/policies-sgs-2/{name}/",
    )
    return load_graph_index_from_config(config)


@st.cache_resource
def load_review_index():
    return load_vector_index()


def load_tenants():
    documents = load_policies()
    airlines = extract_tenants(documents, tenant_key="airline")
    return airlines


@st.cache_resource
def load_chat_memory(user: str):
    # add memory:
    chat_store = SimpleChatStore()
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key=user,
    )
    return memory


# 6 funcs
def plot_airline_trends(airline: str = None, aspect: AspectEnum = None):
    """Useful for understanding consumer sentiment on airlines over time."""
    ## How to let plot a certain aspect?
    if airline:
        chart_data = trend_df[trend_df['Airline'] == airline]
    else:
        chart_data = trend_df
    st.line_chart(chart_data, x = 'Year', y = aspect)


def plot_airline_rate(airline: str, aspect: AspectEnum = None):
    """Useful for understanding consumer sentiment on airlines over time."""
    ## How to let plot a certain aspect?
    if aspect:
        chart_data = rate_df[(rate_df['Airline'] == airline) & (rate_df['Category'] == aspect)]
        height = 200
        title = f"Passenger Rating of {aspect} on {airline}"
    else:
        chart_data = rate_df[rate_df['Airline'] == airline]
        height = 400
        title = f"Passenger Rating of {airline}"
    
    color_scale = alt.Scale(
    domain=[
        "Poor",
        "Not good",
        "Neutral",
        "Good" ,
        "Very Good" 
    ],
    range=["#c30d24", "#f3a583", "#cccccc", "#94c6da", "#1770ab"],
)
    c = (alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("percentage_start:Q"),
            x2="percentage_end:Q",
            y=alt.Y("Category:N").axis(alt.Axis(title="Rating", offset=5, ticks=False, minExtent=60, domain=False)),
            color=alt.Color("Rating Type:N").title("Rating").scale(color_scale)
            )
        .properties(
            width=600,
            height=height,
            # title='Passenger Rating the Airline'
            title=title,
))
    st.altair_chart(c, use_container_width=True)


def plot_compare_airline_rate(aspect: AspectEnum = None, airlines: list = []):
    """Useful for understanding consumer sentiment on airlines over time."""
    ## How to let plot a certain aspect?
    chart_data = rate_df[(rate_df['Category'] == aspect)][rate_df['Airline'].isin(airlines)]
    height = 400
    title = f"Passenger Rating of {aspect}"
    
    color_scale = alt.Scale(
    domain=[
        "Poor",
        "Not good",
        "Neutral",
        "Good" ,
        "Very Good" 
    ],
    range=["#c30d24", "#f3a583", "#cccccc", "#94c6da", "#1770ab"],
)
    c = (alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("percentage_start:Q"),
            x2="percentage_end:Q",
            y=alt.Y("Airline:N").axis(alt.Axis(title="Rating", offset=5, ticks=False, minExtent=60, domain=False)),
            color=alt.Color("Rating Type:N").title("Rating").scale(color_scale)
        )
        .properties(
            width=600,
            height=height,
            title=title,
))
    st.altair_chart(c, use_container_width=True)


def inquire_about_airline(airline: str):
    """Useful for answering questions about an airline. Particularly helpful
    for summarizing consumer sentiment toward a specific airline.
    """
    # st.write(f"inquire_about_airline: {airline=}")
    # The chart displayed here: 1 airline multiple aspects
    plot_airline_rate(airline)
    # The text displayed here
    review_index = load_review_index()
    query_engine = review_index.as_query_engine(
        filter=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="Airline",
                    value=airline,
                )
            ]
        ),
        hybrid_top_k=10,
        similarity_top_k=10,
    )
    response = query_engine.query(
        f"Tell me about {airline}. Be as thorough as possible."
    )
    return response


def describe_airline_sentiment_over_time(airline: str, aspect: AspectEnum):
    """Useful for understanding consumer sentiment on airlines over time."""
    # plot = st.markdown(f"{airline=} {aspect=}")
    # The chart displayed here: 1 airline 1 aspect trend
    plot_airline_trends(airline, aspect)
    # The text displayed here
    # st.write(f"describe_airline_sentiment_over_time: {airline=} {aspect=}")
    review_index = load_review_index()
    query_engine = review_index.as_query_engine(
        filter=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="Airline",
                    value=airline,
                )
            ]
        ),
        hybrid_top_k=10,
        similarity_top_k=10,
    )
    response = query_engine.query(
        f"Tell me about {aspect} on {airline}. Be as thorough as possible."
    )
    return response


def compare_airlines_by_aspect(aspect: AspectEnum):
    """Useful for answering questions about comparing airlines by a
    specific aspect, such as seat comfort, staff service, food and beverage,
    inflight entertainment, value for money, and overall ratings. 
    """
    # st.write(f"compare_airlines_by_aspect: {aspect=}")
    review_index = load_review_index()
    response = review_index.as_query_engine(
        hybrid_top_k=10,
        similarity_top_k=10,
    ).query(
        f"Compare airlines by {aspect}. Be as thorough as possible."
    )
    chart_airlines = []
    for node in response.source_nodes:
        chart_airlines.append(node.metadata["Airline"])
    chart_airlines = list(set(chart_airlines))
    # The chart displayed here: multiple airline 1 aspect rate
    # st.subheader(f'Plot for {aspect} of {chart_airlines[0]}')
    plot_compare_airline_rate(aspect, airlines=chart_airlines)
    return response


def inquire_about_aspect_on_airline(airline: str, aspect: AspectEnum):
    """Useful for answering questions about consumer sentiment on an
    airline about a specific aspect, such as seat comfort, staff service,
    food and beverage, inflight entertainment, value for money. 
    """
    # st.write(f"inquire_about_aspect_on_airline: {airline=} {aspect=}")
    review_index = load_review_index()
    query_engine = review_index.as_query_engine(
        filter=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="Airline",
                    value=airline,
                )
            ]
        ),
        hybrid_top_k=10,
        similarity_top_k=10,
    )
    response = query_engine.query(
        f"Tell me about {aspect} on {airline}. Describe only this aspect and be as thorough as possible."
    )
    # The chart displayed here: 1 airline 1 aspect rate
    plot_airline_rate(airline, aspect)
    return response


@st.cache_resource
def load_agent():
    # prepare model:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        # num_beams=4,
        # additional_kwargs={
        #    "extra_body": {
        #        "use_beam_search": True,
        #        "best_of": 3,
        #    }
        # }
    )
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

    airlines = ["American Airlines", "Air France", "Delta Air lines"]
    # airlines = load_tenants()
    tools = []

    # load review tools:
    review_index = load_vector_index()

    # create tools for each airline:
    for airline in airlines:
        airline_key = slugify(airline, separator="_")
        logging.info(f"Creating tools for {airline}")

        # load policy tools:
        tools += [
            QueryEngineTool(
                query_engine=load_graph_index(airline_key).as_query_engine(
                    response_mode="tree_summarize",
                    # hybrid_top_k=10,
                    # similarity_top_k=5,
                ),
                metadata=ToolMetadata(
                    name=f"{airline_key}_policies",
                    description=(
                        f"Provides information about {airline} policies. "
                        "Useful for answering questions about restricted items, checked bags, mishandled bags, "
                        "special notices, delayed bags, delayed or cancelled trips, refunds, what you can fly with, excess baggage guidelines, "
                        f"terms and conditions, oversized or bulky baggage, partner airlines, receipts, and all other policy questions related to {airline}. "
                        "Use a detailed plain text question as input to the tool."
                    ),
                ),
            ),
        ]

        # reviews:
        tools += [
            QueryEngineTool(
                query_engine=review_index.as_query_engine(
                    filter=MetadataFilters(
                        filters=[
                            MetadataFilter(
                                key="Airline",
                                value=airline,
                            )
                        ],
                    ),
                    hybrid_top_k=5,
                    similarity_top_k=5,
                ),
                metadata=ToolMetadata(
                    name=f"{airline_key}_reviews",
                    description=(
                        f"Provides summaries about consumer reports, customer reviews and sentiment toward {airline}. "
                        "Use a detailed plain text question as input to the tool."
                    ),
                ),
            )
        ]

    tools += [
        FunctionTool.from_defaults(inquire_about_airline),
        FunctionTool.from_defaults(inquire_about_aspect_on_airline),
        FunctionTool.from_defaults(compare_airlines_by_aspect),
        FunctionTool.from_defaults(describe_airline_sentiment_over_time),
    ]


    # give agent tools:
    agent = OpenAIAgent.from_tools(
        memory=load_chat_memory("user"),
        tools=tools,
        verbose=True,
        # the following system prompt makes it lie sadly
        # system_prompt="Without using your prior knowledge, and only using the given context, answer the question while being as thorough as possible.",
        # more parameters: https://docs.llamaindex.ai/en/stable/api_reference/agent/openai/#llama_index.agent.openai.OpenAIAgent.from_tools
        # callback_manager = None
        max_function_calls=1,
    )
    return agent



# initialize page:
st.set_page_config(layout="centered")
st.header("Chat with our bot 💬 ✈️")
# initialize message history:
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Ask me a question about Airline travel!"
    }]

st.sidebar.title("Ask about airlines")
st.sidebar.markdown('Our chatbot knows about how passengers reviewed the airline. Inquire about overall rating, seat comfort, staff service, food & baverage, inflight entertainment, value for money for any airlines you are interesting in.')
st.sidebar.markdown('The chatbot can also answer questions concerningpolicies of baggage, change reservation, refund, and delayed/canceled flights. ')

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
    avatar = {
        "assistant": "🧑‍✈️",
        "user": "✈️",
    }
    with st.chat_message(message["role"], avatar=avatar[message["role"]]):
        st.markdown(message["content"].replace("$", "\$"))

# 3.6. Pass query to chat engine and display response:
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="🧑‍✈️"):
        with st.spinner("Thinking..."):
            try:
                response = agent.chat(prompt)
                logging.info(response)
                st.markdown(response.response.replace("$", "\$"))
                # Add response to message history
                message = {"role": "assistant", "content": response.response}
            except ValueError as err:
                logging.error(err)
                # Add response to message history
                st.markdown("We couldn't find that one...")
                message = {"role": "assistant", "content": "We don't know!"}
            st.session_state.messages.append(message)
