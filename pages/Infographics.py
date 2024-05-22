import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from util.dataset import load_trend_data, load_rate_data
from streamlit_float import *

df = load_trend_data()
# year dtypes
# df['Year'] = df['Year'].astype('str')
df_rate = load_rate_data()
# Define the "How to Use" message
how_to_use = """
**How to Use**
1. Enter text in the text area
2. Click the 'Analyze' button to get the predicted sentiment of the text
"""

# define likert chart function
def plot_airline_rate(airline: str):
    """Useful for understanding consumer sentiment on airlines over time."""
    ## How to let plot a certain aspect?
    
    chart_df = df_rate[df_rate['Airline'] == airline]
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


    # c = (
    #     alt.Chart(chart_data).mark_bar().encode(
    #         x=alt.X("Start Percentage:Q"),
    #         x2="End Percentage:Q",
    #         y=alt.Y("Category:N").axis(alt.Axis(title="Rating", offset=5, ticks=False, minExtent=60, domain=False)),
    #         color=alt.Color("Rating Type:N").title("Rating").scale(color_scale)
    #     ).properties(
    #         width=600,
    #         height=400,
    #         title='Passenger Rating the Airline'
    #     )).interactive()
    st.dataframe(chart_df)
    c = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Start Percentage:Q"),
        x2="End Percentage:Q",
        y=alt.Y("Category:N").axis(alt.Axis(title="Rating", offset=5, ticks=False, minExtent=60, domain=False)),
        color=alt.Color("Rating Type:N").title("Rating").scale(color_scale)
    ).properties(
        width=600,
        height=400,
        title='Passenger Rating the Airline'
    )
    
    return c
    
    # st.altair_chart(c, use_container_width=True)
    # st.dataframe(chart_data)
    # return st.markdown(f"Plot {airline} rate chart!!!")


st.set_page_config(layout="wide")
# Functions
def main():
    # Add a sidebar

    # Add other filters as needed

    # Display the selected airline
    # st.write("Selected Airline:", selected_airline)

    st.sidebar.title("Airline Dashboard")
    st.sidebar.markdown("Get the information about the trend and rating of different aspects of an airline! ")

# Add a dropdown menu for selecting airline
    selected_airline = st.sidebar.selectbox("Select Airline", 
                                        ['Air France',
                                            'Air India',
                                            'Air New Zealand',
                                            'Air Tahiti Nui',
                                            'Alaska Airlines',
                                            'All Nippon Airways',
                                            'American Airlines',
                                            'Cathay Pacific Airways',
                                            'Copa Airlines',
                                            'Delta Airlines',
                                            'EVA Air',
                                            'Eithad Airways',
                                            'Emirates',
                                            'Frontier Airlines',
                                            'Hawaiian Airlines',
                                            'Japan Airlines',
                                            'Jetblue Airways',
                                            'Korean Air',
                                            'Qatar Airways',
                                            'Singapore Airlines',
                                            'Skywest Airlines',
                                            'Southwest Airlines',
                                            'Spirit Airlines',
                                            'Turkish Airlines',
                                            'United Airlines',
                                            'Virgin Atlantic Airways'])

    # Create two columns in the main panel
    # col1, col2 = st.columns([1, 1], gap = 'medium')
    col1,col2 = st.columns(2)

    # Left column for submitting reviews
    with col1: 
        # header_container = st.container(height=100)
        # with header_container:
        st.title(f"Trends of {selected_airline}")
        selected_year = st.slider(
            "Select years range",
            2014, 2024, (2014, 2024))
    # Right column for visualization
        chart_data = df[(df['Airline'] == selected_airline) & (df['Year'].astype(int) >= selected_year[0]) & (df['Year'].astype(int) <= selected_year[1])]
        col3, col4 = col1.columns([1,1], gap = 'small')
        with col3:
            with st.container(height=200):
                chart = alt.Chart(chart_data).mark_line(color='red').encode(
                x=alt.X('Year'),  # Ordinal scale for years without axis labels
                y=alt.Y('Overall Rating:Q', axis=None)  # Quantitative scale for values
            ).properties(
                width=300,  # Set the width of the chart
                height=200,  # Set the height of the chart
                title='Overall Rating'
            ).interactive()

            # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True, theme=None)
            with st.container(height=200):
                chart = alt.Chart(chart_data).mark_line(color='#9467bd').encode(
                x=alt.X('Year'),  # Ordinal scale for years without axis labels
                y=alt.Y('Staff Service:Q', axis=None)  # Quantitative scale for values
            ).properties(
                width=300,  # Set the width of the chart
                height=200,  # Set the height of the chart
                title='Staff Service'
            ).interactive()

            # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True, theme=None)
            with st.container(height=200):
                chart = alt.Chart(chart_data).mark_line(color='#9467bd').encode(
                x=alt.X('Year'),  # Ordinal scale for years without axis labels
                y=alt.Y('Seat Comfort:Q', axis=None)  # Quantitative scale for values
            ).properties(
                width=300,  # Set the width of the chart
                height=200,  # Set the height of the chart
                title='Seat Comfort'
            ).interactive()

            # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True, theme=None)

        with col4:
            # Add your visualization code here
            #date = df[df['Airline'] == selected_airline]['Review Date']
            with st.container(height=200):
                chart = alt.Chart(chart_data).mark_line(color='#9467bd').encode(
                x=alt.X('Year'),  # Ordinal scale for years without axis labels
                y=alt.Y('Food & Beverages:Q', axis=None)  # Quantitative scale for values
            ).properties(
                width=300,  # Set the width of the chart
                height=200,  # Set the height of the chart
                title='Food & Beverages'
            ).interactive()

            # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True, theme=None)
            with st.container(height=200):
                chart = alt.Chart(chart_data).mark_line(color='#9467bd').encode(
                x=alt.X('Year'),  # Ordinal scale for years without axis labels
                y=alt.Y('Inflight Entertainment:Q', axis=None)  # Quantitative scale for values
            ).properties(
                width=300,  # Set the width of the chart
                height=200,  # Set the height of the chart
                title='Inflight Entertainment'
            ).interactive()

            # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True, theme=None)

            with st.container(height=200):
                chart = alt.Chart(chart_data).mark_line(color='#9467bd').encode(
                x=alt.X('Year'),  # Ordinal scale for years without axis labels
                y=alt.Y('Value For Money:Q', axis=None)  # Quantitative scale for values
            ).properties(
                width=300,  # Set the width of the chart
                height=200,  # Set the height of the chart
                title='Value For Money'
            ).interactive()

            # Display the chart in Streamlit
                st.altair_chart(chart, use_container_width=True, theme=None)


    with col2:
        # with st.container(height=100):
        st.title(f"Rating of {selected_airline}")
        options = st.multiselect(
            "Select aspects you want to compare:",
            ['Seat Comfort','Staff Service','Food & Beverages','Inflight Entertainment','Value For Money','Overall Rating'],
            ['Seat Comfort','Staff Service','Food & Beverages','Inflight Entertainment','Value For Money','Overall Rating'])

        chart_df = df_rate[(df_rate['Airline'] == selected_airline) & (df_rate['Category'].isin(options))]
        # chart_df = chart_df.fillna(0)
        with st.container(height=650):
            color_scale = alt.Scale(
            domain=[
                "Poor",
                "Not good",
                "Neutral",
                "Good" ,
                "Very Good" 
            ],
            range=["#e7ba52", "#a7a7a7", "#aec7e8", "#1f77b4", "#9467bd"],
        )
            c = (alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("percentage_start:Q"),
                    x2="percentage_end:Q",
                    y=alt.Y("Category:N").axis(alt.Axis(title="Rating", offset=5, ticks=False, minExtent=60, domain=False)),
                    color=alt.Color("Rating Type:N").title("Rating").scale(color_scale)
                    )
                .properties(
                    # width=600,
                    height=600,
                    title='Passenger Rating the Airline'
        ))
            st.altair_chart(c, use_container_width=True)
            # st.dataframe(chart_df)
  


# Call the main function to run the Streamlit app
if __name__ == "__main__":
    main()


