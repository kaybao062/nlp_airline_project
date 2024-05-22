import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from util.dataset import load_trend_data, load_rate_data

df = load_trend_data()
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
    col2, col3 = st.columns([1, 1], gap = 'medium')
    # Left column for submitting reviews

    # Right column for visualization
    with col2:
        with st.container(height=100):
            st.title(f"Trends of {selected_airline}")
        # Add your visualization code here
        chart_data = df[df['Airline'] == selected_airline]
        #date = df[df['Airline'] == selected_airline]['Review Date']
        with st.container(height=300):
            st.line_chart(chart_data, x = 'Year', y = 'Overall Rating')
        with st.container(height=300):
            st.line_chart(chart_data, x = 'Year', y = 'Score')
        with st.container(height=300):
            st.line_chart(chart_data, x = 'Year', y = 'Seat Comfort')
        with st.container(height=300):
            st.line_chart(chart_data, x = 'Year', y = 'Staff Service')
        with st.container(height=300):
            st.line_chart(chart_data, x = 'Year', y = 'Food & Beverages')
        with st.container(height=300):
            st.line_chart(chart_data, x = 'Year', y = 'Inflight Entertainment')
        with st.container(height=300):
            st.line_chart(chart_data, x = 'Year', y = 'Value For Money')


    with col3:
        with st.container(height=100):
            st.title(f"Rating of {selected_airline}")
        chart_df = df_rate[df_rate['Airline'] == selected_airline]
        # chart_df = chart_df.fillna(0)
        with st.container(height=800):
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
            c = (alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("percentage_start:Q"),
                    x2="percentage_end:Q",
                    y=alt.Y("Category:N").axis(alt.Axis(title="Rating", offset=5, ticks=False, minExtent=60, domain=False)),
                    color=alt.Color("Rating Type:N").title("Rating").scale(color_scale)
                    )
                .properties(
                    width=600,
                    height=400,
                    title='Passenger Rating the Airline'
        ))
            st.altair_chart(c, use_container_width=True)
            st.dataframe(chart_df)
  


# Call the main function to run the Streamlit app
if __name__ == "__main__":
    main()


