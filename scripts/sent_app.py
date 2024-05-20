import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_csv('airline_trend.csv')
df_index = pd.read_csv('airline_index.csv')
# Define the "How to Use" message
how_to_use = """
**How to Use**
1. Enter text in the text area
2. Click the 'Analyze' button to get the predicted sentiment of the text
"""
st.set_page_config(layout="wide")
# Functions
def main():
    # Add a sidebar

    # Add other filters as needed

    # Display the selected airline
    # st.write("Selected Airline:", selected_airline)
    tabs = ["Ask a Chatbot", "Airline Index"]
    selected_tab = st.radio("Select Tab", tabs)

    if selected_tab == "Airline Index":
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
        col2, col3, col1 = st.columns([1, 1, 1], gap = 'medium')
        # Left column for submitting reviews
        with col1:
            st.title("Submit Reviews for an Airline")
            st.subheader("Leave your comment here")

            with st.form(key="nlpForm"):
                raw_text = st.text_area("Enter Text Here")
                submit_button = st.form_submit_button(label="Analyze")

            if submit_button:
                # Display balloons
                st.balloons()

                st.info("Results")
                model_name = "bert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)

                # Tokenize the input text
                inputs = tokenizer(raw_text, return_tensors="pt")

                # Make a forward pass through the model
                outputs = model(**inputs)

                # Get the predicted class and associated score
                predicted_class = outputs.logits.argmax().item()
                score = outputs.logits.softmax(dim=1)[0][predicted_class].item()

                # Compute the scores for all sentiments
                positive_score = outputs.logits.softmax(dim=1)[0][1].item()
                negative_score = outputs.logits.softmax(dim=1)[0][0].item()

                # Compute the confidence level
                confidence_level = np.max(outputs.logits.detach().numpy())

                # Print the predicted class and associated score
                st.write(f"Predicted class: {predicted_class}, Score: {score:.3f}, Confidence Level: {confidence_level:.2f}")

                # Emoji
                if predicted_class == 1:
                    st.markdown("Sentiment: Positive 😊")
                else:
                    st.markdown("Sentiment: Negative 😠")

                # Create the results DataFrame
                results_df = pd.DataFrame({
                    "Sentiment Class": ['Positive', 'Negative'],
                    "Score": [positive_score, negative_score]
                })

                # Create the Altair chart
                chart = alt.Chart(results_df).mark_bar(width=50).encode(
                    x="Sentiment Class",
                    y="Score",
                    color="Sentiment Class"
                )

                # Display the chart
                st.altair_chart(chart, use_container_width=True)
                st.write(results_df)

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
            # For example:
            st.write("Placeholder for visualization")

        with col3:
            with st.container(height=100):
                st.title(f"Overall Index of {selected_airline}")
            chart_data_index = df_index[df_index['Airline'] == selected_airline]
            with st.container(height=300):
                st.metric(label=f"Overall Rating: {chart_data_index['ranked_Overall Rating'].iloc[0]}", value=chart_data_index['Overall Rating'])
            with st.container(height=300):
                st.metric(label=f"Sentiment Rating: {chart_data_index['ranked_Score'].iloc[0]}", value=chart_data_index['Score'])
            with st.container(height=300):
                st.metric(label=f"Seat Comfort: {chart_data_index['ranked_Seat Comfort'].iloc[0]}", value=chart_data_index['Seat Comfort'])
            with st.container(height=300):
                st.metric(label=f"Staff Service: {chart_data_index['ranked_Staff Service'].iloc[0]}", value=chart_data_index['Staff Service'])
            with st.container(height=300):
                st.metric(label=f"Food & Beverages: {chart_data_index['ranked_Food & Beverages'].iloc[0]}", value=chart_data_index['Food & Beverages'])
            with st.container(height=300):
                st.metric(label=f"Inflight Entertainment: {chart_data_index['ranked_Inflight Entertainment'].iloc[0]}", value=chart_data_index['Inflight Entertainment'])
            with st.container(height=300):
                st.metric(label=f"Value For Money: {chart_data_index['ranked_Value For Money'].iloc[0]}", value=chart_data_index['Value For Money'])
    else:
        st.sidebar.title("Inquire about policy of an airline...")

        st.write('Chatbot')
# Call the main function to run the Streamlit app
if __name__ == "__main__":
    main()


