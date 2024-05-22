import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_csv('data/clean/airline_trend.csv')
df_index = pd.read_csv('data/clean/airline_index.csv')

# create the dataframe to append review data
reviews_df = pd.DataFrame(columns=['Review', 'Sentiment Class', 'Score'])
# Define the "How to Use" message
how_to_use = """
**How to Use**
1. Enter text in the text area
2. Click the 'Analyze' button to get the predicted sentiment of the text
"""
st.set_page_config(layout="centered")
# Functions
def main():
    # Add a sidebar

    # Add other filters as needed

    # Display the selected airline
    # st.write("Selected Airline:", selected_airline)

    st.sidebar.title("Submit your review")
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

    st.title("Submit Reviews for an Airline ✈️" )
    st.subheader("Leave your comment here")
    st.write('Have something to say about an airline? Leave your comment here. Your sentiment will be analyzed in a second, and we will store it to inform more passengers. ')
    reviews_df = pd.DataFrame(columns=['Airline', 'Review', 'Sentiment Class', 'Score'])

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
        # confidence_level = np.max(outputs.logits.detach().numpy())

        # Print the predicted class and associated score
        # st.write(f"Predicted class: {predicted_class}, Score: {score:.3f}, Confidence Level: {confidence_level:.2f}")

        # Emoji
        st.success("Review submitted successfully!")

        if predicted_class == 1:
            st.markdown("Sentiment: Positive 😊")
            st.write(f'We are glad that you enjoyed your trip with {selected_airline}. Your feed back is stored and will be communicated to the company soon. ')
        else:
            st.markdown("Sentiment: Negative 😠")
            st.write(f'We are sorry that you had a bad experience in your trip with {selected_airline}. Your feed back is stored and will be communicated to the company soon. ')
        # Create the results DataFrame
        results_df = pd.DataFrame({
            'Airline': selected_airline, 
            'Review': raw_text, 
            "Sentiment Class": ['Positive', 'Negative'],
            "Score": [positive_score, negative_score]
        })

        reviews_df = pd.concat([reviews_df, results_df])
        reviews_df.to_csv('data/clean/reviews.csv', index=False)

        # Create the Altair chart

        # Display the chart
        fig = px.pie(results_df, values='Score', 
                     names='Sentiment Class', 
                     title="Your sentiment score",
                     color = 'Sentiment Class', 
                     color_discrete_map={'Negative':'lightcyan',
                                        'Positive':'royalblue'})
        st.plotly_chart(fig, theme=None)

        # st.write(results_df)


# Call the main function to run the Streamlit app
if __name__ == "__main__":
    main()


