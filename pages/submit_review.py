import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_csv('data/clean/airline_trend.csv')
df_index = pd.read_csv('data/clean/airline_index.csv')
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


# Call the main function to run the Streamlit app
if __name__ == "__main__":
    main()


