# Import libraries for the web application
import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Configure the application page
st.set_page_config(
    page_title="Researchly Abstract Classifier",
    page_icon="pi",
    layout="centered"
)

# Load the trained model and feature extraction tools
model = joblib.load("researchly_random_forest.pkl")
vectorizer = joblib.load("researchly_tfidf.pkl")
threshold = joblib.load("researchly_threshold.pkl")

# Define stopwords for text preprocessing
stop_words = set(ENGLISH_STOP_WORDS)
stop_words = stop_words.difference(
    {"no", "not", "nor", "without"}
)

# Clean the research abstract using the training preprocessing steps
def clean_text(text):

    text = text.lower()

    # Remove URLs and HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove punctuation and extra spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove common stopwords
    words = text.split()
    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)

# Display the application heading
st.title("Researchly Research Abstract Classifier")

st.write(
    "This application classifies research abstracts "
    "as either AI-related or classical research."
)

# Create a text input area
abstract_text = st.text_area(
    "Enter research abstract:",
    height=200,
    placeholder="Paste your research abstract here..."
)

# Classify the abstract when the button is clicked
if st.button("Classify Abstract"):

    if not abstract_text.strip():
        st.warning("Please enter a research abstract.")

    else:
        # Preprocess the entered abstract
        cleaned_abstract = clean_text(abstract_text)

        if not cleaned_abstract:
            st.warning("Please enter an abstract containing meaningful words.")

        else:
            # Convert the abstract into TF-IDF features
            abstract_features = vectorizer.transform(
                [cleaned_abstract]
            )

            # Predict class probabilities
            probabilities = model.predict_proba(
                abstract_features
            )[0]

            # Identify the AI class probability
            ai_index = list(model.classes_).index("ai")
            ai_probability = probabilities[ai_index]

            # Apply the optimised classification threshold
            if ai_probability >= threshold:
                prediction = "artificial intelligence"
            else:
                prediction = "classical research"

            # Display classification results
            st.subheader("Prediction")

            st.success(
                f"The abstract is classified as: {prediction}"
            )

            st.write(
                f"AI classification score: {ai_probability:.2%}"
            )
