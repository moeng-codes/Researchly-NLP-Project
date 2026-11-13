Instructions for running the Researchly application.

# Researchly Application Deployment

## Overview
Researchly is a Streamlit application that classifies research abstracts as either artificial intelligence-related or classical research using a trained Random Forest classifier.

## Requirements
- Python 3
- Streamlit
- Scikit-learn
- Joblib

## Installation
1. Download or clone the GitHub repository.
2. Navigate to the `deployment` folder.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Start the application:

   `streamlit run app.py`

5. Open the local URL displayed in the terminal.

## Model Files
The application uses:
- `researchly_random_forest.pkl`
- `researchly_tfidf.pkl`
- `researchly_threshold.pkl`

## Functionality
Users enter a research abstract, click **Classify Abstract**, and receive a predicted research category and AI classification score.
