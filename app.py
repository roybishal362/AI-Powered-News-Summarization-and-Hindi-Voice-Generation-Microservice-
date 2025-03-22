################### app.py ######################
import os
import streamlit as st
import requests
import json
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

st.set_page_config(layout="wide", page_title="News Sentiment Analyzer")

# List of popular companies for the dropdown
POPULAR_COMPANIES = [
    "Tesla", "Apple", "Microsoft", "Google", "Amazon", 
    "Meta", "Netflix", "NVIDIA", "Intel", "AMD", 
    "Coca-Cola", "Pepsi", "Walmart", "Target", "Reliance Industries",
    "Tata Motors", "Infosys", "TCS", "Wipro", "HDFC Bank"
]

def get_download_link(json_data, company_name):
    """Generate a download link for the JSON data"""
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{company_name}_{timestamp}.json"
    
    # Convert to JSON string
    json_str = json.dumps(json_data, indent=2)
    
    # Create download link
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">Download JSON Results</a>'
    return href

def main():
    st.title("Company News Sentiment Analyzer")
    
    st.write("""
    This application extracts news about a company, analyzes sentiment, and provides a summary with Hindi text-to-speech output.
    """)
    
    # Selection method - dropdown or manual input
    selection_method = st.radio(
        "Choose how to select a company:",
        ["Select from popular companies", "Enter company name manually"],
        key="selection_method"
    )
    
    # Initialize company_name variable
    company_name = None
    
    # Based on selection method, show appropriate input
    if selection_method == "Select from popular companies":
        company_name = st.selectbox(
            "Select a company:",
            options=POPULAR_COMPANIES,
            key="company_dropdown"
        )
    else:
        company_name = st.text_input(
            "Enter a company name:",
            value="Tesla",
            key="company_manual_input"
        )
    
    # Make API call when the analyze button is clicked
    if st.button("Analyze", key="analyze_button"):
        if company_name:
            with st.spinner("Fetching and analyzing news articles... This may take a minute."):
                try:
                    # Make API call
                    #API_URL = os.environ.get("API_URL", "http://localhost:5000")
                    # response = requests.post(
                    #     "http://localhost:5000/api/news-analysis",
                    #     json={"company_name": company_name},
                    #     timeout=600  # Increased timeout for complex analysis
                    # )
                    API_URL = os.environ.get("API_URL", "")
                    response = requests.post(
                        "/api/news-analysis",  # Use relative URL
                        json={"company_name": company_name},
                        timeout=720
                    )
                    if response.status_code == 200:
                        data = response.json()
                        display_results(data)
                        
                        # Add download button for JSON results
                        st.markdown(get_download_link(data, company_name), unsafe_allow_html=True)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                        st.info("If you're seeing a connection error, please ensure the backend API is running on port 5000.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("If you're seeing a connection error, please ensure the backend API is running on port 5000.")
        else:
            st.warning("Please enter or select a company name.")

def display_results(data):
    """Display the results in a nicely formatted way."""
    st.header(f"News Analysis for {data['Company']}")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Articles", "Sentiment Analysis", "Comparative Analysis", "Hindi TTS"])
    
    with tab1:
        st.subheader("News Articles")
        for i, article in enumerate(data['Articles']):
            with st.expander(f"{i+1}. {article['Title']}"):
                st.write(f"**Summary:** {article['Summary']}")
                st.write(f"**Sentiment:** {article['Sentiment']}")
                st.write(f"**Topics:** {', '.join(article['Topics'])}")
                if article['URL'] != "#":
                    st.write(f"**Source:** [Link to Article]({article['URL']})")
    
    with tab2:
        st.subheader("Sentiment Analysis")
        
        # Create a sentiment distribution dataframe
        sentiment_dist = data['Comparative Sentiment Score']['Sentiment Distribution']
        sentiment_df = pd.DataFrame({
            'Sentiment': list(sentiment_dist.keys()),
            'Count': list(sentiment_dist.values())
        })
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Sentiment', y='Count', data=sentiment_df, palette={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}, ax=ax)
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)
        
        # Display the final sentiment
        st.subheader("Final Sentiment")
        st.write(data['Final Sentiment Analysis'])
    
    with tab3:
        st.subheader("Comparative Analysis")
        
        # Display coverage differences
        st.write("#### Coverage Differences")
        for diff in data['Comparative Sentiment Score'].get('Coverage Differences', []):
            st.write(f"**Comparison:** {diff['Comparison']}")
            st.write(f"**Impact:** {diff['Impact']}")
            st.write("---")
        
        # Display topic overlap
        st.write("#### Topic Analysis")
        topic_data = data['Comparative Sentiment Score'].get('Topic Overlap', {})
        
        if 'Common Topics' in topic_data:
            st.write(f"**Common Topics:** {', '.join(topic_data['Common Topics'])}")
        
        if 'Unique Topics' in topic_data:
            st.write(f"**Unique Topics:** {', '.join(topic_data['Unique Topics'])}")
    
    with tab4:
        st.subheader("Hindi Translation and Audio")
        
        # Display Hindi text
        st.write("#### Hindi Translation")
        st.write(data['Hindi Translation'])
        
        # Display audio player
        st.write("#### Audio")
        if data['Audio']:
            audio_bytes = base64.b64decode(data['Audio'])
            st.audio(audio_bytes, format='audio/mp3')
        else:
            st.write("Audio not available")

if __name__ == "__main__":
    main()