############ Api.py #################
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
from utils import (
    get_news_articles, 
    analyze_sentiment, 
    setup_llm, 
    extract_key_topics, 
    create_article_summary, 
    conduct_comparative_analysis,
    create_final_sentiment_summary,
    generate_hindi_tts
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/news-analysis', methods=['POST'])
def analyze_news():
    """API endpoint to analyze news for a given company."""
    try:
        data = request.json
        company_name = data.get('company_name')
        
        if not company_name:
            return jsonify({
                'error': 'Company name is required'
            }), 400
        
        logger.info(f"Received request for company: {company_name}")
        
        # Set up LLM
        logger.info("Setting up LLM...")
        llm = setup_llm()
        
        # Get news articles
        logger.info("Fetching news articles...")
        articles = get_news_articles(company_name)
        logger.info(f"Found {len(articles)} articles")
        
        # Process each article
        processed_articles = []
        for i, article in enumerate(articles):
            logger.info(f"Processing article {i+1}/{len(articles)}")
            
            # Add company name to the article
            article['company_name'] = company_name
            
            # Analyze sentiment
            logger.info(f"Analyzing sentiment for article {i+1}")
            article['sentiment'] = analyze_sentiment(article['content'])
            
            # Create summary
            logger.info(f"Creating summary for article {i+1}")
            article['summary'] = create_article_summary(article, llm)
            
            # Extract topics
            logger.info(f"Extracting topics for article {i+1}")
            article['topics'] = extract_key_topics(article, llm)
            
            # Add to processed articles
            processed_articles.append({
                'Title': article['title'],
                'Summary': article['summary'],
                'Sentiment': article['sentiment'],
                'Topics': article['topics'],
                'URL': article['url']
            })
        
        # Conduct comparative analysis
        logger.info("Conducting comparative analysis...")
        comparative_analysis = conduct_comparative_analysis(articles, llm)
        
        # Create final sentiment summary
        logger.info("Creating final sentiment summary...")
        final_sentiment = create_final_sentiment_summary(articles, comparative_analysis, llm)
        
        # Generate Hindi TTS
        logger.info("Generating Hindi TTS...")
        tts_result = generate_hindi_tts(final_sentiment)
        
        # Create response
        response = {
            'Company': company_name,
            'Articles': processed_articles,
            'Comparative Sentiment Score': comparative_analysis,
            'Final Sentiment Analysis': final_sentiment,
            'Hindi Translation': tts_result['hindi_text'],
            'Audio': tts_result['audio_data']
        }
        
        logger.info("Request processed successfully")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({
            'error': 'An error occurred while processing your request',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)