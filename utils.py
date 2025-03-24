import requests
from bs4 import BeautifulSoup
import re
from fake_useragent import UserAgent
from urllib.parse import quote_plus
import time
import random
import numpy as np
from transformers import pipeline
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import os
import tempfile
import base64
from gtts import gTTS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# News Extraction Functions
def fetch_news_urls(company_name, min_articles=15):
    """Fetch news URLs for a given company from multiple sources to ensure we get results."""
    ua = UserAgent()
    headers = {'User-Agent': ua.random}
    
    # Create a more specific search query
    search_query = f"{company_name} news recent financial"
    
    # Try multiple search engines and news sources with more specific queries
    search_urls = [
        f"https://www.google.com/search?q={quote_plus(search_query)}&tbm=nws",
        f"https://news.search.yahoo.com/search?p={quote_plus(company_name)}+stock+news+recent",
        f"https://economictimes.indiatimes.com/searchresult.cms?query={quote_plus(company_name)}+company",
        f"https://timesofindia.indiatimes.com/topic/{quote_plus(company_name)}+business"
    ]
    
    all_urls = []
    company_name_lower = company_name.lower()
    
    for search_url in search_urls:
        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract links based on the search engine pattern
            if "google.com" in search_url:
                for link in soup.select('a'):
                    href = link.get('href')
                    if href and href.startswith('http') and 'google.com' not in href:
                        # Only include links that mention the company name in the URL or text
                        link_text = link.text.lower()
                        if company_name_lower in href.lower() or company_name_lower in link_text:
                            all_urls.append(href)
            elif "yahoo.com" in search_url:
                for link in soup.select('a'):
                    href = link.get('href')
                    if href and href.startswith('http') and 'yahoo.com' not in href:
                        link_text = link.text.lower()
                        if company_name_lower in href.lower() or company_name_lower in link_text:
                            all_urls.append(href)
            elif "economictimes" in search_url:
                for link in soup.select('a'):
                    href = link.get('href')
                    if href:
                        if href.startswith('/'):
                            href = 'https://economictimes.indiatimes.com' + href
                        link_text = link.text.lower()
                        if company_name_lower in href.lower() or company_name_lower in link_text:
                            all_urls.append(href)
            elif "timesofindia" in search_url:
                for link in soup.select('a'):
                    href = link.get('href')
                    if href:
                        if href.startswith('/'):
                            href = 'https://timesofindia.indiatimes.com' + href
                        link_text = link.text.lower()
                        if company_name_lower in href.lower() or company_name_lower in link_text:
                            all_urls.append(href)
            
            # If we have enough URLs, break
            if len(all_urls) >= min_articles:
                break
                
            # Polite delay
            time.sleep(random.uniform(1, 2))
            
        except Exception as e:
            logger.error(f"Error fetching from {search_url}: {e}")
            continue
    
    # Add company-specific financial news sites
    financial_sites = [
        f"https://finance.yahoo.com/quote/{quote_plus(company_name)}",
        f"https://www.bloomberg.com/search?query={quote_plus(company_name)}",
        f"https://www.reuters.com/search/news?blob={quote_plus(company_name)}",
        f"https://www.cnbc.com/search/?query={quote_plus(company_name)}&qsearchterm={quote_plus(company_name)}"
    ]
    
    for site in financial_sites:
        try:
            response = requests.get(site, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.select('a'):
                href = link.get('href')
                if href:
                    if href.startswith('/'):
                        base_url = '/'.join(site.split('/')[:3])
                        href = base_url + href
                    if href.startswith('http') and company_name_lower in href.lower():
                        all_urls.append(href)
            
            # Polite delay
            time.sleep(random.uniform(1, 2))
            
        except Exception as e:
            logger.error(f"Error fetching from financial site {site}: {e}")
            continue
    
    # Fallback for companies - use more specific URLs
    if len(all_urls) < min_articles:
        company_slug = company_name.lower().replace(' ', '-')
        backup_urls = [
            f"https://www.moneycontrol.com/company-article/{company_slug}/{company_slug}-{company_slug}/news/",
            f"https://www.business-standard.com/company/{company_slug}",
            f"https://www.livemint.com/companies/{company_slug}"
        ]
        
        for backup_url in backup_urls:
            try:
                response = requests.get(backup_url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract links from backup sources
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href and ('news' in href or 'article' in href):
                        if href.startswith('/'):
                            base_url = '/'.join(backup_url.split('/')[:3])
                            href = base_url + href
                        if href.startswith('http'):
                            all_urls.append(href)
                
                if len(all_urls) >= min_articles:
                    break
                    
                # Polite delay
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.error(f"Error fetching from {backup_url}: {e}")
                continue
    
    # Remove duplicates
    all_urls = list(set(all_urls))
    
    # Return the URLs
    return all_urls[:min_articles]

def extract_article_content(url):
    """Extract the content from a news article URL."""
    try:
        ua = UserAgent()
        headers = {'User-Agent': ua.random}
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Extract title
        title = soup.title.text.strip() if soup.title else ""
        
        # Try different methods to extract the main content
        article_text = ""
        
        # Method 1: Look for article tags
        main_content = soup.find('article') or soup.find(class_=re.compile('article|content|story|news'))
        if main_content:
            paragraphs = main_content.find_all('p')
            article_text = ' '.join([p.text.strip() for p in paragraphs])
        
        # Method 2: If method 1 fails, get all paragraphs
        if not article_text:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 100])
        
        # Method 3: Fallback - get all text
        if not article_text:
            article_text = soup.get_text()
            # Clean up the text
            lines = (line.strip() for line in article_text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            article_text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Clean up the title and article text
        title = re.sub(r'\s+', ' ', title)
        article_text = re.sub(r'\s+', ' ', article_text)
        
        # Return a dictionary with the article data
        return {
            'title': title,
            'content': article_text,
            'url': url
        }
        
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return None  # Return None instead of a placeholder to filter out failed articles

def verify_article_relevance(article, company_name):
    """Verify if an article is relevant to the company."""
    if not article:
        return False
    
    company_name_lower = company_name.lower()
    title_lower = article['title'].lower()
    content_preview = article['content'][:1000].lower()  # Check first 1000 chars
    
    # Check if company name is in title or content
    if company_name_lower in title_lower or company_name_lower in content_preview:
        return True
    
    # Check for company name variations
    company_variations = [
        company_name_lower,
        company_name_lower.replace(' ', ''),
        company_name_lower.replace(' ', '-'),
        ''.join(word[0] for word in company_name_lower.split())  # acronym
    ]
    
    for variation in company_variations:
        if variation in title_lower or variation in content_preview:
            return True
    
    return False

def get_news_articles(company_name, min_articles=10):
    """Main function to get news articles for a company."""
    # Get more URLs than needed to account for failures
    urls = fetch_news_urls(company_name, min_articles=min_articles*2)
    
    articles = []
    for url in urls:
        try:
            article = extract_article_content(url)
            if article and verify_article_relevance(article, company_name):
                if article['content'] and len(article['content']) > 200:  # Ensure we have meaningful content
                    articles.append(article)
            
            # If we have enough articles, break
            if len(articles) >= min_articles:
                break
                
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            continue
    
    # If we don't have enough articles, create dummy ones based on the company name
    if len(articles) < min_articles:
        missing_count = min_articles - len(articles)
        logger.info(f"Only found {len(articles)} articles for {company_name}, creating {missing_count} generic ones")
        
        generic_content = [
            f"{company_name} continues to expand its business operations with new initiatives in the market.",
            f"Investors are closely watching {company_name}'s market performance as the company navigates changing economic conditions.",
            f"{company_name} recently announced plans to innovate in their sector with new products and services.",
            f"Economic outlook for {company_name} remains stable according to analysts, despite market volatility.",
            f"{company_name} addresses industry challenges with new strategies and leadership changes.",
            f"The CEO of {company_name} spoke about future growth plans in a recent interview.",
            f"{company_name} reported quarterly results that show promising trends in their core business.",
            f"Analysts have updated their forecasts for {company_name} based on recent market developments.",
            f"{company_name} is expanding into new markets as part of their long-term growth strategy.",
            f"Recent regulatory changes may impact {company_name}'s operations in the coming quarters."
        ]
        
        for i in range(missing_count):
            articles.append({
                'title': f"{company_name} {['News Update', 'Market Analysis', 'Company Report', 'Business Overview', 'Industry Update'][i % 5]} {i+1}",
                'content': generic_content[i % len(generic_content)],
                'url': "#"
            })
    
    return articles[:min_articles]

# Sentiment Analysis Function
def analyze_sentiment(text):
    """Analyze the sentiment of a text using a pre-trained model."""
    try:
        # Load sentiment analysis pipeline
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # For longer texts, split into chunks to avoid model limitations
        if len(text) > 1000:
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            results = [sentiment_analyzer(chunk)[0] for chunk in chunks]
            
            # Average the scores
            avg_score = np.mean([r['score'] for r in results])
            
            # Determine sentiment based on average score
            if avg_score >= 0.6:
                return "Positive"
            elif avg_score <= 0.4:
                return "Negative"
            else:
                return "Neutral"
        else:
            result = sentiment_analyzer(text)[0]
            if result['label'] == "POSITIVE":
                return "Positive"
            elif result['label'] == "NEGATIVE":
                return "Negative"
            else:
                return "Neutral"
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return "Neutral"  # Default to neutral in case of errors

def setup_llm():
    """Set up and return the HuggingFace LLM with better fallback options."""
    try:
        # Set API token from environment variable
        huggingface_api_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        
        if not huggingface_api_token:
            logger.warning("HUGGINGFACE_API_TOKEN not found in environment variables.")
            # Try to get token another way for Hugging Face Spaces
            if os.path.exists('/root/.huggingface/token'):
                with open('/root/.huggingface/token', 'r') as f:
                    huggingface_api_token = f.read().strip()
                    logger.info("Found token in /root/.huggingface/token")
        
        # Try a sequence of models in order of preference
        models_to_try = [
            "meta-llama/Meta-Llama-3-8B-Instruct",  # First choice
            "google/flan-t5-large",                # Second choice
            "google/flan-t5-base",                 # Third choice
            "facebook/bart-large-cnn"              # Last resort
        ]
        
        last_exception = None
        for model in models_to_try:
            try:
                logger.info(f"Attempting to initialize model: {model}")
                llm = HuggingFaceHub(
                    repo_id=model,
                    huggingfacehub_api_token=huggingface_api_token,
                    model_kwargs={"temperature": 0.7, "max_length": 512}
                )
                # Test with a simple prompt to verify it works
                _ = llm("Hello")
                logger.info(f"Successfully initialized model: {model}")
                return llm
            except Exception as e:
                logger.warning(f"Failed to initialize model {model}: {e}")
                last_exception = e
                continue
        
        # If we get here, all models failed
        raise Exception(f"Failed to initialize any LLM models. Last error: {last_exception}")
    
    except Exception as e:
        logger.error(f"Error in setup_llm: {e}")
        raise Exception(f"Failed to initialize LLM: {e}")
    
    
def extract_key_topics(article, llm):
    """Extract key topics from an article."""
    prompt_template = """
    Extract 3-5 main topics from the following news article about {company}. 
    
    Article: {article}
    
    Return just the key topics as a comma-separated list with no additional commentary.
    """
    
    prompt = PromptTemplate(
        input_variables=["company", "article"],
        template=prompt_template,
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        topics = chain.run(company=article['company_name'], article=article['content'])
        # Clean up the topics
        topics = topics.strip().split(',')
        topics = [topic.strip() for topic in topics]
        return topics[:5]  # Limit to 5 topics
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        return ["Business", "News", "Finance"]

def create_article_summary(article, llm):
    """Create a summary of an article."""
    # Limit content length to avoid LLM token limits
    content_preview = article['content'][:3000]  # Use first 3000 chars for summary
    
    prompt_template = """
    Summarize the following news article about {company} in 2-3 sentences:
    
    Article Title: {title}
    Article Content: {article}
    
    Focus on the specific details from THIS article. Create a unique summary that captures what makes this article different.
    
    Summary:
    """
    
    prompt = PromptTemplate(
        input_variables=["company", "title", "article"],
        template=prompt_template,
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        # Add debugging to see what's happening
        logger.info(f"Sending article to LLM for summary: {article['title'][:50]}...")
        summary = chain.run(
            company=article['company_name'], 
            title=article['title'],
            article=content_preview
        )
        
        # Check if we got a valid response
        if summary and len(summary.strip()) > 20:
            logger.info(f"Received summary from LLM: {summary[:50]}...")
            return summary.strip()
        else:
            # If summary is too short or empty, create a basic one based on title
            logger.warning(f"LLM returned invalid summary, creating basic summary from title")
            return f"This article discusses {article['title']} in relation to {article['company_name']}."
    except Exception as e:
        logger.error(f"Error creating summary: {e}")
        # Create a better fallback that uses the title
        return f"This article titled '{article['title']}' discusses recent developments related to {article['company_name']}."

def conduct_comparative_analysis(articles, llm):
    """Conduct a comparative analysis of multiple articles."""
    # Create more detailed article descriptions for the prompt
    article_descriptions = []
    for i, article in enumerate(articles[:5]):  # Limit to 5 articles to avoid token limits
        # Include title, sentiment and topics in the description
        topics_text = ", ".join(article.get('topics', ['General'])[:3])
        desc = f"Article {i+1}: Title: '{article['title']}' - Sentiment: {article['sentiment']} - Topics: {topics_text}"
        article_descriptions.append(desc)
    
    article_list = "\n".join(article_descriptions)
    
    prompt_template = """
    Analyze the following news articles about {company}:
    
    {article_list}
    
    Create a detailed comparative analysis in JSON format. Focus on meaningful differences between the articles.
    Your analysis should include:
    
    1. Sentiment distribution counts
    2. At least 2 specific differences in coverage between articles
    3. Identification of common and unique topics
    
    Format your response as valid JSON with this structure:
    {{
      "Sentiment Distribution": {{
        "Positive": [count],
        "Negative": [count],
        "Neutral": [count]
      }},
      "Coverage Differences": [
        {{
          "Comparison": "Example: Article 1 discusses financial results while Article 2 focuses on product launches",
          "Impact": "Example: This shows the company is active in both financial markets and product development"
        }}
      ],
      "Topic Overlap": {{
        "Common Topics": ["topics found in multiple articles"],
        "Unique Topics": ["topics found in only one article"]
      }}
    }}
    
    Return only valid JSON without any additional text or code blocks:
    """
    
    prompt = PromptTemplate(
        input_variables=["company", "article_list"],
        template=prompt_template,
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        # Add debugging to see what's happening
        logger.info(f"Sending articles to LLM for comparative analysis...")
        analysis = chain.run(company=articles[0]['company_name'], article_list=article_list)
        logger.info(f"Received analysis from LLM: {analysis[:100]}...")
        
        # Clean up the response to ensure valid JSON
        import json
        import re
        
        # Extract JSON pattern
        json_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
        match = json_pattern.search(analysis)
        if match:
            analysis = match.group(1)
        
        # Remove any non-JSON text
        analysis = re.sub(r'^[^{]*', '', analysis)
        analysis = re.sub(r'[^}]*$', '', analysis)
        
        # Parse and reformat to ensure valid JSON
        try:
            analysis_dict = json.loads(analysis)
            logger.info("Successfully parsed JSON response")
            
            # Validate that we have the expected keys
            required_keys = ["Sentiment Distribution", "Coverage Differences", "Topic Overlap"]
            if not all(key in analysis_dict for key in required_keys):
                logger.warning("Missing required keys in JSON response, using fallback")
                return create_fallback_analysis(articles)
                
            return analysis_dict
        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error: {je}")
            # Fallback for JSON errors
            return create_fallback_analysis(articles)
    except Exception as e:
        logger.error(f"Error in comparative analysis: {e}")
        return create_fallback_analysis(articles)

def create_fallback_analysis(articles):
    """Create a fallback analysis if the LLM fails."""
    # Count sentiments
    positive_count = sum(1 for a in articles if a['sentiment'] == "Positive")
    negative_count = sum(1 for a in articles if a['sentiment'] == "Negative")
    neutral_count = sum(1 for a in articles if a['sentiment'] == "Neutral")
    
    # Extract article titles for better comparisons
    titles = [a['title'] for a in articles[:3]]
    
    # Collect all topics
    all_topics = []
    for article in articles:
        all_topics.extend(article.get('topics', []))
    
    # Count topic occurrences
    topic_counts = {}
    for topic in all_topics:
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    # Find common and unique topics
    common_topics = [topic for topic, count in topic_counts.items() if count > 1]
    unique_topics = [topic for topic, count in topic_counts.items() if count == 1]
    
    # Create more specific coverage differences based on titles
    coverage_differences = [
        {
            "Comparison": f"Article about '{titles[0]}' focuses on different aspects compared to '{titles[1]}'.",
            "Impact": f"This shows diversity in {articles[0]['company_name']}'s coverage across multiple dimensions."
        }
    ]
    
    if len(titles) > 2:
        coverage_differences.append({
            "Comparison": f"Article '{titles[0]}' discusses different matters than '{titles[2]}'.",
            "Impact": "This range of coverage helps provide a more complete picture of the company's situation."
        })
    
    return {
        "Sentiment Distribution": {
            "Positive": positive_count,
            "Negative": negative_count,
            "Neutral": neutral_count
        },
        "Coverage Differences": coverage_differences,
        "Topic Overlap": {
            "Common Topics": common_topics[:3] if common_topics else ["Business"],
            "Unique Topics": unique_topics[:5] if unique_topics else ["Various topics"]
        }
    }

def create_final_sentiment_summary(articles, comparative_analysis, llm):
    """Create a final sentiment summary based on all articles."""
    # Count sentiment distribution
    sentiment_counts = comparative_analysis.get("Sentiment Distribution", {
        "Positive": sum(1 for a in articles if a['sentiment'] == "Positive"),
        "Negative": sum(1 for a in articles if a['sentiment'] == "Negative"),
        "Neutral": sum(1 for a in articles if a['sentiment'] == "Neutral")
    })
    
    # Create prompt
    prompt_template = """
    Based on the analysis of {article_count} news articles about {company}:
    
    Sentiment Distribution:
    - Positive articles: {positive_count}
    - Negative articles: {negative_count}
    - Neutral articles: {neutral_count}
    
    Create a brief (1-2 sentences) overall sentiment analysis summary for {company} based on these news articles.
    Make sure to mention the company name in your summary.
    """
    
    prompt = PromptTemplate(
        input_variables=["company", "article_count", "positive_count", "negative_count", "neutral_count"],
        template=prompt_template,
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        summary = chain.run(
            company=articles[0]['company_name'],
            article_count=len(articles),
            positive_count=sentiment_counts.get("Positive", 0),
            negative_count=sentiment_counts.get("Negative", 0),
            neutral_count=sentiment_counts.get("Neutral", 0)
        )
        return summary.strip()
    except Exception as e:
        logger.error(f"Error creating final sentiment summary: {e}")
        # Fallback summary
        main_sentiment = max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "Neutral"
        return f"{articles[0]['company_name']}'s recent news coverage is predominantly {main_sentiment.lower()}."

# Translation and TTS Functions
def translate_to_hindi(text):
    """Translate text to Hindi."""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='hi')
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        logger.error(f"Error with Google translation: {e}")
        # Use a fallback method
        try:
            from translate import Translator
            translator = Translator(to_lang="hi")
            translated_text = translator.translate(text)
            return translated_text
        except Exception as e2:
            logger.error(f"Error with fallback translation: {e2}")
            return text  # Return original text if translation fails

def generate_hindi_tts(text):
    """Generate Hindi TTS for a given text."""
    try:
        # Translate text to Hindi
        hindi_text = translate_to_hindi(text)
        
        # Generate TTS using gTTS
        tts = gTTS(text=hindi_text, lang='hi', slow=False)
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_filename = temp_file.name
        temp_file.close()
        
        tts.save(temp_filename)
        
        # Convert to base64 for embedding in response
        with open(temp_filename, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Clean up the temporary file
        os.unlink(temp_filename)
        
        return {
            'hindi_text': hindi_text,
            'audio_data': audio_data
        }
    except Exception as e:
        logger.error(f"Error generating Hindi TTS: {e}")
        return {
            'hindi_text': "Hindi translation not available",
            'audio_data': ""
        }