import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyzes sentiment in social media comments."""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment scores to the dataframe."""
        logger.info("Starting sentiment analysis...")
        
        df = df.copy()
        
        # Calculate sentiment scores
        sentiment_scores = df['comment_text'].apply(self._get_sentiment_scores)
        
        # Extract individual components
        df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
        df['sentiment_positive'] = sentiment_scores.apply(lambda x: x['pos'])
        df['sentiment_negative'] = sentiment_scores.apply(lambda x: x['neg'])
        df['sentiment_neutral'] = sentiment_scores.apply(lambda x: x['neu'])
        
        # Classify sentiment
        df['sentiment_label'] = df['sentiment_compound'].apply(self._classify_sentiment)
        
        logger.info("Sentiment analysis complete")
        return df
    
    def _get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """Get VADER sentiment scores for a text."""
        if pd.isna(text):
            return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
            
        return self.analyzer.polarity_scores(text)
    
    def _classify_sentiment(self, compound_score: float) -> str:
        """Classify sentiment based on compound score."""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_sentiment_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze sentiment trends over time and across different dimensions."""
        
        # Overall sentiment distribution
        sentiment_dist = df['sentiment_label'].value_counts(normalize=True)
        
        # Daily sentiment trends
        daily_sentiment = df.groupby('date').agg({
            'sentiment_compound': ['mean', 'std', 'count'],
            'sentiment_label': lambda x: x.value_counts(normalize=True).to_dict()
        })
        
        # Sentiment by day of week
        dow_sentiment = df.groupby('day_of_week')['sentiment_compound'].mean().sort_values(ascending=False)
        
        # Sentiment by hour
        hourly_sentiment = df.groupby('hour')['sentiment_compound'].mean()
        
        # Most positive and negative comments
        most_positive = df.nlargest(10, 'sentiment_compound')[['comment_text', 'sentiment_compound', 'timestamp']]
        most_negative = df.nsmallest(10, 'sentiment_compound')[['comment_text', 'sentiment_compound', 'timestamp']]
        
        return {
            'overall_distribution': sentiment_dist.to_dict(),
            'daily_trends': daily_sentiment,
            'day_of_week_sentiment': dow_sentiment.to_dict(),
            'hourly_sentiment': hourly_sentiment.to_dict(),
            'most_positive_comments': most_positive.to_dict('records'),
            'most_negative_comments': most_negative.to_dict('records'),
            'avg_sentiment_score': df['sentiment_compound'].mean()
        }
    
    def analyze_product_sentiment(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze sentiment for specific product mentions."""
        product_keywords = [
            'scrub', 'lotion', 'serum', 'oil', 'wash', 'polish',
            'vanilla', 'coconut', 'shea', 'moroccan rose', 'tropical mango',
            'sugar scrub', 'body butter', 'tangerine', 'grapefruit'
        ]
        
        product_sentiment = {}
        
        for product in product_keywords:
            # Find comments mentioning this product
            mask = df['comment_text'].str.contains(rf'\b{re.escape(product)}\b', case=False, na=False)
            product_comments = df[mask]
            
            if len(product_comments) > 0:
                product_sentiment[product] = {
                    'comment_count': len(product_comments),
                    'avg_sentiment': product_comments['sentiment_compound'].mean(),
                    'sentiment_distribution': product_comments['sentiment_label'].value_counts(normalize=True).to_dict(),
                    'top_positive_comment': product_comments.loc[product_comments['sentiment_compound'].idxmax(), 'comment_text'] if len(product_comments) > 0 else None,
                    'top_negative_comment': product_comments.loc[product_comments['sentiment_compound'].idxmin(), 'comment_text'] if len(product_comments) > 0 else None
                }
        
        # Sort by comment count
        return dict(sorted(product_sentiment.items(), key=lambda x: x[1]['comment_count'], reverse=True))
    
    def identify_sentiment_drivers(self, df: pd.DataFrame) -> Dict:
        """Identify what drives positive vs negative sentiment."""
        
        positive_comments = df[df['sentiment_label'] == 'positive']
        negative_comments = df[df['sentiment_label'] == 'negative']
        
        # Common words in positive vs negative comments
        from collections import Counter
        
        def get_common_words(comments_series, top_n=20):
            all_text = ' '.join(comments_series.dropna().str.lower())
            # Simple word extraction (could be improved with proper NLP)
            words = re.findall(r'\b[a-z]{3,}\b', all_text)
            return Counter(words).most_common(top_n)
        
        positive_words = get_common_words(positive_comments['comment_text'])
        negative_words = get_common_words(negative_comments['comment_text'])
        
        return {
            'positive_comment_count': len(positive_comments),
            'negative_comment_count': len(negative_comments),
            'positive_avg_length': positive_comments['comment_length'].mean(),
            'negative_avg_length': negative_comments['comment_length'].mean(),
            'common_positive_words': positive_words,
            'common_negative_words': negative_words
        }