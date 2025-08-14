import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import nltk
import logging

logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """Analyzes trends in social media comments."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # Try to get NLTK stopwords, fallback to manual list if unavailable
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback to manual stopword list
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
                'just', 'don', 'should', 'now'
            }
        
        # Add common social media stop words
        self.stop_words.update(['treehut', 'tree', 'hut', 'love', 'like', 'good', 'great', 'nice', 'amazing', 'beautiful'])
        
    def extract_keywords(self, text_series: pd.Series, top_n: int = 50) -> List[Tuple[str, int]]:
        """Extract top keywords from text using TF-IDF."""
        try:
            # Clean and prepare text
            cleaned_text = text_series.apply(self._clean_text).dropna()
            
            if len(cleaned_text) == 0:
                return []
            
            # Use TF-IDF to find important terms
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            tfidf_matrix = vectorizer.fit_transform(cleaned_text)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Create keyword-score pairs and sort
            keywords = list(zip(feature_names, mean_scores))
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            return keywords[:top_n]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean text for keyword extraction."""
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, hashtags (keep the text after #)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content
        
        # Remove emojis and special characters, keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_product_mentions(self) -> Dict[str, int]:
        """Identify and count product mentions."""
        product_keywords = [
            'scrub', 'lotion', 'serum', 'oil', 'wash', 'polish', 'moisturizer',
            'vanilla', 'coconut', 'shea', 'moroccan rose', 'tropical mango',
            'sugar', 'salt', 'exfoliate', 'hydrate', 'moisturize',
            'tangerine', 'grapefruit', 'lime', 'orange', 'citrus',
            'body butter', 'hand cream', 'foot cream'
        ]
        
        product_counts = {}
        
        for product in product_keywords:
            # Case insensitive search
            pattern = rf'\b{re.escape(product)}\b'
            count = self.df['comment_text'].str.contains(pattern, case=False, na=False).sum()
            if count > 0:
                product_counts[product] = count
                
        return dict(sorted(product_counts.items(), key=lambda x: x[1], reverse=True))
    
    def analyze_temporal_trends(self) -> Dict[str, pd.DataFrame]:
        """Analyze how trends change over time."""
        # Group by date and analyze top keywords per day
        daily_trends = {}
        
        for date in self.df['date'].unique():
            day_df = self.df[self.df['date'] == date]
            keywords = self.extract_keywords(day_df['comment_text'], top_n=10)
            
            daily_trends[str(date)] = {
                'keywords': keywords,
                'comment_count': len(day_df),
                'avg_sentiment': day_df.get('sentiment_score', pd.Series()).mean() if 'sentiment_score' in day_df.columns else None
            }
        
        return daily_trends
    
    def identify_emerging_trends(self, window_days: int = 7) -> List[Dict]:
        """Identify keywords that are becoming more popular over time."""
        # Sort by timestamp
        df_sorted = self.df.sort_values('timestamp')
        
        # Split into time windows
        date_range = df_sorted['date'].max() - df_sorted['date'].min()
        total_days = date_range.days
        
        if total_days < window_days * 2:
            return []
        
        # Compare first half vs second half of March
        mid_point = df_sorted['date'].min() + pd.Timedelta(days=total_days//2)
        
        first_half = df_sorted[df_sorted['date'] <= mid_point]
        second_half = df_sorted[df_sorted['date'] > mid_point]
        
        keywords_first = dict(self.extract_keywords(first_half['comment_text'], top_n=30))
        keywords_second = dict(self.extract_keywords(second_half['comment_text'], top_n=30))
        
        emerging_trends = []
        
        for keyword in keywords_second:
            first_score = keywords_first.get(keyword, 0)
            second_score = keywords_second[keyword]
            
            # Calculate growth rate
            if first_score > 0:
                growth_rate = (second_score - first_score) / first_score
            else:
                growth_rate = float('inf') if second_score > 0 else 0
            
            if growth_rate > 0.5 and second_score > 0.01:  # At least 50% growth and meaningful presence
                emerging_trends.append({
                    'keyword': keyword,
                    'growth_rate': growth_rate,
                    'first_half_score': first_score,
                    'second_half_score': second_score
                })
        
        return sorted(emerging_trends, key=lambda x: x['growth_rate'], reverse=True)
    
    def analyze_engagement_patterns(self) -> Dict:
        """Analyze what drives higher engagement."""
        # Group by media post and calculate engagement metrics
        post_metrics = self.df.groupby('media_id').agg({
            'comment_text': 'count',
            'comment_length': 'mean',
            'word_count': 'mean'
        }).rename(columns={'comment_text': 'comment_count'})
        
        # Merge with captions
        post_captions = self.df.groupby('media_id')['media_caption'].first()
        post_metrics = post_metrics.merge(post_captions, left_index=True, right_index=True)
        
        # Identify high-engagement posts (top 20%)
        engagement_threshold = post_metrics['comment_count'].quantile(0.8)
        high_engagement_posts = post_metrics[post_metrics['comment_count'] >= engagement_threshold]
        
        # Extract keywords from high-engagement captions
        high_engagement_keywords = self.extract_keywords(high_engagement_posts['media_caption'], top_n=20)
        
        # Get top posts as records with proper index handling
        top_posts_df = post_metrics.nlargest(5, 'comment_count').reset_index()
        top_posts = []
        for _, row in top_posts_df.iterrows():
            top_posts.append({
                'media_id': row['media_id'],
                'comment_count': row['comment_count'],
                'avg_comment_length': row['comment_length'],
                'avg_word_count': row['word_count'],
                'media_caption': row['media_caption'][:100] + '...' if len(row['media_caption']) > 100 else row['media_caption']
            })
        
        return {
            'total_posts': len(post_metrics),
            'high_engagement_threshold': engagement_threshold,
            'high_engagement_posts_count': len(high_engagement_posts),
            'avg_comments_per_post': post_metrics['comment_count'].mean(),
            'high_engagement_keywords': high_engagement_keywords,
            'top_posts': top_posts
        }