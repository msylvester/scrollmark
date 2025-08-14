import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data loading, cleaning, and preprocessing for social media comments."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate the CSV data."""
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Loaded {len(self.df)} rows of data")
            
            required_columns = ['timestamp', 'media_id', 'media_caption', 'comment_text']
            missing_cols = [col for col in required_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess the data."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        logger.info("Starting data cleaning...")
        
        # Convert timestamp to datetime - handle mixed formats
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='mixed')
        
        # Remove null comments
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['comment_text'])
        logger.info(f"Removed {initial_count - len(self.df)} null comments")
        
        # Remove empty comments
        self.df = self.df[self.df['comment_text'].str.strip() != '']
        
        # Add derived features
        self.df['date'] = self.df['timestamp'].dt.date
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.day_name()
        self.df['comment_length'] = self.df['comment_text'].str.len()
        self.df['word_count'] = self.df['comment_text'].str.split().str.len()
        
        logger.info(f"Data cleaning complete. Final dataset: {len(self.df)} rows")
        return self.df
    
    def extract_emojis(self, text: str) -> List[str]:
        """Extract emojis from text."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.findall(text)
    
    def clean_text_for_analysis(self, text: str) -> str:
        """Clean text for NLP analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags for pure text analysis
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_data_summary(self) -> Dict:
        """Generate summary statistics of the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded.")
            
        summary = {
            'total_comments': len(self.df),
            'unique_media_posts': self.df['media_id'].nunique(),
            'date_range': {
                'start': self.df['timestamp'].min(),
                'end': self.df['timestamp'].max()
            },
            'avg_comment_length': self.df['comment_length'].mean(),
            'avg_words_per_comment': self.df['word_count'].mean(),
            'most_active_day': self.df['day_of_week'].mode().iloc[0],
            'most_active_hour': self.df['hour'].mode().iloc[0]
        }
        
        return summary