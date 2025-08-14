import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Creates visualizations for social media trend analysis."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        plt.style.use('seaborn-v0_8')
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
    def create_sentiment_timeline(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """Create an interactive timeline of sentiment trends."""
        
        # Daily sentiment aggregation
        daily_sentiment = df.groupby('date').agg({
            'sentiment_compound': 'mean',
            'comment_text': 'count'
        }).reset_index()
        daily_sentiment.columns = ['date', 'avg_sentiment', 'comment_count']
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['avg_sentiment'],
                mode='lines+markers',
                name='Average Sentiment',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=8)
            ),
            secondary_y=False,
        )
        
        # Add comment count bars
        fig.add_trace(
            go.Bar(
                x=daily_sentiment['date'],
                y=daily_sentiment['comment_count'],
                name='Comment Count',
                opacity=0.3,
                marker_color='#4ECDC4'
            ),
            secondary_y=True,
        )
        
        # Update layout
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Average Sentiment Score", secondary_y=False)
        fig.update_yaxes(title_text="Number of Comments", secondary_y=True)
        
        fig.update_layout(
            title="Daily Sentiment Trends and Comment Volume",
            height=500,
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(f"{self.output_dir}/{save_path}")
            
        return fig
    
    def create_keyword_trends(self, temporal_trends: Dict, save_path: str = None) -> go.Figure:
        """Create visualization of keyword trends over time."""
        
        # Prepare data for visualization
        dates = []
        keywords = []
        scores = []
        
        for date_str, data in temporal_trends.items():
            date = pd.to_datetime(date_str)
            for keyword, score in data['keywords'][:5]:  # Top 5 keywords per day
                dates.append(date)
                keywords.append(keyword)
                scores.append(score)
        
        trend_df = pd.DataFrame({
            'date': dates,
            'keyword': keywords,
            'score': scores
        })
        
        # Get top overall keywords
        top_keywords = trend_df.groupby('keyword')['score'].mean().nlargest(10).index
        filtered_df = trend_df[trend_df['keyword'].isin(top_keywords)]
        
        # Create line plot
        fig = px.line(
            filtered_df,
            x='date',
            y='score',
            color='keyword',
            title='Top Keyword Trends Over Time',
            labels={'score': 'TF-IDF Score', 'date': 'Date'}
        )
        
        fig.update_layout(height=500)
        
        if save_path:
            fig.write_html(f"{self.output_dir}/{save_path}")
            
        return fig
    
    def create_product_sentiment_chart(self, product_sentiment: Dict, save_path: str = None) -> go.Figure:
        """Create visualization of sentiment by product."""
        
        products = []
        sentiment_scores = []
        comment_counts = []
        
        for product, data in product_sentiment.items():
            if data['comment_count'] >= 5:  # Only products with sufficient mentions
                products.append(product.title())
                sentiment_scores.append(data['avg_sentiment'])
                comment_counts.append(data['comment_count'])
        
        # Create bubble chart
        fig = go.Figure(data=go.Scatter(
            x=sentiment_scores,
            y=products,
            mode='markers',
            marker=dict(
                size=comment_counts,
                sizemode='diameter',
                sizeref=2.*max(comment_counts)/(40.**2),
                sizemin=4,
                color=sentiment_scores,
                colorscale='RdYlGn',
                colorbar=dict(title="Sentiment Score"),
                line=dict(width=1, color='DarkSlateGray')
            ),
            text=[f"Comments: {count}" for count in comment_counts],
            hovertemplate='%{y}<br>Sentiment: %{x:.3f}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Product Sentiment Analysis<br><sub>Bubble size = number of mentions</sub>',
            xaxis_title='Average Sentiment Score',
            yaxis_title='Product',
            height=600,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(f"{self.output_dir}/{save_path}")
            
        return fig
    
    def create_engagement_heatmap(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """Create heatmap of engagement patterns by day and hour."""
        
        # Create engagement matrix
        engagement_matrix = df.groupby(['day_of_week', 'hour']).size().reset_index(name='comments')
        
        # Pivot for heatmap
        heatmap_data = engagement_matrix.pivot(index='day_of_week', columns='hour', values='comments')
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Blues',
            colorbar=dict(title="Number of Comments")
        ))
        
        fig.update_layout(
            title='Comment Activity Heatmap',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            height=400
        )
        
        if save_path:
            fig.write_html(f"{self.output_dir}/{save_path}")
            
        return fig
    
    def create_wordcloud(self, text_series: pd.Series, title: str = "Word Cloud", save_path: str = None) -> None:
        """Create and save a word cloud visualization."""
        
        # Combine all text
        text = ' '.join(text_series.dropna().astype(str))
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(f"{self.output_dir}/{save_path}", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_summary_dashboard(self, summary_stats: Dict, save_path: str = None) -> go.Figure:
        """Create a summary dashboard with key metrics."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Comment Distribution', 'Sentiment Overview', 'Daily Activity', 'Top Insights'),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "table"}]]
        )
        
        # This would be populated with actual summary statistics
        # For now, creating placeholder structure
        
        fig.update_layout(
            height=800,
            title_text="@treehut Social Media Analytics Dashboard",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(f"{self.output_dir}/{save_path}")
            
        return fig