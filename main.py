#!/usr/bin/env python3
"""
Main analysis script for @treehut social media trend identification.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_processor import DataProcessor
from trend_analyzer import TrendAnalyzer
from sentiment_analyzer import SentimentAnalyzer
from visualizer import Visualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main analysis pipeline."""
    
    logger.info("Starting @treehut social media trend analysis...")
    
    # Initialize components
    data_processor = DataProcessor('engagements.csv')
    visualizer = Visualizer('reports')
    
    # Create output directories
    os.makedirs('reports', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        # Step 1: Load and clean data
        logger.info("Step 1: Loading and cleaning data...")
        df = data_processor.load_data()
        df = data_processor.clean_data()
        
        # Get data summary
        summary = data_processor.get_data_summary()
        logger.info(f"Data summary: {summary['total_comments']} comments across {summary['unique_media_posts']} posts")
        
        # Step 2: Sentiment Analysis
        logger.info("Step 2: Performing sentiment analysis...")
        sentiment_analyzer = SentimentAnalyzer()
        df = sentiment_analyzer.analyze_sentiment(df)
        
        sentiment_trends = sentiment_analyzer.analyze_sentiment_trends(df)
        product_sentiment = sentiment_analyzer.analyze_product_sentiment(df)
        sentiment_drivers = sentiment_analyzer.identify_sentiment_drivers(df)
        
        # Step 3: Trend Analysis
        logger.info("Step 3: Analyzing trends and keywords...")
        trend_analyzer = TrendAnalyzer(df)
        
        # Download NLTK data if needed
        try:
            import nltk
            nltk.download('stopwords', quiet=True)
        except:
            logger.warning("Could not download NLTK stopwords")
        
        top_keywords = trend_analyzer.extract_keywords(df['comment_text'], top_n=30)
        product_mentions = trend_analyzer.analyze_product_mentions()
        temporal_trends = trend_analyzer.analyze_temporal_trends()
        emerging_trends = trend_analyzer.identify_emerging_trends()
        engagement_patterns = trend_analyzer.analyze_engagement_patterns()
        
        # Step 4: Generate Visualizations
        logger.info("Step 4: Creating visualizations...")
        
        # Sentiment timeline
        sentiment_fig = visualizer.create_sentiment_timeline(df, 'sentiment_timeline.html')
        
        # Product sentiment chart
        product_fig = visualizer.create_product_sentiment_chart(product_sentiment, 'product_sentiment.html')
        
        # Engagement heatmap
        engagement_fig = visualizer.create_engagement_heatmap(df, 'engagement_heatmap.html')
        
        # Keyword trends (if we have temporal data)
        if temporal_trends:
            keyword_fig = visualizer.create_keyword_trends(temporal_trends, 'keyword_trends.html')
        
        # Create word clouds
        positive_comments = df[df['sentiment_label'] == 'positive']['comment_text']
        visualizer.create_wordcloud(positive_comments, 'Positive Comments Word Cloud', 'positive_wordcloud.png')
        
        # Step 5: Generate comprehensive report
        logger.info("Step 5: Generating analysis report...")
        
        report_data = {
            'summary': summary,
            'sentiment_trends': sentiment_trends,
            'product_sentiment': product_sentiment,
            'sentiment_drivers': sentiment_drivers,
            'top_keywords': top_keywords,
            'product_mentions': product_mentions,
            'temporal_trends': temporal_trends,
            'emerging_trends': emerging_trends,
            'engagement_patterns': engagement_patterns,
            'total_comments': len(df),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save processed data
        df.to_csv('data/processed/comments_with_analysis.csv', index=False)
        
        # Generate markdown report
        generate_markdown_report(report_data)
        
        logger.info("Analysis complete! Check the reports/ directory for outputs.")
        
        # Print key insights to console
        print_key_insights(report_data)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

def generate_markdown_report(data: dict):
    """Generate a comprehensive markdown report."""
    
    report_content = f"""# @treehut Social Media Trend Analysis Report

*Generated on {data['analysis_date']}*

## Executive Summary

This analysis examines {data['total_comments']:,} comments from @treehut's Instagram posts in March 2025 to identify trends, sentiment patterns, and actionable insights for social media strategy.

**Key Findings:**
- **Overall Sentiment:** {data['sentiment_trends']['avg_sentiment_score']:.3f} (on scale -1 to +1)
- **Sentiment Distribution:** {data['sentiment_trends']['overall_distribution']['positive']:.1%} positive, {data['sentiment_trends']['overall_distribution']['negative']:.1%} negative, {data['sentiment_trends']['overall_distribution']['neutral']:.1%} neutral
- **Most Active Day:** {data['summary']['most_active_day']}
- **Peak Engagement Hour:** {data['summary']['most_active_hour']}:00

## Top Trending Keywords

"""
    
    # Add top keywords
    for i, (keyword, score) in enumerate(data['top_keywords'][:10]):
        report_content += f"{i+1}. **{keyword}** (relevance: {score:.3f})\n"
    
    report_content += f"""

## Product Sentiment Analysis

"""
    
    # Add product sentiment
    for product, sentiment_data in list(data['product_sentiment'].items())[:8]:
        report_content += f"- **{product.title()}:** {sentiment_data['avg_sentiment']:.3f} sentiment ({sentiment_data['comment_count']} mentions)\n"
    
    if data['emerging_trends']:
        report_content += f"""

## Emerging Trends

"""
        for trend in data['emerging_trends'][:5]:
            growth = trend['growth_rate']
            if growth == float('inf'):
                report_content += f"- **{trend['keyword']}:** New trending topic (appeared in second half of March)\n"
            else:
                report_content += f"- **{trend['keyword']}:** {growth:.1%} growth rate\n"
    
    report_content += f"""

## Engagement Insights

- **Total Posts Analyzed:** {data['engagement_patterns']['total_posts']}
- **Average Comments per Post:** {data['engagement_patterns']['avg_comments_per_post']:.1f}
- **High-Engagement Threshold:** {data['engagement_patterns']['high_engagement_threshold']:.0f} comments

### Top-Performing Posts
"""
    
    for i, post in enumerate(data['engagement_patterns']['top_posts'][:3]):
        report_content += f"{i+1}. **{post['comment_count']} comments** - Media ID: {post['media_id']}\n"
        report_content += f"   Caption: {post['media_caption']}\n"
    
    report_content += f"""

## Actionable Recommendations

### Content Strategy
1. **Leverage Positive Sentiment:** With {data['sentiment_trends']['overall_distribution']['positive']:.1%} positive sentiment, continue current content approach
2. **Optimize Timing:** Post during peak engagement hours around {data['summary']['most_active_hour']}:00
3. **Focus on High-Performing Products:** Prioritize content around products with highest positive sentiment

### Community Engagement
1. **Monitor Emerging Trends:** Track keywords showing growth for early trend adoption
2. **Address Concerns:** Review negative sentiment comments for product improvement opportunities
3. **Encourage User-Generated Content:** Leverage positive community sentiment

### Product Development
1. **Product Popularity:** Focus marketing on most-mentioned products
2. **Sentiment Monitoring:** Track product-specific sentiment for quality insights

## Methodology

- **Data Source:** {data['total_comments']:,} Instagram comments from March 2025
- **Sentiment Analysis:** VADER sentiment analysis tool
- **Trend Identification:** TF-IDF keyword extraction and temporal analysis
- **Visualization:** Interactive charts and statistical analysis

---

*This analysis was generated using automated tools and should be validated with domain expertise.*
"""
    
    # Save the report
    with open('reports/analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)

def print_key_insights(data: dict):
    """Print key insights to console."""
    
    print("\n" + "="*60)
    print("KEY INSIGHTS FOR @TREEHUT SOCIAL MEDIA STRATEGY")
    print("="*60)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • {data['total_comments']:,} total comments analyzed")
    print(f"   • {data['summary']['unique_media_posts']} unique posts")
    print(f"   • {data['summary']['avg_comment_length']:.1f} average comment length")
    
    print(f"\n😊 SENTIMENT ANALYSIS:")
    print(f"   • Overall sentiment: {data['sentiment_trends']['avg_sentiment_score']:.3f}/1.0")
    print(f"   • {data['sentiment_trends']['overall_distribution']['positive']:.1%} positive comments")
    print(f"   • {data['sentiment_trends']['overall_distribution']['negative']:.1%} negative comments")
    
    print(f"\n🔥 TOP TRENDING KEYWORDS:")
    for i, (keyword, score) in enumerate(data['top_keywords'][:5]):
        print(f"   {i+1}. {keyword} (relevance: {score:.3f})")
    
    print(f"\n📈 ENGAGEMENT PATTERNS:")
    print(f"   • Peak day: {data['summary']['most_active_day']}")
    print(f"   • Peak hour: {data['summary']['most_active_hour']}:00")
    print(f"   • Avg comments per post: {data['engagement_patterns']['avg_comments_per_post']:.1f}")
    
    if data['emerging_trends']:
        print(f"\n🚀 EMERGING TRENDS:")
        for trend in data['emerging_trends'][:3]:
            if trend['growth_rate'] == float('inf'):
                print(f"   • {trend['keyword']}: NEW trending topic")
            else:
                print(f"   • {trend['keyword']}: {trend['growth_rate']:.1%} growth")
    
    print(f"\n💡 ACTIONABLE RECOMMENDATIONS:")
    print(f"   • Schedule posts around {data['summary']['most_active_hour']}:00 on {data['summary']['most_active_day']}s")
    print(f"   • Leverage {data['sentiment_trends']['overall_distribution']['positive']:.1%} positive sentiment in content")
    print(f"   • Monitor emerging trends for early adoption opportunities")
    
    print("\n" + "="*60)
    print("📁 Full reports saved in: reports/")
    print("📊 Interactive visualizations: reports/*.html")
    print("📝 Detailed analysis: reports/analysis_report.md")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()