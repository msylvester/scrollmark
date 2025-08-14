# @treehut Social Media Trend Analysis

A comprehensive data analytics pipeline for identifying trends, sentiment patterns, and actionable insights from social media comments.

## 📊 Project Overview

This project analyzes 17,812 Instagram comments from @treehut's March 2025 posts to extract actionable insights for social media strategy. The analysis combines sentiment analysis, trend identification, and temporal pattern recognition to provide data-driven recommendations.

## 🎯 Key Features

- **Sentiment Analysis**: VADER-based sentiment scoring with positive/negative/neutral classification
- **Trend Identification**: TF-IDF keyword extraction and temporal trend analysis
- **Product Sentiment Tracking**: Product-specific sentiment monitoring
- **Engagement Pattern Analysis**: Time-based engagement optimization insights
- **Interactive Visualizations**: Plotly-powered charts and dashboards
- **Automated Reporting**: Comprehensive markdown reports with executive summaries

## 🏗️ Architecture

```
src/
├── data_processor.py     # Data loading, cleaning, and preprocessing
├── sentiment_analyzer.py # VADER sentiment analysis pipeline
├── trend_analyzer.py     # TF-IDF trend extraction and analysis
└── visualizer.py         # Interactive chart generation

main.py                   # Main analysis pipeline
requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone/Extract the repository**
   ```bash
   cd scrollmark_two
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**
   ```bash
   python main.py
   ```

## 🔬 Extension Proposal

### Priority 1: Asynchronous Real-Time Data Processing Pipeline
**Business Impact: Critical | Development Effort: 4-6 weeks**

Implement async processing architecture using Python's `asyncio` and `aiohttp` for concurrent Instagram API data ingestion, sentiment analysis, and trend detection. This would enable:

- **Concurrent API calls**: Process multiple posts/comments simultaneously instead of sequentially
- **Background sentiment scoring**: Queue-based VADER analysis using `asyncio.gather()` for parallel processing
- **Live dashboard updates**: WebSocket connections for real-time visualization refresh
- **Scalable data ingestion**: Handle 10x larger datasets (100K+ comments) without blocking

**Technical Implementation**: Replace current synchronous pipeline with async/await patterns, implement Redis queue for comment processing, and use asyncpg for database operations. This architecture would reduce processing time from minutes to seconds and enable real-time monitoring capabilities.


### Priority 2: AI Agent Comment Relevance Filtering
**Business Impact: High | Development Effort: 3-4 weeks**

Deploy an LLM-based agent to pre-filter comments for relevance before sentiment analysis, eliminating noise from spam, off-topic discussions, and bot comments. This would include:

- **Relevance scoring**: Agent evaluates each comment's relationship to brand/products (0-1 relevance score)
- **Context understanding**: Natural language processing to identify genuine customer feedback vs. irrelevant chatter
- **Quality gating**: Only comments scoring >0.7 relevance proceed to sentiment analysis pipeline
- **Spam detection**: Automated filtering of promotional, bot, or duplicate content

**Technical Implementation**: Integrate OpenAI/Claude API with custom prompt engineering for comment relevance assessment. Implement async agent calls within the processing pipeline, with fallback rules for API failures. Cache agent decisions to reduce API costs for similar comment patterns.


## 🔍 AI & Tool Usage Disclosure

### Primary Tools Used
- **Claude Code**: Code generation, analysis design, and optimization
- **Python Libraries**: pandas, scikit-learn, VADER, Plotly, matplotlib
- **Data Science Stack**: Standard ML/analytics libraries for robust analysis

### AI Assistance
- **Code Architecture**: AI-assisted modular design and best practices
- **Analysis Methods**: AI-recommended sentiment analysis and trend detection approaches
- **Visualization Design**: AI-guided chart selection for optimal insight presentation
- **Report Generation**: AI-assisted executive summary and recommendation synthesis


