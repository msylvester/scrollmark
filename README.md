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

### Expected Output

The analysis generates:
- Interactive HTML visualizations in `reports/`
- Comprehensive markdown report at `reports/analysis_report.md`
- Processed data with sentiment scores at `data/processed/comments_with_analysis.csv`
- Console summary with key insights

## 📈 Key Insights Generated

Based on the March 2025 analysis:

### Sentiment Analysis
- **Overall Sentiment**: 0.145/1.0 (mildly positive)
- **Distribution**: 26.9% positive, 4.2% negative, 68.8% neutral
- Strong positive community sentiment indicates healthy brand perception

### Engagement Optimization
- **Peak Day**: Friday (highest comment volume)
- **Peak Hour**: 19:00 (optimal posting time)
- **Average Engagement**: 50.5 comments per post

### Trending Keywords
1. **PR** (0.285 relevance) - Indicates strong PR campaign engagement
2. **Need** (0.013 relevance) - Shows product demand signals
3. **Scent** (0.005 relevance) - Key product attribute focus

## 🔧 Technical Implementation

### Data Pipeline
1. **Data Loading**: Robust CSV parsing with error handling
2. **Data Cleaning**: Missing value removal, timestamp normalization
3. **Feature Engineering**: Derived metrics (word count, engagement timing)
4. **Analysis Pipeline**: Modular analysis components

### Sentiment Analysis
- **Tool**: VADER Sentiment Analysis (optimized for social media)
- **Metrics**: Compound score (-1 to +1), positive/negative/neutral classification
- **Product-Specific**: Individual sentiment tracking per product mention

### Trend Analysis
- **Method**: TF-IDF vectorization for keyword importance
- **Temporal Analysis**: Time-window comparison for emerging trends
- **Filtering**: Custom stopword lists for social media relevance

## 📊 Visualizations

### Generated Charts
1. **Sentiment Timeline** (`sentiment_timeline.html`)
   - Daily sentiment trends with comment volume overlay
   - Interactive time series with hover details

2. **Product Sentiment Analysis** (`product_sentiment.html`)
   - Bubble chart showing sentiment vs. mention frequency
   - Color-coded sentiment mapping

3. **Engagement Heatmap** (`engagement_heatmap.html`)
   - Day-of-week vs. hour engagement patterns
   - Optimal posting time identification

4. **Keyword Trends** (`keyword_trends.html`)
   - Top keyword evolution over time
   - Trend emergence detection

5. **Word Cloud** (`positive_wordcloud.png`)
   - Visual representation of positive comment themes

## 🎯 Actionable Recommendations

### Content Strategy
1. **Timing Optimization**: Schedule posts for Friday evenings around 19:00
2. **Sentiment Leverage**: Build on 26.9% positive sentiment foundation
3. **Product Focus**: Emphasize products with highest positive sentiment

### Community Engagement
1. **PR Integration**: Leverage high PR campaign engagement
2. **Demand Signals**: Monitor "need" keyword for product opportunities
3. **Scent Marketing**: Highlight fragrance attributes in content

### Performance Monitoring
1. **Sentiment Tracking**: Monitor sentiment score trends monthly
2. **Emerging Trends**: Track new keyword emergence for early adoption
3. **Engagement Patterns**: Adjust posting schedule based on activity data

## 🔬 Extension Proposal

### Phase 1: Enhanced Analytics (Weeks 1-2)
1. **Competitor Analysis**
   - Cross-brand sentiment comparison
   - Market positioning insights
   - Competitive keyword tracking

2. **Influencer Impact Analysis**
   - User-level engagement scoring
   - Influencer content performance
   - Community leader identification

### Phase 2: Advanced ML Features (Weeks 3-4)
3. **Predictive Modeling**
   - Engagement prediction algorithms
   - Viral content likelihood scoring
   - Optimal posting time ML models

4. **Topic Modeling**
   - LDA/BERTopic for theme extraction
   - Automatic topic labeling
   - Topic evolution tracking

### Phase 3: Real-time Dashboard (Weeks 5-6)
5. **Live Monitoring Dashboard**
   - Real-time sentiment tracking
   - Alert system for negative sentiment spikes
   - Interactive filtering and drill-down

6. **API Integration**
   - Instagram Graph API connection
   - Automated data collection
   - Webhook-based real-time updates

### Phase 4: Strategic Intelligence (Weeks 7-8)
7. **Campaign Performance Analytics**
   - A/B testing framework
   - Campaign ROI measurement
   - Cross-platform correlation analysis

8. **Customer Journey Mapping**
   - Comment-to-conversion tracking
   - User lifecycle analysis
   - Retention prediction models

## 🛠️ Development & Testing

### Running Tests
```bash
# Validate data pipeline
python -c "from src.data_processor import DataProcessor; dp = DataProcessor('engagements.csv'); df = dp.load_data(); print('✅ Data loading successful')"

# Test sentiment analysis
python -c "from src.sentiment_analyzer import SentimentAnalyzer; sa = SentimentAnalyzer(); print('✅ Sentiment analyzer ready')"
```

### Code Quality
- **Error Handling**: Comprehensive try-catch blocks with logging
- **Type Hints**: Full type annotation for better maintainability
- **Modular Design**: Separate classes for each analysis component
- **Documentation**: Detailed docstrings and inline comments

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

## 📝 Project Structure

```
scrollmark_two/
├── README.md                 # This file
├── main.py                   # Main analysis pipeline
├── requirements.txt          # Python dependencies
├── engagements.csv          # Raw data (17,812 comments)
├── src/                     # Analysis modules
│   ├── data_processor.py    # Data loading and cleaning
│   ├── sentiment_analyzer.py # Sentiment analysis pipeline
│   ├── trend_analyzer.py    # Trend identification
│   └── visualizer.py        # Chart generation
├── data/                    # Processed data outputs
│   └── processed/
│       └── comments_with_analysis.csv
└── reports/                 # Analysis outputs
    ├── analysis_report.md   # Comprehensive analysis report
    ├── sentiment_timeline.html
    ├── product_sentiment.html
    ├── engagement_heatmap.html
    ├── keyword_trends.html
    └── positive_wordcloud.png
```

## 🎉 Conclusion

This analysis provides @treehut with data-driven insights to optimize their social media strategy. The positive community sentiment (26.9%) and clear engagement patterns offer strong foundations for growth. The modular codebase and comprehensive extension plan ensure scalability for future analytics needs.

---

*Analysis completed on 2025-08-14 | Generated using automated data science pipeline*# scrollmark
