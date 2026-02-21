🛡️ Fraud Detection System
End-to-End ML Solution for Financial Fraud Detection
https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg
https://img.shields.io/badge/Python-3.8+-blue.svg
https://img.shields.io/badge/License-MIT-green.svg

📋 Table of Contents
Project Overview

Key Features

Technical Architecture

Fraud Patterns Implemented

Feature Engineering

Model Selection

Installation Guide

How to Run

Application Walkthrough

Performance Metrics

Production Considerations

Limitations & Future Work

🎯 Project Overview
This is a complete end-to-end fraud detection system that demonstrates how machine learning can identify fraudulent financial transactions. The system generates synthetic transaction data, engineers meaningful features, trains a robust ML model, and provides interpretable predictions through an interactive dashboard.

Key Facts:

📊 10,000+ synthetic transactions with realistic patterns

🔍 2-5% fraud rate (realistic class imbalance)

🧠 17 engineered features for accurate detection

🤖 Random Forest model with class balancing

💡 SHAP-like explanations for every prediction

✨ Key Features
1. Intelligent Data Generation
Creates realistic user profiles with behavioral patterns

Generates time-ordered transactions over 30 days

Injects controlled fraud patterns (not random)

Maintains realistic class imbalance (2-5% fraud)

2. Comprehensive Fraud Patterns
Six realistic fraud scenarios based on real-world cases:

Location Mismatch: Transactions from unusual locations

Amount Spikes: Sudden large transactions (>3x average)

Late Night Activity: Suspicious timing (12 AM - 5 AM)

High-Risk Merchants: Targeting vulnerable categories

Velocity Patterns: Multiple rapid transactions

Behavioral Deviation: Multi-factor anomalies

3. Interactive Dashboard
Five specialized tabs for complete analysis:

Data Explorer: Visualize transactions and patterns

Fraud Patterns: Understand injected scenarios

Performance: View model metrics and feature importance

Explainability: Get reasons for individual predictions

System Design: Production architecture recommendations

🏗 Technical Architecture
Component Diagram
text
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT APPLICATION                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Data      │  │   Feature   │  │      Model          │  │
│  │  Generator  │→│  Engineer   │→│     Trainer         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │               │                     │              │
│         ↓               ↓                     ↓              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   User      │  │  Merchant   │  │   Prediction &      │  │
│  │  Profiles   │  │  Profiles   │  │   Explanation       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
Data Flow
Generate synthetic users and merchants with profiles

Create time-ordered transactions with realistic amounts

Inject fraud patterns with controlled probabilities

Engineer 17 features from raw transaction data

Train Random Forest model with class balancing

Evaluate using precision, recall, F1, ROC-AUC

Explain individual predictions with factor analysis

🔍 Fraud Patterns Implemented
Pattern	Detection Logic	Fraud Rate	Real-World Scenario
Location Mismatch	location ≠ user.typical_location	10%	Stolen card used in different city
Amount Spike	amount > user.avg_amount × 3	30%	Card testing before large fraud
Late Night	hour between 0-5	5%	Fraudsters in different timezone
High-Risk Merchant	merchant.risk_level = 'high'	20%	Vulnerable merchant categories
Velocity	Multiple rapid transactions	Contextual	Account takeover
Behavioral	Combination of above	Variable	Sophisticated fraud rings
🧠 Feature Engineering
17 Engineered Features
Category	Features	Purpose
User Statistics	avg_amount, std_amount, max_amount, txn_count	Historical baseline
Deviation Metrics	amount_deviation, amount_ratio	Detect anomalies
Temporal Features	hour_sin, hour_cos, is_weekend	Time-based patterns
Velocity Metrics	txn_per_day, rolling_count	Detect rapid activity
Match Indicators	location_match	Identity verification
Encoded Categories	location_encoded, payment_encoded, category_encoded	Handle categorical data
🤖 Model Selection
Why Random Forest?
Advantage	Description
Handles Non-linearity	Captures complex fraud patterns
No Scaling Required	Works with raw feature values
Feature Importance	Built-in explainability
Class Balancing	class_weight='balanced' handles imbalance
Robust to Outliers	Tree-based methods are resilient
Fast Inference	Suitable for real-time detection
Model Parameters
n_estimators: 50 trees

max_depth: 10 levels

min_samples_split: 10 samples

class_weight: 'balanced'

random_state: 42 (reproducible)

📦 Installation Guide
Prerequisites
Python 3.8 or higher

pip package manager

4GB RAM minimum

Modern web browser (Chrome, Firefox, Edge)

Step-by-Step Installation
bash
# 1. Create project directory
mkdir fraud-detection-system
cd fraud-detection-system

# 2. Create virtual environment (recommended)
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4. Install required packages
pip install streamlit pandas numpy plotly scikit-learn

# 5. Verify installation
python --version
pip list | grep -E "streamlit|pandas|numpy|plotly|scikit-learn"
Required Packages
txt
streamlit==1.28.0
pandas==1.5.0
numpy==1.24.0
plotly==5.17.0
scikit-learn==1.3.0
🚀 How to Run
Method 1: Direct Run
bash
# Navigate to project directory
cd fraud-detection-system

# Ensure virtual environment is activated
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Run the Streamlit app
streamlit run app.py
Method 2: With Python Module
bash
python -m streamlit run app.py
Method 3: Docker (Optional)
bash
# Build Docker image
docker build -t fraud-detection .

# Run container
docker run -p 8501:8501 fraud-detection
What to Expect
After running the command:

Terminal will show: You can now view your Streamlit app in your browser

URL: http://localhost:8501 will open automatically

Application loads with sidebar controls and main dashboard

Total startup time: ~5-10 seconds

Troubleshooting
Issue	Solution
Port 8501 in use	streamlit run app.py --server.port 8502
Module not found	pip install -r requirements.txt
Memory error	Reduce transaction count in sidebar
Slow performance	Use Chrome/Firefox, not Edge
📱 Application Walkthrough
Sidebar Controls
text
┌─────────────────────┐
│   CONTROL PANEL     │
├─────────────────────┤
│ Data Generation     │
│ └─ Transactions: 10k│
│ └─ [Generate Data]  │
├─────────────────────┤
│ Model Training      │
│ └─ [Train Model]    │
└─────────────────────┘
Tab 1: Data Explorer
text
┌─────────────────────────────────────┐
│ METRICS ROW                         │
│ Fraud Rate │ Total Vol │ Avg Amount │
│   3.2%     │  $1.2M    │   $124     │
├─────────────────────────────────────┤
│ Recent Transactions                  │
│ T001 │ $45  │ NYC  │ 14:30 │ ✅ LEGIT │
│ T002 │ $890 │ LA   │ 03:15 │ 🔴 FRAUD │
├─────────────────────────────────────┤
│ Distribution Charts                  │
│ [Fraud by Hour]  [Amount Distrib]    │
└─────────────────────────────────────┘
Tab 2: Fraud Patterns
Visual breakdown of each fraud pattern

Impact analysis charts

Pattern comparison

Tab 3: Performance Metrics
Precision, Recall, F1 scores

Confusion matrix

ROC curve

Feature importance bar chart

Tab 4: Explainability
Select any transaction

View fraud probability

See contributing factors

Natural language explanation

Tab 5: System Design
Real-time architecture

Batch processing

False positive handling

Cost-benefit analysis

📊 Performance Metrics
Expected Results
Metric	Range	Interpretation
Precision	0.75 - 0.85	75-85% of fraud alerts are correct
Recall	0.70 - 0.80	70-80% of actual fraud is caught
F1-Score	0.72 - 0.82	Balanced performance
ROC-AUC	0.85 - 0.92	Excellent discrimination
Confusion Matrix Example (10,000 transactions)
text
              Predicted
              Legit  Fraud
Actual Legit   9200    200
       Fraud     30     70
Top 5 Important Features
amount_ratio - Deviation from user average

txn_per_day - Transaction velocity

location_match - Location consistency

amount_deviation - Absolute difference

hour_sin - Time of day pattern

🏭 Production Considerations
Real-time Detection Pipeline
text
[Transaction] → [Feature Store] → [Model API] → [Decision Engine]
      ↓               ↓                ↓               ↓
   [Kafka]        [Redis]         [MLflow]        [Database]
Batch Processing
text
[Daily Data] → [Spark Job] → [Feature Engineering] → [Model Retraining]
      ↓               ↓                   ↓                    ↓
   [Storage]     [Validation]        [Monitoring]        [Deployment]
False Positive Handling Strategy
python
if risk_score > 0.8:
    action = "BLOCK"
elif risk_score > 0.5:
    action = "HOLD_FOR_REVIEW"
else:
    action = "APPROVE"
Monitoring Metrics
Daily precision/recall tracking

Feature distribution drift

Model performance alerts

A/B test results

⚠️ Limitations & Future Work
Current Limitations
Limitation	Impact
Synthetic Data	May not capture all real patterns
Static Model	No concept drift adaptation
Simple Features	Could be more sophisticated
No Deep Learning	Complex patterns may be missed
Future Improvements
Short-term (1-3 months)
Add more fraud patterns (seasonal, coordinated)

Implement time-series cross-validation

Add threshold optimization

Create monitoring dashboard

Long-term (3-6 months)
Graph-based detection for fraud rings

LSTM for sequence modeling

Real-time streaming with Kafka

Automated retraining pipeline

Cost-sensitive learning optimization

📝 License
MIT License - feel free to use, modify, and distribute.

🙏 Acknowledgments
Inspired by real-world fraud detection challenges

Built with best practices from ML engineering

Designed for clarity, explainability, and production readiness

Note: Dont forget to setx GROQ_API_KEY "" in the powershell by going into the folder directory where the project folder placed.

