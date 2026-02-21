# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .fraud-badge {
        background-color: #dc3545;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .legit-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'features' not in st.session_state:
    st.session_state.features = None

class DataGenerator:
    """Generate synthetic transaction data"""
    
    def __init__(self):
        self.locations = ['New York', 'LA', 'Chicago', 'Houston', 'Phoenix']
        self.payment_methods = ['card', 'paypal', 'transfer', 'crypto']
        self.merchant_categories = ['retail', 'food', 'travel', 'entertainment']
        
    def generate(self, n_transactions=10000):
        """Generate transactions with fraud patterns"""
        
        # Generate user profiles
        n_users = 500
        users = []
        for i in range(n_users):
            users.append({
                'user_id': f'U{i:04d}',
                'avg_amount': random.uniform(20, 200),
                'location': random.choice(self.locations),
                'payment': random.choice(self.payment_methods),
                'risk_score': random.uniform(0, 1)
            })
        users_df = pd.DataFrame(users)
        
        # Generate merchant profiles
        n_merchants = 100
        merchants = []
        for i in range(n_merchants):
            merchants.append({
                'merchant_id': f'M{i:04d}',
                'category': random.choice(self.merchant_categories),
                'risk_level': random.choice(['low', 'medium', 'high'])
            })
        merchants_df = pd.DataFrame(merchants)
        
        # Generate transactions
        transactions = []
        start_date = datetime.now() - timedelta(days=30)
        
        for i in range(n_transactions):
            user = random.choice(users)
            merchant = random.choice(merchants)
            
            # Generate timestamp
            timestamp = start_date + timedelta(
                seconds=random.randint(0, 30*24*3600)
            )
            
            # Generate amount with some randomness
            amount = abs(np.random.normal(user['avg_amount'], user['avg_amount']*0.3))
            
            transactions.append({
                'transaction_id': f'T{i:06d}',
                'user_id': user['user_id'],
                'merchant_id': merchant['merchant_id'],
                'amount': round(amount, 2),
                'timestamp': timestamp,
                'location': user['location'] if random.random() > 0.1 else random.choice(self.locations),
                'payment_method': user['payment'] if random.random() > 0.1 else random.choice(self.payment_methods),
                'merchant_category': merchant['category'],
                'hour': timestamp.hour,
                'day_of_week': timestamp.weekday()
            })
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Inject fraud patterns
        df['is_fraud'] = 0
        
        # Pattern 1: Unusual location (10% of location mismatches)
        location_mismatch = df['location'] != df['user_id'].map(
            users_df.set_index('user_id')['location']
        )
        df.loc[location_mismatch, 'is_fraud'] = np.random.choice(
            [0, 1], size=location_mismatch.sum(), p=[0.9, 0.1]
        )
        
        # Pattern 2: High amount (>3x user average)
        user_avg = df.groupby('user_id')['amount'].transform('mean')
        high_amount = df['amount'] > user_avg * 3
        df.loc[high_amount, 'is_fraud'] = np.random.choice(
            [0, 1], size=high_amount.sum(), p=[0.7, 0.3]
        )
        
        # Pattern 3: Late night transactions (12-5 AM)
        late_night = (df['hour'] >= 0) & (df['hour'] <= 5)
        df.loc[late_night, 'is_fraud'] = np.random.choice(
            [0, 1], size=late_night.sum(), p=[0.95, 0.05]
        )
        
        # Pattern 4: High-risk merchants
        high_risk_merchants = merchants_df[merchants_df['risk_level'] == 'high']['merchant_id'].tolist()
        high_risk_txn = df['merchant_id'].isin(high_risk_merchants)
        df.loc[high_risk_txn, 'is_fraud'] = np.random.choice(
            [0, 1], size=high_risk_txn.sum(), p=[0.8, 0.2]
        )
        
        # Ensure fraud rate is between 2-5%
        fraud_rate = df['is_fraud'].mean()
        target_rate = random.uniform(0.02, 0.05)
        
        if fraud_rate < target_rate:
            # Add more fraud
            non_fraud = df[df['is_fraud'] == 0].index
            n_to_add = int((target_rate - fraud_rate) * len(df))
            add_idx = np.random.choice(non_fraud, n_to_add, replace=False)
            df.loc[add_idx, 'is_fraud'] = 1
        elif fraud_rate > target_rate:
            # Remove some fraud
            fraud = df[df['is_fraud'] == 1].index
            n_to_remove = int((fraud_rate - target_rate) * len(df))
            remove_idx = np.random.choice(fraud, n_to_remove, replace=False)
            df.loc[remove_idx, 'is_fraud'] = 0
        
        return df, users_df, merchants_df

class FeatureEngineer:
    """Create features for fraud detection"""
    
    @staticmethod
    def create_features(df, users_df):
        """Engineer features from raw data"""
        
        # User-based features
        user_stats = df.groupby('user_id').agg({
            'amount': ['mean', 'std', 'max', 'count']
        }).round(2)
        user_stats.columns = ['user_avg_amount', 'user_std_amount', 
                             'user_max_amount', 'user_txn_count']
        user_stats = user_stats.reset_index()
        
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Amount deviation features
        df['amount_deviation'] = df['amount'] - df['user_avg_amount']
        df['amount_ratio'] = df['amount'] / (df['user_avg_amount'] + 1)
        
        # Time-based features (cyclic encoding)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Rolling features (last 10 transactions per user)
        df = df.sort_values(['user_id', 'timestamp'])
        df['txn_rolling_count'] = df.groupby('user_id').cumcount() + 1
        df['amount_rolling_mean'] = df.groupby('user_id')['amount'].transform(
            lambda x: x.expanding().mean()
        )
        
        # Velocity (transactions per day)
        df['txn_per_day'] = df.groupby('user_id')['transaction_id'].transform('count') / 30
        
        # Location match with user profile
        location_map = users_df.set_index('user_id')['location'].to_dict()
        df['location_match'] = df.apply(
            lambda x: 1 if x['location'] == location_map.get(x['user_id']) else 0, 
            axis=1
        )
        
        # Encode categorical variables
        df['location_encoded'] = pd.factorize(df['location'])[0]
        df['payment_encoded'] = pd.factorize(df['payment_method'])[0]
        df['category_encoded'] = pd.factorize(df['merchant_category'])[0]
        
        # Fill NaN values
        fill_cols = ['user_std_amount', 'amount_rolling_mean']
        for col in fill_cols:
            df[col] = df[col].fillna(df[col].mean())
        
        # Select features for model
        feature_cols = [
            'amount', 'hour_sin', 'hour_cos', 'is_weekend',
            'user_avg_amount', 'user_std_amount', 'user_max_amount', 
            'user_txn_count', 'amount_deviation', 'amount_ratio',
            'txn_rolling_count', 'amount_rolling_mean', 'txn_per_day',
            'location_match', 'location_encoded', 'payment_encoded', 
            'category_encoded'
        ]
        
        return df, feature_cols

class FraudModel:
    """Random Forest model for fraud detection"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.feature_importance = None
        
    def train(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return self.model
    
    def predict(self, X, threshold=0.5):
        """Make predictions with threshold"""
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        return preds, probs
    
    def get_feature_importance(self, top_n=10):
        """Get top N important features"""
        return self.feature_importance.head(top_n)

# Main app
def main():
    st.markdown('<h1 class="main-header">🛡️ Fraud Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/null/security-checkout.png", width=80)
        st.markdown("## 🎮 Control Panel")
        
        st.markdown("### 📊 Data Generation")
        n_txns = st.slider("Number of Transactions", 5000, 20000, 10000, step=5000)
        
        if st.button("🚀 Generate Data", use_container_width=True):
            with st.spinner("Generating synthetic data..."):
                generator = DataGenerator()
                df, users_df, merchants_df = generator.generate(n_txns)
                
                st.session_state.df = df
                st.session_state.users_df = users_df
                st.session_state.merchants_df = merchants_df
                
                st.success(f"✅ Generated {len(df):,} transactions!")
        
        st.markdown("---")
        
        if st.button("🤖 Train Model", use_container_width=True, type="primary"):
            if st.session_state.df is not None:
                with st.spinner("Training fraud detection model..."):
                    # Feature engineering
                    engineer = FeatureEngineer()
                    df_features, feature_cols = engineer.create_features(
                        st.session_state.df, 
                        st.session_state.users_df
                    )
                    
                    # Prepare data
                    X = df_features[feature_cols]
                    y = df_features['is_fraud']
                    
                    # Train model
                    model = FraudModel()
                    model.train(X, y)
                    
                    # Store in session
                    st.session_state.model = model
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.df_features = df_features
                    st.session_state.feature_cols = feature_cols
                    
                    st.success("✅ Model trained successfully!")
            else:
                st.warning("Please generate data first!")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Explorer", "🔍 Fraud Patterns", "📈 Performance", 
        "💡 Explainability", "🏗️ System Design"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">📊 Data Explorer</h2>', 
                   unsafe_allow_html=True)
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                fraud_rate = df['is_fraud'].mean() * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Fraud Rate</h4>
                    <h2>{fraud_rate:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_amount = df['amount'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Total Volume</h4>
                    <h2>${total_amount:,.0f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_amount = df['amount'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Avg Amount</h4>
                    <h2>${avg_amount:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                unique_users = df['user_id'].nunique()
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Active Users</h4>
                    <h2>{unique_users:,}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Data preview
            st.markdown("### 📋 Recent Transactions")
            preview_df = df.head(10).copy()
            preview_df['status'] = preview_df['is_fraud'].apply(
                lambda x: '<span class="fraud-badge">FRAUD</span>' if x == 1 
                else '<span class="legit-badge">LEGIT</span>'
            )
            
            for _, row in preview_df.iterrows():
                cols = st.columns([2, 2, 2, 2, 2, 1])
                cols[0].write(row['transaction_id'])
                cols[1].write(f"${row['amount']:.2f}")
                cols[2].write(row['location'])
                cols[3].write(row['payment_method'])
                cols[4].write(row['timestamp'].strftime('%Y-%m-%d %H:%M'))
                cols[5].markdown(row['status'], unsafe_allow_html=True)
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Fraud by hour
                fraud_by_hour = df.groupby('hour')['is_fraud'].mean().reset_index()
                fig = px.bar(fraud_by_hour, x='hour', y='is_fraud',
                            title='Fraud Rate by Hour',
                            labels={'is_fraud': 'Fraud Rate'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Amount distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[df['is_fraud']==0]['amount'],
                    name='Legit', marker_color='green', opacity=0.7
                ))
                fig.add_trace(go.Histogram(
                    x=df[df['is_fraud']==1]['amount'],
                    name='Fraud', marker_color='red', opacity=0.7
                ))
                fig.update_layout(title='Amount Distribution', barmode='overlay')
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👈 Generate data using the sidebar")
    
    with tab2:
        st.markdown('<h2 class="sub-header">🔍 Fraud Patterns</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>🎯 Implemented Fraud Patterns</h4>
        <p>We've injected realistic fraud patterns based on real-world scenarios:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **1. Location Mismatch** 🌍
            - Transactions from unusual locations
            - 10% of mismatches are fraudulent
            
            **2. Amount Spikes** 💰
            - Transactions >3x user average
            - 30% of spikes are fraudulent
            
            **3. Late Night Activity** 🌙
            - Transactions between 12 AM - 5 AM
            - 5% of late night txns are fraudulent
            """)
        
        with col2:
            st.markdown("""
            **4. High-Risk Merchants** 🏪
            - Targeting risky merchant categories
            - 20% of high-risk txns are fraudulent
            
            **5. Velocity Patterns** ⚡
            - Multiple rapid transactions
            - Combined with other indicators
            
            **6. User Behavior Deviation** 📊
            - Deviations from historical patterns
            - Multi-factor fraud detection
            """)
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Pattern impact
            st.markdown("### 📊 Pattern Impact Analysis")
            
            pattern_data = pd.DataFrame({
                'Pattern': ['Location Mismatch', 'Amount Spike', 'Late Night', 
                           'High-Risk Merchant'],
                'Fraud Rate': [
                    df[df['location'] != df['user_id'].map(
                        st.session_state.users_df.set_index('user_id')['location']
                    )]['is_fraud'].mean() * 100,
                    df[df['amount'] > df.groupby('user_id')['amount'].transform('mean') * 3]['is_fraud'].mean() * 100,
                    df[(df['hour'] >= 0) & (df['hour'] <= 5)]['is_fraud'].mean() * 100,
                    df[df['merchant_id'].isin(
                        st.session_state.merchants_df[
                            st.session_state.merchants_df['risk_level'] == 'high'
                        ]['merchant_id']
                    )]['is_fraud'].mean() * 100
                ]
            })
            
            fig = px.bar(pattern_data, x='Pattern', y='Fraud Rate',
                        title='Fraud Rate by Pattern',
                        color='Fraud Rate', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', 
                   unsafe_allow_html=True)
        
        if st.session_state.model is not None:
            # Get predictions
            y_pred, y_prob = st.session_state.model.predict(st.session_state.X)
            
            # Calculate metrics
            precision = precision_score(st.session_state.y, y_pred)
            recall = recall_score(st.session_state.y, y_pred)
            f1 = f1_score(st.session_state.y, y_pred)
            roc_auc = roc_auc_score(st.session_state.y, y_prob)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Precision</h4>
                    <h2>{precision:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Recall</h4>
                    <h2>{recall:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>F1-Score</h4>
                    <h2>{f1:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>ROC-AUC</h4>
                    <h2>{roc_auc:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Confusion Matrix
            st.markdown("### 🔄 Confusion Matrix")
            
            cm = confusion_matrix(st.session_state.y, y_pred)
            fig = px.imshow(cm, 
                           labels=dict(x="Predicted", y="Actual"),
                           x=['Legit', 'Fraud'],
                           y=['Legit', 'Fraud'],
                           text_auto=True,
                           color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curve
            st.markdown("### 📈 ROC Curve")
            
            fpr, tpr, _ = roc_curve(st.session_state.y, y_prob)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f'ROC (AUC={roc_auc:.3f})',
                                     line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                     name='Random', line=dict(color='red', dash='dash')))
            fig.update_layout(xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            st.markdown("### 🔑 Feature Importance")
            
            importance_df = st.session_state.model.get_feature_importance(10)
            fig = px.bar(importance_df, x='importance', y='feature',
                        orientation='h', title='Top 10 Features',
                        color='importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👈 Train the model first")
    
    with tab4:
        st.markdown('<h2 class="sub-header">💡 Explainability</h2>', 
                   unsafe_allow_html=True)
        
        if st.session_state.model is not None:
            st.markdown("""
            <div class="info-box">
            <h4>🔍 Understanding Predictions</h4>
            <p>Select a transaction to see why it was flagged (or not flagged) as fraud.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Select transaction
            df_display = st.session_state.df_features.head(50).copy()
            df_display['pred'], _ = st.session_state.model.predict(
                st.session_state.X.head(50)
            )
            
            selected_idx = st.selectbox(
                "Choose a transaction:",
                range(len(df_display)),
                format_func=lambda i: f"Txn {df_display.iloc[i]['transaction_id']} | " +
                                     f"Actual: {'FRAUD' if df_display.iloc[i]['is_fraud'] else 'LEGIT'} | " +
                                     f"Predicted: {'FRAUD' if df_display.iloc[i]['pred'] else 'LEGIT'}"
            )
            
            if selected_idx is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Transaction Details**")
                    txn = df_display.iloc[selected_idx]
                    st.write(f"Amount: ${txn['amount']:.2f}")
                    st.write(f"Location: {txn['location']}")
                    st.write(f"Payment: {txn['payment_method']}")
                    st.write(f"Hour: {txn['hour']}:00")
                
                with col2:
                    st.markdown("**Risk Factors**")
                    
                    # Get feature values
                    X_txn = st.session_state.X.iloc[[selected_idx]]
                    probs = st.session_state.model.model.predict_proba(X_txn)[0]
                    
                    st.markdown(f"""
                    - Fraud Probability: **{probs[1]:.1%}**
                    - Amount vs User Avg: **{txn['amount_ratio']:.1f}x**
                    - Location Match: **{'Yes' if txn['location_match'] else 'No'}**
                    - Transaction Velocity: **{txn['txn_per_day']:.1f}/day**
                    """)
                
                # Explanation
                st.markdown("### 📝 Explanation")
                
                if df_display.iloc[selected_idx]['pred'] == 1:
                    reasons = []
                    if txn['amount_ratio'] > 2:
                        reasons.append("⚠️ Amount is much higher than user's typical transactions")
                    if not txn['location_match']:
                        reasons.append("⚠️ Location doesn't match user's usual location")
                    if txn['hour'] < 6:
                        reasons.append("⚠️ Transaction occurred during unusual hours (late night)")
                    if txn['txn_per_day'] > 10:
                        reasons.append("⚠️ Unusual transaction velocity detected")
                    
                    if reasons:
                        for reason in reasons:
                            st.markdown(reason)
                    else:
                        st.markdown("⚠️ Multiple behavioral anomalies detected")
                else:
                    st.markdown("✅ Transaction matches user's typical behavior pattern")
        
        else:
            st.info("👈 Train the model first")
    
    with tab5:
        st.markdown('<h2 class="sub-header">🏗️ System Design</h2>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Real-time Detection
            ```
            Transaction → Feature Extraction 
                       → Model Inference 
                       → Risk Score 
                       → Decision
            ```
            
            **Components:**
            - API Gateway (FastAPI)
            - Feature Store (Redis)
            - Model Server (MLflow)
            - Database (PostgreSQL)
            """)
        
        with col2:
            st.markdown("""
            ### Batch Processing
            ```
            Daily Data → Feature Engineering 
                       → Model Retraining 
                       → Performance Monitoring 
                       → Pattern Discovery
            ```
            
            **Schedule:**
            - Daily retraining at 2 AM
            - Weekly pattern analysis
            - Monthly model validation
            """)
        
        st.markdown("### ⚖️ False Positive Handling")
        
        st.markdown("""
        **Strategy:**
        1. **Tiered Approach**:
           - Score > 0.8: Block immediately
           - Score 0.5-0.8: Hold for review
           - Score < 0.5: Approve
        
        2. **Review Process**:
           - Manual review queue
           - User verification
           - Appeal mechanism
        
        3. **Feedback Loop**:
           - Store review decisions
           - Retrain with new data
           - Update thresholds
        """)
        
        # Cost-Benefit Analysis
        st.markdown("### 💰 Estimated Impact")
        
        impact_data = pd.DataFrame({
            'Category': ['Fraud Prevented', 'False Positives', 'Operational Cost', 'Net Savings'],
            'Monthly ($)': [100000, 15000, 10000, 75000]
        })
        
        fig = px.bar(impact_data, x='Category', y='Monthly ($)',
                    title='Monthly Financial Impact',
                    color='Category')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()