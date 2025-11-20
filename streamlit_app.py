# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import StringIO
from pathlib import Path
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration with custom theme
st.set_page_config(
    page_title="ALS Risk Assessment",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #21618C;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stProgress .st-bo {
        background-color: #3498DB;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .stAlert {
        background-color: #EBF5FB;
        border-left-color: #3498DB;
        padding: 1rem;
        margin: 1rem 0;
    }
    div[data-baseweb="select"] {
        margin-top: 0.5rem;
    }
    .reportview-container {
        background-color: #F8F9F9;
    }
    div.stDataFrame {
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-radius: 0.5rem;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header with professional styling
st.title("🧬 ALS Risk Assessment Platform")
st.markdown("---")

# Sidebar for controls
st.sidebar.header("📊 Control Panel")

# Model loading
with st.sidebar.expander("🤖 Model Information", expanded=False):
    model_dir = "models"
    model_path = Path(model_dir)/"xgb_als_model.pkl"
    scaler_path = Path(model_dir)/"scaler.pkl"
    feat_path = Path(model_dir)/"feature_names.pkl"

    if not model_path.exists() or not scaler_path.exists() or not feat_path.exists():
        st.error("Model/scaler/feature_names not found. Train the model first (run train_tabular.py).")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feat_path)
    st.info(f"Model trained using XGBoost with {len(feature_names)} features")

# Data Input Section
st.sidebar.markdown("### 📥 Data Input")
uploaded = st.sidebar.file_uploader("Upload CSV test file", type=["csv"])

# Mock Data Generation
# with st.sidebar.expander("🔄 Generate Mock Data", expanded=False):
#     n = st.number_input("Number of samples", min_value=1, max_value=100, value=10)
#     if st.button("Generate Data"):
#         X = np.random.normal(0,1,(n, len(feature_names)))
#         df_mock = pd.DataFrame(X, columns=feature_names)
#         csv_buf = StringIO()
#         df_mock.to_csv(csv_buf, index=False)
#         csv_buf.seek(0)
#         uploaded = csv_buf
#         st.success("✅ Mock data generated successfully")

if uploaded:
    # Load and process data
    df = pd.read_csv(uploaded)
    
    # Check for label column with multiple possible names
    label_column = None
    possible_label_columns = ['label', 'Diagnosis(ALS)', 'Diagnosis (ALS)', 'Diagnosis', 'ALS']
    
    # First try exact match
    for col in possible_label_columns:
        if col in df.columns:
            label_column = col
            break
    
    # If no exact match, try case-insensitive match and handle spaces
    if label_column is None:
        df_cols_lower = [col.lower().replace(' ', '') for col in df.columns]
        for col in df.columns:
            if col.lower().replace(' ', '') in [p.lower().replace(' ', '') for p in possible_label_columns]:
                label_column = col
                break
    
    labels_present = label_column is not None
    if labels_present:
        y = df[label_column].astype(int)
    
    # Feature processing and data validation
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        st.warning(f"⚠️ Some required features are missing in the uploaded file: {', '.join(missing_features[:5])}{'...' if len(missing_features) > 5 else ''}")
        st.info("💡 The model will use default values (0) for missing features. This may affect prediction accuracy.")
    
    X = pd.DataFrame(0, index=df.index, columns=feature_names)
    for c in feature_names:
        if c in df.columns:
            X[c] = df[c]
    Xs = scaler.transform(X.values)
    proba = model.predict_proba(Xs)[:,1]
    df['ALS_Risk_Probability'] = (proba*100).round(2)
    
    # Check for required columns and inform user
    # Handle age column with case-insensitive check
    age_column = None
    for col in df.columns:
        if col.lower() == 'age':
            age_column = col
            break
    
    has_age_data = age_column is not None
    if not has_age_data:
        st.info("ℹ️ Age data is not present in the uploaded file. Some visualizations will be limited. To see age-based analysis, please include an 'age' column in your data.")
    
    if not labels_present:
        st.info("ℹ️ Label data is not present in the uploaded file. To see model performance metrics, please include one of these columns with actual ALS diagnoses (0 or 1): 'label', 'Diagnosis(ALS)', 'Diagnosis (ALS)', 'Diagnosis', or 'ALS'.")
    
    # Main content area
    st.markdown("### 📋 Results Overview")
    
    # Detect sex column (case-insensitive)
    sex_column = None
    for col in df.columns:
        if col.lower() == 'sex':
            sex_column = col
            break
    has_sex_data = sex_column is not None

    # Filters in expander (range-based filters + reset)
    with st.expander("🔍 Filter Options", expanded=False):
        # Probability range slider
        prob_low, prob_high = st.slider(
            "Risk Probability (%)",
            0,
            100,
            (0, 100),
            key='prob_range'
        )

        # Age range slider (if age present)
        if has_age_data:
            try:
                amin = int(df[age_column].min())
                amax = int(df[age_column].max())
            except Exception:
                amin, amax = 0, 100
            age_low, age_high = st.slider(
                "Age range",
                amin,
                amax,
                (amin, amax),
                key='age_range'
            )
        else:
            age_low, age_high = None, None

        # Sex filter (if present)
        if has_sex_data:
            sex_options = ['All'] + sorted(df[sex_column].dropna().astype(str).unique().tolist())
            sex_choice = st.selectbox("Sex", sex_options, index=0, key='sex_filter')
        else:
            sex_choice = 'All'

        # Reset filters button
        # if st.button("Reset filters"):
        #     # reset session state values
        #     st.session_state['prob_range'] = (0, 100)
        #     if 'age_range' in st.session_state:
        #         st.session_state['age_range'] = (int(df[age_column].min()) if has_age_data else 0, int(df[age_column].max()) if has_age_data else 100)
        #     if 'sex_filter' in st.session_state:
        #         st.session_state['sex_filter'] = 'All'

    # Apply filters
    # read current filter values from session_state (slider writes there)
    prob_low, prob_high = st.session_state.get('prob_range', (0, 100))
    mask = (df['ALS_Risk_Probability'] >= prob_low) & (df['ALS_Risk_Probability'] <= prob_high)

    if has_age_data and 'age_range' in st.session_state:
        age_low, age_high = st.session_state.get('age_range', (None, None))
        if age_low is not None and age_high is not None:
            mask &= (df[age_column] >= age_low) & (df[age_column] <= age_high)

    if has_sex_data:
        sex_choice = st.session_state.get('sex_filter', 'All')
        if sex_choice and sex_choice != 'All':
            mask &= (df[sex_column].astype(str) == str(sex_choice))

    filtered_df = df[mask].copy()
    
    # Results layout
        
        # Summary table with ID and Risk Probability only
    st.markdown("#### 🎯 Risk Probability Summary")
    risk_summary = filtered_df[['ALS_Risk_Probability']].copy()
    risk_summary.index.name = 'Patient ID'
    risk_summary = risk_summary.reset_index()
    st.dataframe(
            risk_summary.style
            .background_gradient(
                subset=['ALS_Risk_Probability'],
                cmap='RdYlBu_r',
                vmin=0,
                vmax=100
            )
            .format({'ALS_Risk_Probability': '{:.2f}%'})
            .set_properties(**{
                'font-size': '1.1rem',
                'text-align': 'center',
                'padding': '0.5rem'
            }),
            height=300
        )
    
    results_col1, results_col2 = st.columns([2, 1])
    
    with results_col1:
        # Show filtered data with enhanced styling
        st.markdown("#### 📊 Patient Risk Assessment")
        st.dataframe(
            filtered_df.style
            .background_gradient(
                subset=['ALS_Risk_Probability'],
                cmap='RdYlBu_r',
                vmin=0,
                vmax=100
            )
            .format({'ALS_Risk_Probability': '{:.2f}%'}),
            height=400
        )

    st.markdown("### 📊 Risk Distribution")
        # Create histogram of probabilities
    fig = px.histogram(
            df,
            x='ALS_Risk_Probability',
            nbins=20,
            title='Distribution of ALS Risk Probabilities',
            labels={'ALS_Risk_Probability': 'Risk Probability (%)'}
        )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key='hist_prob')
    
    if labels_present:
        preds = (proba >= 0.5).astype(int)
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs([
            "📊 Key Metrics", 
            "🎯 Confusion Matrix", 
            "📑 Detailed Report"
        ])
        
        with tab1:
            # Calculate metrics
            metrics = {
                "Accuracy": accuracy_score(y, preds),
                "Recall": recall_score(y, preds, zero_division=0),
                "ROC AUC": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else None
            }
            
            # Display metrics in columns with custom styling
            metric_cols = st.columns(len(metrics))
            for col, (metric_name, value) in zip(metric_cols, metrics.items()):
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3 style="color: #2E86C1; margin-bottom: 0.5rem;">{metric_name}</h3>
                            <p style="font-size: 2rem; font-weight: bold; margin: 0;">
                                {f"{value:.3f}" if value is not None else "N/A"}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        with tab2:
            # Confusion Matrix (responsive Plotly heatmap)
            cm = confusion_matrix(y, preds)
            cm_labels = ['No ALS', 'ALS']
            fig = px.imshow(
                cm,
                labels=dict(x='Predicted', y='Actual', color='Count'),
                x=cm_labels,
                y=cm_labels,
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig.update_layout(title='Confusion Matrix', xaxis_side='top')
            fig.update_traces(textfont_size=16)
            st.plotly_chart(fig, use_container_width=True, key='cm_tab')
        
        with tab3:
            # Detailed Classification Report
            st.markdown("#### 📋 Detailed Classification Report")
            report = classification_report(y, preds)
            st.code(report, language='text')
    
    # Risk Analysis Section
    st.markdown("### 🎯 Risk Analysis Dashboard")
    
    # Create two columns for the analysis
    analysis_col1, analysis_col2 = st.columns([1, 1])
    
    with analysis_col1:
        if has_age_data:
            st.markdown("#### 📊 Risk Probability by Age")
            # Calculate average risk probability by age
            age_risk = df.groupby(age_column)['ALS_Risk_Probability'].agg(['mean', 'count']).reset_index()
            age_risk.columns = ['Age', 'Average Risk', 'Patient Count']
            
            fig = px.bar(
                age_risk,
                x='Age',
                y='Average Risk',
                title='Average Risk Probability by Age',
                labels={
                    'Age': 'Age (years)',
                    'Average Risk': 'Average Risk Probability (%)'
                },
                hover_data=['Patient Count']
            )
            fig.update_layout(
                template='plotly_white',
                showlegend=False,
                yaxis_title='Average Risk Probability (%)',
                xaxis_title='Age (years)',
                hovermode='x'
            )
            fig.update_traces(
                marker_color='#3498DB',
                hovertemplate="<br>".join([
                    "Age: %{x}",
                    "Average Risk: %{y:.1f}%",
                    "Patient Count: %{customdata[0]}",
                    "<extra></extra>"
                ])
            )
            st.plotly_chart(fig, use_container_width=True, key='age_bar')
        else:
            st.markdown("#### 📊 Alternative Risk Analysis")
            # Create risk categories
            risk_categories = pd.cut(
                df['ALS_Risk_Probability'],
                bins=[0, 25, 50, 75, 100],
                labels=['Low (0-25%)', 'Moderate (25-50%)', 'High (50-75%)', 'Very High (75-100%)']
            )
            risk_dist = risk_categories.value_counts().sort_index()
            
            fig = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                title='Distribution of Risk Categories',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True, key='risk_pie')
    
    with analysis_col2:
        if labels_present:
            st.markdown("#### 🎯 Model Performance Metrics")
            
            # ROC Score
            roc = roc_auc_score(y, proba)
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3 style="color: #2E86C1; margin-bottom: 0.5rem;">ROC AUC Score</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0;">
                        {roc:.3f}
                    </p>
                    <p style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">
                        Score > 0.8 indicates good model performance
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Confusion Matrix (responsive Plotly heatmap)
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y, preds)
            cm_labels = ['No ALS', 'ALS']
            fig = px.imshow(
                cm,
                labels=dict(x='Predicted', y='Actual', color='Count'),
                x=cm_labels,
                y=cm_labels,
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig.update_layout(title='Confusion Matrix', xaxis_side='top')
            fig.update_traces(textfont_size=16)
            st.plotly_chart(fig, use_container_width=True, key='cm_perf')
            
            # Add explanation of the confusion matrix
            st.markdown("""
            **Understanding the Confusion Matrix:**
            - Top Left: Correctly predicted non-ALS cases
            - Bottom Right: Correctly predicted ALS cases
            - Top Right: False ALS predictions
            - Bottom Left: Missed ALS cases
            """)
        else:
            st.markdown("#### 📈 Risk Distribution Overview")
            # Create a violin plot of risk probabilities
            fig = px.violin(
                df,
                y='ALS_Risk_Probability',
                box=True,
                title='Risk Probability Distribution Pattern',
                labels={'ALS_Risk_Probability': 'Risk Probability (%)'}
            )
            fig.update_layout(
                template='plotly_white',
                showlegend=False,
                yaxis_title='Risk Probability (%)'
            )
            st.plotly_chart(fig, use_container_width=True, key='violin_plot')
            
            # Add summary statistics
            st.markdown("#### 📊 Risk Level Summary")
            risk_levels = pd.cut(
                df['ALS_Risk_Probability'],
                bins=[0, 20, 40, 60, 80, 100],
                labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
            ).value_counts().sort_index()
            
            risk_df = pd.DataFrame({
                'Risk Level': risk_levels.index,
                'Count': risk_levels.values,
                'Percentage': (risk_levels.values / len(df) * 100).round(1)
            })
            
            st.dataframe(
                risk_df.style
                .background_gradient(subset=['Percentage'], cmap='Blues')
                .format({'Percentage': '{:.1f}%'}),
                height=200
            )
