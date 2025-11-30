# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, recall_score, confusion_matrix

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ALS Risk Assessment Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CUSTOM CSS & THEME
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Main Layout Background */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card Styling */
    .css-1r6slb0 {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e6e9ef;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1f2937;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Button */
    .stButton>button {
        background-color: #0ea5e9;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #0284c7;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load model, scaler, features, and static metrics."""
    model_dir = Path("models")
    try:
        model = joblib.load(model_dir / "xgb_als_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        features = joblib.load(model_dir / "feature_names.pkl")
        
        metrics_path = model_dir / "metrics.json"
        static_metrics = None
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                static_metrics = json.load(f)
                
        return model, scaler, features, static_metrics
    except FileNotFoundError as e:
        st.error(f"Essential model files missing: {e}")
        return None, None, None, None

def create_gauge_chart(value, title):
    """Create a simple gauge chart for metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#0ea5e9"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
        }
    ))
    # Increase top margin so the chart title doesn't underlap surrounding headers
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=60, b=20))
    return fig

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    # st.image("https://img.icons8.com/fluency/96/dna-helix.png", width=60)
    st.title("Control Panel")
    
    # st.subheader("1. Model Status")
    model, scaler, feature_names, static_metrics = load_artifacts()
    # if model:
    #     st.success(f"✅ Model Loaded ({len(feature_names)} features)")
    #     if static_metrics:
    #         with st.expander("View Training Metrics"):
    #             st.json(static_metrics)
    
    # st.subheader("2. Data Input")
    uploaded_file = st.file_uploader("Upload Patient Data (CSV)", type=['csv'])
    
    # st.info("ℹ️ **Note:** Ensure your CSV contains the required biomarker columns for accurate prediction.")

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------

# Hero Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🧬 ALS Risk Assessment Platform")
    st.markdown("Artificial Intelligence system for early detection and risk stratification of Amyotrophic Lateral Sclerosis.")

if not uploaded_file:
    st.divider()
    st.markdown("""
    ### 👋 Welcome
    Please upload a CSV file in the sidebar to begin analysis. 
    
    **The system will generate:**
    * Individual risk probabilities
    * Population-level risk distribution
    * Comparative analysis by age and gender
    * Performance validation (if diagnostic labels are provided)
    """)
else:
    # -------------------------------------------------------------------------
    # DATA PROCESSING
    # -------------------------------------------------------------------------
    try:
        df = pd.read_csv(uploaded_file)
        
        # Auto-detect label column (no manual selection)
        label_column = None
        possible_label_columns = ['label', 'Diagnosis(ALS)', 'Diagnosis (ALS)', 'Diagnosis', 'ALS']

        # First try exact match
        for col in possible_label_columns:
            if col in df.columns:
                label_column = col
                break

        # If no exact match, try case-insensitive match and handle spaces
        if label_column is None:
            normalized_targets = [p.lower().replace(' ', '') for p in possible_label_columns]
            for col in df.columns:
                if col.lower().replace(' ', '') in normalized_targets:
                    label_column = col
                    break

        labels_present = label_column is not None
        if labels_present:
            try:
                y = df[label_column].astype(int)
            except Exception:
                st.warning(f"Could not coerce label column `{label_column}` to integer 0/1 labels. Please ensure labels are numeric or rename the column.")
                labels_present = False

        # Backwards-compatible names used in other parts of the app
        has_labels = labels_present
        y_true = y if labels_present else None

        # --- Feature Prep ---
        X = pd.DataFrame(0, index=df.index, columns=feature_names)
        present_features = [c for c in feature_names if c in df.columns]
        missing_features = [c for c in feature_names if c not in df.columns]
        
        if missing_features:
            st.warning(f"⚠️ Imputing {len(missing_features)} missing features with default values (0).")
            
        X[present_features] = df[present_features]
        
        # --- Prediction ---
        X_scaled = scaler.transform(X.values)
        probas = model.predict_proba(X_scaled)[:, 1]
        
        # Append results
        df['ALS_Risk_Probability'] = (probas * 100).round(2)
        # Create risk categories; include_lowest so 0 maps into the first bin
        df['Risk_Category'] = pd.cut(
            df['ALS_Risk_Probability'], 
            bins=[0, 25, 50, 75, 100], 
            labels=['Low', 'Moderate', 'High', 'Very High'],
            include_lowest=True,
        )
        # Convert to plain object dtype so we can assign a new category label 'No Risk'
        df['Risk_Category'] = df['Risk_Category'].astype(object)
        # Explicitly label exact 0% as 'No Risk' and fill any remaining NaNs
        df.loc[df['ALS_Risk_Probability'] == 0, 'Risk_Category'] = 'No Risk'
        df['Risk_Category'] = df['Risk_Category'].fillna('No Risk')
        
        # --- Helper Columns ---
        age_col = next((c for c in df.columns if c.lower() == 'age'), None)
        sex_col = next((c for c in df.columns if c.lower() == 'sex'), None)

        # -------------------------------------------------------------------------
        # DASHBOARD LAYOUT
        # -------------------------------------------------------------------------
        st.divider()
        
        # Top Level KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Patients", len(df))
        kpi2.metric("Avg. Risk Probability", f"{df['ALS_Risk_Probability'].mean():.1f}%")
        
        high_risk_count = len(df[df['ALS_Risk_Probability'] > 75])
        delta_color = "inverse" if high_risk_count > 0 else "normal"
        kpi3.metric("High Risk Cases (>75%)", high_risk_count, delta_color="inverse")
        
        if has_labels:
            acc = accuracy_score(y_true, (probas >= 0.5).astype(int))
            kpi4.metric("Model Accuracy (Current Batch)", f"{acc*100:.1f}%")
        else:
            kpi4.metric("Prediction Status", "Active")

        # Tabs
        tab_dash, tab_detail, tab_model = st.tabs(["📊 Executive Dashboard", "📋 Patient Analysis", "⚙️ Model Performance"])

        # --- TAB 1: EXECUTIVE DASHBOARD ---
        with tab_dash:
            row1_1, row1_2 = st.columns(2)
            
            with row1_1:
                st.subheader("Risk Distribution")
                fig_pie = px.pie(
                    df, 
                    names='Risk_Category', 
                    title='Patient Segmentation by Risk Level',
                    color='Risk_Category',
                    color_discrete_map={'No Risk':'#e5e7eb','Low':'#bfdbfe', 'Moderate':'#60a5fa', 'High':'#2563eb', 'Very High':'#1e3a8a'},
                    hole=0.4
                )
                fig_pie.update_layout(margin=dict(t=60))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with row1_2:
                st.subheader("Probability Density")
                fig_hist = px.histogram(
                    df, 
                    x='ALS_Risk_Probability', 
                    nbins=20,
                    title='Distribution of Risk Probabilities',
                    labels={'ALS_Risk_Probability': 'Risk Score (%)'},
                    color_discrete_sequence=['#0ea5e9']
                )
                fig_hist.update_layout(bargap=0.1, showlegend=False, template="plotly_white")
                fig_hist.update_layout(margin=dict(t=60))
                st.plotly_chart(fig_hist, use_container_width=True)

            if age_col:
                st.subheader("Age-Based Risk Profile")
                df_age = df.groupby(age_col)['ALS_Risk_Probability'].mean().reset_index()
                fig_line = px.line(
                    df_age, 
                    x=age_col, 
                    y='ALS_Risk_Probability',
                    title='Average Risk Trajectory by Age',
                    labels={age_col: 'Age', 'ALS_Risk_Probability': 'Avg Risk (%)'},
                    markers=True
                )
                fig_line.update_traces(line_color='#2563eb', line_width=3)
                fig_line.update_layout(template="plotly_white", margin=dict(t=60))
                st.plotly_chart(fig_line, use_container_width=True)

        # --- TAB 2: PATIENT ANALYSIS ---
        with tab_detail:
            st.subheader("Detailed Patient Stratification")
            
            # Filters
            c_fil1, c_fil2 = st.columns(2)
            with c_fil1:
                min_risk, max_risk = st.slider("Filter by Risk Probability", 0, 100, (0, 100))
            with c_fil2:
                search_query = st.text_input("Search by ID (if available)", placeholder="Enter patient ID...")

            # Filtering Logic
            mask = (df['ALS_Risk_Probability'] >= min_risk) & (df['ALS_Risk_Probability'] <= max_risk)
            df_filtered = df[mask]
            
            st.dataframe(
                df_filtered.style.background_gradient(
                    subset=['ALS_Risk_Probability'], 
                    cmap='Blues', 
                    vmin=0, 
                    vmax=100
                ).format({'ALS_Risk_Probability': '{:.2f}%'}),
                use_container_width=True,
                height=500
            )
            
            # Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Full Results (CSV)",
                csv,
                "als_risk_assessment_results.csv",
                "text/csv",
                key='download-csv'
            )

        # --- TAB 3: MODEL PERFORMANCE ---
        with tab_model:
            if has_labels:
                preds = (probas >= 0.5).astype(int)
                
                st.subheader("Diagnostic Performance (Current Batch)")
                m1, m2, m3 = st.columns(3)
                
                m1.plotly_chart(create_gauge_chart(accuracy_score(y_true, preds), "Accuracy"), use_container_width=True)
                m2.plotly_chart(create_gauge_chart(roc_auc_score(y_true, probas), "ROC AUC"), use_container_width=True)
                m3.plotly_chart(create_gauge_chart(recall_score(y_true, preds), "Recall (Sensitivity)"), use_container_width=True)
                
                col_conf, col_rep = st.columns([1, 2])
                
                with col_conf:
                    st.markdown("#### Confusion Matrix")
                    cm = confusion_matrix(y_true, preds)
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['No ALS', 'ALS'],
                        y=['No ALS', 'ALS'],
                        color_continuous_scale='Blues'
                    )
                    fig_cm.update_layout(margin=dict(t=60))
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                with col_rep:
                    st.markdown("#### Detailed Classification Report")
                    report_dict = classification_report(y_true, preds, output_dict=True)
                    st.dataframe(pd.DataFrame(report_dict).transpose().style.format("{:.3f}"), use_container_width=True)
            
            else:
                st.info("⚠️ Ground truth labels not found in uploaded data. Performance metrics cannot be calculated for this batch.")
                if static_metrics:
                    st.divider()
                    st.markdown("### 📚 Reference Training Metrics")
                    st.markdown(f"""
                    The model was originally trained with the following performance:
                    - **Accuracy:** {static_metrics.get('accuracy', 'N/A')}
                    - **ROC AUC:** {static_metrics.get('roc_auc', 'N/A')}
                    """)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
        st.exception(e)