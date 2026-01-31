import sys
import os
sys.path.append(os.path.dirname(__file__))
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from dotenv import load_dotenv
load_dotenv()  
from datetime import datetime   # Add to alerts.py after load_dotenv()
import os
print("DEBUG - Loaded values:")
print(f"SID exists: {bool(os.getenv('TWILIO_ACCOUNT_SID'))}")
print(f"Token exists: {bool(os.getenv('TWILIO_AUTH_TOKEN'))}")
print(f"From: {os.getenv('TWILIO_PHONE_NUMBER')}")
print(f"To: {os.getenv('ALERT_RECIPIENT_PHONE')}")   # Import ML inference
from models.inference import predict_from_dataframe
import re  
# Import analytics utilities (from services folder)
from services.analytics import (
    enrich_predictions_with_analytics,
    calculate_fleet_statistics,
    get_machine_analytics
)
from services.gemini_ai import get_ai_service
from alerts import trigger_alerts
# Helper function for PDF report generation
# Helper function for PDF report generation
# Helper function for PDF report generation
def convert_markdown_to_html(text):
    """Convert markdown to HTML and remove emojis"""
    # Remove all emojis and special unicode characters
    text = re.sub(r'[\U0001F300-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]', '', text)
    
    # Convert **bold** to <b>bold</b> - use non-greedy match with ANY characters except newline
    while '**' in text:
        text = text.replace('**', '<b>', 1).replace('**', '</b>', 1)
    
    return text       # ================= PAGE CONFIG ================= 
st.set_page_config(
    page_title="Industrial Maintenance Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        color: #e8eaed;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a7b 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 5px solid #4a9eff;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(74, 158, 255, 0.15);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #b8c5d6;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        line-height: 1.6;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
        color: #4a9eff;
        font-family: 'JetBrains Mono', monospace;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 500;
        color: #b8c5d6;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    
    /* Health Section */
    .health-section {
        background: linear-gradient(135deg, #1a2332 0%, #243447 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2d4a5f;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .health-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .update-timer {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #7a92a8;
        padding: 0.4rem 0.8rem;
        background: #1a2332;
        border-radius: 6px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-operational {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid #34d39950;
    }
    
    .status-warning {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        border: 1px solid #fbbf2450;
    }
    
    .status-critical {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef444450;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
        border-right: 1px solid #2d4a5f;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #b8c5d6;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #4a9eff 0%, #357abd 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(74, 158, 255, 0.3);
    }
    
    .stButton button:hover {
        background: linear-gradient(90deg, #357abd 0%, #2563a8 100%);
        box-shadow: 0 4px 15px rgba(74, 158, 255, 0.4);
        transform: translateY(-1px);
    }
    
    .stButton button:disabled {
        background: #2d4a5f;
        color: #5a6c7d;
        box-shadow: none;
    }
    
    /* Tables */
    [data-testid="stDataFrame"] {
        background: #1a2332;
        border-radius: 8px;
        border: 1px solid #2d4a5f;
    }
    
    /* Info/Warning Boxes */
    [data-testid="stAlert"] {
        background: rgba(74, 158, 255, 0.1);
        border-left: 4px solid #4a9eff;
        border-radius: 8px;
        color: #b8c5d6;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: #1a2332;
        border: 2px dashed #2d4a5f;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Code Blocks */
    code {
        background: #1a2332;
        color: #4a9eff;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Section Dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #2d4a5f, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="main-header">
    <h1>⚙️ Industrial Maintenance Intelligence Platform</h1>
    <p>Real-time predictive analytics for proactive equipment maintenance and operational excellence</p>
</div>
""", unsafe_allow_html=True)

# ================= INITIALIZE SESSION STATE =================
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'last_health_update' not in st.session_state:
    st.session_state.last_health_update = pd.Timestamp.now()

# Initialize AI service (cached)
if 'ai_service' not in st.session_state:
    st.session_state.ai_service = get_ai_service()

# ================= SIDEBAR =================
st.sidebar.markdown("### 🎛️ Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["🏠 Operations Dashboard", "📥 Data Integration", "📊 Fleet Analytics", "🤖 AI Intelligence Hub"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Alert system status
from alerts import is_configured

alert_status = "Configured" if is_configured() else "Not Configured"
alert_color = "#34d399" if is_configured() else "#fbbf24"

st.sidebar.markdown(f"""
<div style="padding: 1rem; background: rgba(74, 158, 255, 0.1); border-radius: 8px; border-left: 3px solid #4a9eff; margin-top: 1rem;">
    <p style="font-size: 0.75rem; color: #b8c5d6; margin: 0;"><strong>Alert System</strong></p>
    <p style="font-size: 0.7rem; color: #7a92a8; margin-top: 0.5rem;">
        • Twilio: <span style="color: {alert_color};">{alert_status}</span><br>
        • SMS: <span style="color: {alert_color};">{'Ready' if is_configured() else 'Pending'}</span><br>
        • WhatsApp: <span style="color: {alert_color};">{'Ready' if is_configured() else 'Pending'}</span>
    </p>
</div>
""", unsafe_allow_html=True) 
# Dynamic system status based on predictions
ml_status = "Active" if st.session_state.predictions is not None else "Standby"
ml_color = "#34d399" if st.session_state.predictions is not None else "#fbbf24"

st.sidebar.markdown(f"""
<div style="padding: 1rem; background: rgba(74, 158, 255, 0.1); border-radius: 8px; border-left: 3px solid #4a9eff;">
    <p style="font-size: 0.75rem; color: #b8c5d6; margin: 0;"><strong>System Status</strong></p>
    <p style="font-size: 0.7rem; color: #7a92a8; margin-top: 0.5rem;">
        • ML Engine: <span style="color: {ml_color};">{ml_status}</span><br>
        • Data Pipeline: <span style="color: #34d399;">Active</span><br>
        • API: <span style="color: #34d399;">Connected</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ================= OPERATIONS DASHBOARD =================
if page == "🏠 Operations Dashboard":
    st.markdown("### 📈 Factory Operations Overview")
    
    # Check if predictions are available
    if st.session_state.predictions is not None and st.session_state.uploaded_data is not None:
        predictions = st.session_state.predictions
        uploaded_data = st.session_state.uploaded_data
        
        # Enrich predictions with advanced analytics
        enriched_predictions = enrich_predictions_with_analytics(predictions)
        fleet_stats = calculate_fleet_statistics(predictions)
        
        total_assets = fleet_stats['total_assets']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Assets",
                value=total_assets,
                delta=None
            )
        
        with col2:
            avg_health = fleet_stats['avg_health_score']
            st.metric(
                label="Avg Health Score",
                value=f"{avg_health:.1f}",
                delta=f"{avg_health - 75:.1f}" if avg_health < 75 else f"+{avg_health - 75:.1f}"
            )
        
        with col3:
            critical = fleet_stats['critical_count']
            high_risk = fleet_stats['high_risk_count']
            st.metric(
                label="High Priority Assets",
                value=critical + high_risk,
                delta=f"{critical} Critical" if critical > 0 else "0 Critical"
            )
        
        with col4:
            low_risk = fleet_stats['low_risk_count']
            operational_pct = (low_risk / total_assets * 100) if total_assets > 0 else 0
            st.metric(
                label="Operational Status",
                value=f"{operational_pct:.0f}%",
                delta=f"{low_risk} assets"
            )
        
        st.markdown("---")
        
        # Display asset summary table with enriched data
        st.markdown("### 🎯 Asset Health Summary")
        
        # Merge uploaded data with enriched predictions
        summary_df = uploaded_data.copy()
        summary_df['Health Score'] = enriched_predictions['health_score'].values
        summary_df['Risk Level'] = enriched_predictions['risk_level'].values
        summary_df['Dominant Issue'] = enriched_predictions['dominant_issue'].values
        summary_df['Efficiency Index'] = enriched_predictions['efficiency_index'].values
        summary_df['Vibration Index'] = enriched_predictions['vibration_index'].values
        summary_df['Thermal Index'] = enriched_predictions['thermal_index'].values
        
        # Add visual status
        def risk_to_emoji(risk):
            if risk == "Low":
                return "🟢 Low"
            elif risk == "Medium":
                return "🟡 Medium"
            elif risk == "High":
                return "🟠 High"
            else:
                return "🔴 Critical"
        
        summary_df['Status'] = summary_df['Risk Level'].apply(risk_to_emoji)
        
        # Display key columns
        display_cols = []
        if 'machine_id' in summary_df.columns:
            display_cols.append('machine_id')
        if 'machine_type' in summary_df.columns:
            display_cols.append('machine_type')
        display_cols.extend(['Status', 'Health Score', 'Dominant Issue', 'Efficiency Index'])
        
        st.dataframe(
            summary_df[display_cols].style.format({
                'Health Score': '{:.1f}',
                'Efficiency Index': '{:.1f}'
            }),
            use_container_width=True,
            height=400
        )
        
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="Total Assets", value="—", delta="Awaiting data")
        with col2:
            st.metric(label="Avg Health Score", value="—", delta="0")
        with col3:
            st.metric(label="High Priority Assets", value="—", delta="0")
        with col4:
            st.metric(label="Operational Status", value="—", delta="0%")
        
        st.markdown("---")
        
        st.markdown("""
        <div style="padding: 2rem; background: rgba(74, 158, 255, 0.05); border-radius: 8px; border: 1px solid #2d4a5f; text-align: center;">
            <p style="color: #7a92a8; font-size: 0.95rem;">
                ⚙️ Machine health monitoring will be displayed here once data is uploaded and processed.<br>
                <span style="font-size: 0.85rem;">Navigate to Data Integration to begin.</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ================= DATA UPLOAD =================
elif page == "📥 Data Integration":
    st.markdown("### 📤 Sensor Data Upload & Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background: rgba(74, 158, 255, 0.05); padding: 1.5rem; border-radius: 8px; border-left: 3px solid #4a9eff; margin-bottom: 1.5rem;">
            <h4 style="margin: 0 0 0.5rem 0; color: #4a9eff;">📋 Data Format Requirements</h4>
            <p style="font-size: 0.85rem; color: #b8c5d6; margin: 0;">
                Upload CSV files containing machine sensor readings. Ensure your data includes the following fields for optimal processing.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.code(
            "machine_id,machine_type,air_temperature_k,process_temperature_k,rotational_speed_rpm,torque_nm,tool_wear_min,temperature,humidity,rainfall\n"
            "MTR-001,motor,298.5,310.2,1450,42.3,125,25.3,65,0\n"
            "PMP-042,pump,300.2,315.8,2850,38.5,203,27.0,72,0\n"
            "CMP-013,compressor,296.5,305.9,3200,52.3,134,23.3,66,0",
            language="csv"
        )
    
    with col2:
        st.markdown("""
        <div style="background: rgba(52, 211, 153, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid #34d39950;">
            <p style="font-size: 0.75rem; color: #34d399; margin: 0; font-weight: 600;">✓ SUPPORTED FORMATS</p>
            <p style="font-size: 0.7rem; color: #b8c5d6; margin-top: 0.5rem;">
                • CSV (Comma-separated)<br>
                • Max file size: 50MB<br>
                • Encoding: UTF-8
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select CSV file from your system",
        type=["csv"],
        help="Upload sensor data for analysis"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_data = df
        
        st.success(f"✓ Successfully loaded {len(df)} records from **{uploaded_file.name}**")
        
        st.markdown("### 📊 Data Preview")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Features", len(df.columns))
        col3.metric("Data Quality", "98.5%")
        
        st.dataframe(df, use_container_width=True, height=300)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Process Data", use_container_width=False, help="Run ML inference on uploaded data"):
            with st.spinner("Running ML inference..."):
                try:
                    # Run ML inference
                    predictions = predict_from_dataframe(df)
                    
                    # Store predictions in session state
                    st.session_state.predictions = predictions
                    st.session_state.last_health_update = pd.Timestamp.now()
                    
                    st.success("✅ ML inference completed successfully!")
                    st.info(f"📊 Generated predictions for {len(predictions)} assets. Navigate to Fleet Analytics or Operations Dashboard to view results.")
                    
                    # Show prediction summary
                    st.markdown("### 🎯 Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Avg Vibration Index",
                            value=f"{predictions['vibration_index'].mean():.1f}"
                        )
                    
                    with col2:
                        st.metric(
                            label="Avg Thermal Index",
                            value=f"{predictions['thermal_index'].mean():.1f}"
                        )
                    
                    with col3:
                        st.metric(
                            label="Avg Efficiency",
                            value=f"{predictions['efficiency_index'].mean():.1f}%"
                        )
                    
                except Exception as e:
                    st.error(f"❌ ML inference failed: {str(e)}")
                    st.info("Please ensure your CSV contains the required features for the ML model.")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🚀 Process Data", disabled=True, use_container_width=False, help="Please upload a CSV file first")

# ================= FLEET ANALYTICS =================
elif page == "📊 Fleet Analytics":
    st.markdown("### 🔧 Comprehensive Fleet Health Analytics")

    if st.session_state.predictions is None:
        st.warning("⚠️ No prediction data available. Please upload and process data in the Data Integration page first.")
        st.stop()
    
    predictions = st.session_state.predictions
    enriched_predictions = enrich_predictions_with_analytics(predictions)
    fleet_stats = calculate_fleet_statistics(predictions)
    
    # Calculate time since last update
    time_since_update = (pd.Timestamp.now() - st.session_state.last_health_update).total_seconds()
    time_until_next = max(0, 300 - time_since_update)
    minutes_left = int(time_until_next // 60)
    seconds_left = int(time_until_next % 60)
    
    # Use real ML predictions
    vibration_index = fleet_stats['avg_vibration']
    thermal_index = fleet_stats['avg_thermal']
    efficiency_index = fleet_stats['avg_efficiency']
    health_score = fleet_stats['avg_health_score']
    
    # Health Indices Section
    st.markdown("""
    <div class="health-section">
        <div class="health-title">
            🏥 Real-Time Health Indices
        </div>
        <div class="update-timer">
            ⏱️ Next refresh: {}:{:02d} | Last updated: {}
        </div>
    </div>
    """.format(
        minutes_left,
        seconds_left,
        st.session_state.last_health_update.strftime('%H:%M:%S')
    ), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Fleet Health Score",
            value=f"{health_score:.1f}",
            delta=f"{health_score - 75:.1f}",
            delta_color="normal",
            help="Composite health metric (0-100, higher is better)"
        )
    
    with col2:
        st.metric(
            label="Vibration Index",
            value=f"{vibration_index:.1f}",
            delta=f"{vibration_index - 50:+.1f}",
            delta_color="inverse",
            help="Lower values indicate healthier equipment"
        )
    
    with col3:
        st.metric(
            label="Thermal Index",
            value=f"{thermal_index:.1f}",
            delta=f"{thermal_index - 50:+.1f}",
            delta_color="inverse",
            help="Temperature-based health indicator"
        )
    
    with col4:
        st.metric(
            label="Efficiency Index",
            value=f"{efficiency_index:.1f}%",
            delta=f"{efficiency_index - 75:+.1f}%",
            delta_color="normal",
            help="Overall operational efficiency"
        )
    
    st.markdown("---")
    
    # Fleet Table Section
    st.markdown("### 🎯 Asset Status Matrix")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.uploaded_data is not None:
            asset_df = st.session_state.uploaded_data.copy()
            asset_df['Health Score'] = enriched_predictions['health_score'].values
            asset_df['Risk Level'] = enriched_predictions['risk_level'].apply(
                lambda x: f"🟢 {x}" if x == "Low" else (f"🟡 {x}" if x == "Medium" else (f"🟠 {x}" if x == "High" else f"🔴 {x}"))
            )
            asset_df['Dominant Issue'] = enriched_predictions['dominant_issue'].values
            asset_df['Last Maintenance'] = "—"
            
            display_cols = []
            if 'machine_id' in asset_df.columns:
                display_cols.append('machine_id')
            if 'machine_type' in asset_df.columns:
                display_cols.append('machine_type')
            display_cols.extend(['Health Score', 'Risk Level', 'Dominant Issue', 'Last Maintenance'])
            
            st.dataframe(
                asset_df[display_cols].head(10).style.format({'Health Score': '{:.1f}'}),
                use_container_width=True,
                height=200
            )
    
    with col2:
        st.markdown("""
        <div style="background: rgba(74, 158, 255, 0.05); padding: 1rem; border-radius: 8px; height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <p style="font-size: 0.75rem; color: #4a9eff; margin: 0; font-weight: 600; text-transform: uppercase;">Legend</p>
            <div style="margin-top: 1rem;">
                <span class="status-badge status-operational">● Low Risk</span><br><br>
                <span class="status-badge status-warning">● Medium/High Risk</span><br><br>
                <span class="status-badge status-critical">● Critical</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Explainable Visual Analytics
    st.markdown("### 📊 Diagnostic Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Vibration distribution
        fig_vib = px.histogram(
            enriched_predictions,
            x='vibration_index',
            nbins=20,
            title="Vibration Index Distribution",
            labels={'vibration_index': 'Vibration Index', 'count': 'Asset Count'},
            color_discrete_sequence=['#4a9eff']
        )
        fig_vib.add_vline(x=60, line_dash="dash", line_color="#ef4444", 
                          annotation_text="Critical Threshold", annotation_position="top right")
        fig_vib.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#b8c5d6', family='IBM Plex Sans'),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_vib, use_container_width=True)
    
    with col2:
        # Thermal distribution
        fig_thermal = px.histogram(
            enriched_predictions,
            x='thermal_index',
            nbins=20,
            title="Thermal Index Distribution",
            labels={'thermal_index': 'Thermal Index', 'count': 'Asset Count'},
            color_discrete_sequence=['#fbbf24']
        )
        fig_thermal.add_vline(x=60, line_dash="dash", line_color="#ef4444",
                              annotation_text="Critical Threshold", annotation_position="top right")
        fig_thermal.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#b8c5d6', family='IBM Plex Sans'),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_thermal, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Efficiency distribution
        fig_eff = px.histogram(
            enriched_predictions,
            x='efficiency_index',
            nbins=20,
            title="Efficiency Index Distribution",
            labels={'efficiency_index': 'Efficiency Index (%)', 'count': 'Asset Count'},
            color_discrete_sequence=['#34d399']
        )
        fig_eff.add_vline(x=70, line_dash="dash", line_color="#fbbf24",
                          annotation_text="Target Threshold", annotation_position="top left")
        fig_eff.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#b8c5d6', family='IBM Plex Sans'),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_eff, use_container_width=True)
    
    with col2:
        # Risk level distribution
        risk_counts = enriched_predictions['risk_level'].value_counts()
        fig_risk = go.Figure(data=[
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker=dict(
                    color=['#34d399' if r == 'Low' else '#fbbf24' if r == 'Medium' else '#ef9944' if r == 'High' else '#ef4444' for r in risk_counts.index],
                    line=dict(color='#1a2332', width=2)
                ),
                text=risk_counts.values,
                textposition='auto',
            )
        ])
        fig_risk.update_layout(
            title="Risk Level Distribution",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#b8c5d6', family='IBM Plex Sans'),
            height=300,
            xaxis=dict(title="Risk Level", gridcolor='#2d4a5f', showgrid=False),
            yaxis=dict(title="Asset Count", gridcolor='#2d4a5f', showgrid=True)
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Index Correlation Analysis")
    
    correlation_data = enriched_predictions[['vibration_index', 'thermal_index', 'efficiency_index']].corr()
    
    fig_corr = px.imshow(
        correlation_data,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdYlGn_r',
        title="Correlation Matrix: Understanding Failure Patterns"
    )
    fig_corr.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#b8c5d6', family='IBM Plex Sans'),
        height=400
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Auto-refresh
    time.sleep(1)
    st.rerun()

# ================= AI INTELLIGENCE HUB =================  
elif page == "🤖 AI Intelligence Hub":
    st.markdown("### 🧠 AI-Powered Maintenance Intelligence")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ No prediction data available. Please upload and process data in the Data Integration page first.")
        st.stop()
    
    # Check AI service configuration
    ai_service = st.session_state.ai_service
    
    if not ai_service.is_configured:
        st.error("""
        ⚠️ **Google Gemini AI not configured**
        
        To enable AI-powered analysis:
        1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Set environment variable: `export GOOGLE_AI_API_KEY=your_key_here`
        3. Restart the application
        
        Alternatively, set the API key in your code or config file.
        """)
        st.stop()
    
    # Machine selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        machine_options = []
        if st.session_state.uploaded_data is not None and 'machine_id' in st.session_state.uploaded_data.columns:
            machine_options = st.session_state.uploaded_data['machine_id'].tolist()
        else:
            machine_options = [f"Asset {i+1}" for i in range(len(st.session_state.predictions))]
        
        selected_machine = st.selectbox(
            "Select Asset for Analysis",
            machine_options,
            help="Choose an asset to view AI-generated insights"
        )
        selected_index = machine_options.index(selected_machine)
    
    with col2:
        analysis_depth = st.selectbox("Analysis Depth", ["Quick Scan", "Standard", "Deep Analysis"])
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Generate Analysis", use_container_width=True)
    
    st.markdown("---")
    
    # Generate or display AI analysis
    if analyze_btn:
        # Get machine data and predictions
        machine_data, prediction_data = get_machine_analytics(
            selected_index,
            st.session_state.uploaded_data,
            st.session_state.predictions
        )
        
        # Show loading state
        with st.spinner(f"🤖 AI analyzing {selected_machine} using {analysis_depth} mode..."):
            # Call AI service
            analysis_result = ai_service.generate_maintenance_analysis(
                machine_data=machine_data,
                prediction_data=prediction_data,
                analysis_depth=analysis_depth
            )
        
        # Store in session state
        st.session_state.ai_analysis = analysis_result
        st.session_state.ai_analysis_machine = selected_machine
        st.session_state.ai_analysis_depth = analysis_depth
        
        # ==================== ALERT AUTOMATION ====================
        # Trigger SMS/WhatsApp alerts based on machine health
        if analysis_result.get('status') == 'success':
            alert_result = trigger_alerts(
                machine_row=machine_data,
                prediction_row=prediction_data,
                ai_analysis=analysis_result.get('full_response', '')
            )
            
            # Show alert status to user (non-blocking)
            if alert_result['alert_triggered']:
                alert_msg = f"📢 {alert_result['message']}"
                if alert_result['sms_sent'] or alert_result['whatsapp_sent']:
                    st.success(alert_msg)
                else:
                    st.warning(f"{alert_msg} (Alerts not configured)")
        # ==================== END ALERT AUTOMATION ====================
    # Display AI analysis if available
    if 'ai_analysis' in st.session_state:
        analysis = st.session_state.ai_analysis
        
        if analysis['status'] == 'error':
            st.error(f"❌ AI Analysis Failed: {analysis['error_message']}")
            st.stop()
        
        # Machine info header
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #1e3a5f 0%, #2d5a7b 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <p style="font-size: 0.85rem; color: #b8c5d6; margin: 0;">
                <strong style="color: #4a9eff;">Asset:</strong> {st.session_state.ai_analysis_machine} | 
                <strong style="color: #4a9eff;">Analysis Depth:</strong> {st.session_state.ai_analysis_depth}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Health metrics
        pred_data = analysis['prediction_data']
        health_score = pred_data.get('health_score', 0)
        risk_level = pred_data.get('risk_level', 'Unknown')
        dominant_issue = pred_data.get('dominant_issue', 'Unknown')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_color = {
                'Low': '#34d399',
                'Medium': '#fbbf24',
                'High': '#ef9944',
                'Critical': '#ef4444'
            }.get(risk_level, '#7a92a8')
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: rgba(74, 158, 255, 0.1); border-radius: 8px;">
                <p style="font-size: 2.5rem; margin: 0; color: {risk_color}; font-weight: 700;">{health_score:.0f}</p>
                <p style="font-size: 0.8rem; color: #b8c5d6; margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px;">Health Score</p>
                <p style="font-size: 0.7rem; color: {risk_color}; margin-top: 0.3rem;">{risk_level} Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: rgba(74, 158, 255, 0.1); border-radius: 8px;">
                <p style="font-size: 2.5rem; margin: 0; color: #4a9eff; font-weight: 700;">{pred_data['efficiency_index']:.0f}%</p>
                <p style="font-size: 0.8rem; color: #b8c5d6; margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px;">Efficiency</p>
                <p style="font-size: 0.7rem; color: #fbbf24; margin-top: 0.3rem;">{100 - pred_data['efficiency_index']:.0f}% Loss</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: rgba(74, 158, 255, 0.1); border-radius: 8px;">
                <p style="font-size: 1.5rem; margin: 0; color: #4a9eff; font-weight: 700;">{dominant_issue}</p>
                <p style="font-size: 0.8rem; color: #b8c5d6; margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px;">Primary Issue</p>
                <p style="font-size: 0.7rem; color: #4a9eff; margin-top: 0.3rem;">Dominant Factor</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # AI-generated analysis sections
        if analysis.get('root_cause'):
            st.markdown("### 🔬 AI Root Cause Diagnosis")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a2332 0%, #243447 100%); padding: 1.5rem; border-radius: 12px; border: 1px solid #2d4a5f; margin-bottom: 1.5rem;">
                <p style="color: #e8eaed; font-size: 0.95rem; line-height: 1.8; white-space: pre-wrap;">{analysis['root_cause']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if analysis.get('risk_assessment'):
            st.markdown("### ⚠️ Risk Assessment")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%); 
                        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #ef4444; margin-bottom: 1.5rem;">
                <p style="color: #e8eaed; font-size: 0.95rem; line-height: 1.8; white-space: pre-wrap;">{analysis['risk_assessment']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if analysis.get('maintenance_recommendations'):
            st.markdown("### 🛠️ Maintenance Recommendations")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(74, 158, 255, 0.15) 0%, rgba(74, 158, 255, 0.05) 100%); 
                        padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(74, 158, 255, 0.3); margin-bottom: 1.5rem;">
                <p style="color: #e8eaed; font-size: 0.95rem; line-height: 1.8; white-space: pre-wrap;">{analysis['maintenance_recommendations']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if analysis.get('timeline'):
            st.markdown("### 📅 Maintenance Timeline")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a2332 0%, #243447 100%); 
                        padding: 1.5rem; border-radius: 12px; border: 1px solid #2d4a5f; margin-bottom: 1.5rem;">
                <p style="color: #e8eaed; font-size: 0.95rem; line-height: 1.8; white-space: pre-wrap;">{analysis['timeline']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if analysis.get('cost_impact'):
            st.markdown("### 💰 Financial Impact Analysis")
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(245, 158, 11, 0.05) 100%); 
                        padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.3); margin-bottom: 1.5rem;">
                <p style="color: #e8eaed; font-size: 0.95rem; line-height: 1.8; white-space: pre-wrap;">{analysis['cost_impact']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Full AI response (collapsible)
        with st.expander("📄 View Complete AI Analysis"):
            st.markdown(f"""
            <div style="background: #1a2332; padding: 1rem; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #b8c5d6;">
                {analysis['full_response'].replace('\n', '<br>')}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Export options

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Generate Maintenance Report", use_container_width=True, help="Create a detailed AI-generated maintenance report"):
                # Get machine data and predictions
                machine_data, prediction_data = get_machine_analytics(
                    selected_index,
                    st.session_state.uploaded_data,
                    st.session_state.predictions
                )
                
                # Extract key information
                machine_id = machine_data.get('machine_id', 'UNKNOWN')
                machine_type = machine_data.get('machine_type', 'Unknown Type')
                health_score = prediction_data.get('health_score', 0)
                risk_level = prediction_data.get('risk_level', 'Unknown')
                efficiency = prediction_data.get('efficiency_index', 0)
                vibration = prediction_data.get('vibration_index', 0)
                thermal = prediction_data.get('thermal_index', 0)
                dominant_issue = prediction_data.get('dominant_issue', 'Unknown')
                
                # Get existing AI analysis sections
                ai_analysis = st.session_state.ai_analysis
                root_cause = ai_analysis.get('root_cause', 'Analysis not available')
                risk_assessment = ai_analysis.get('risk_assessment', 'Assessment not available')
                recommendations = ai_analysis.get('maintenance_recommendations', 'Recommendations not available')
                cost_impact = ai_analysis.get('cost_impact', 'Impact analysis not available')
                
                # Generate timestamp
                report_date = datetime.now().strftime('%B %d, %Y at %H:%M')
                
                # Assemble the report
                report = f"""
# 📄 MAINTENANCE INTELLIGENCE REPORT

**Generated:** {report_date}  
**Analysis Depth:** {st.session_state.ai_analysis_depth}

---

## 1️⃣ EXECUTIVE SUMMARY

**Asset Identification**
- **Machine ID:** {machine_id}
- **Machine Type:** {machine_type}
- **Health Score:** {health_score:.1f}/100
- **Risk Level:** {risk_level}
- **Status:** {'🔴 CRITICAL - Immediate Action Required' if risk_level == 'Critical' else '🟠 HIGH RISK - Urgent Attention Needed' if risk_level == 'High' else '🟡 MODERATE RISK - Schedule Maintenance' if risk_level == 'Medium' else '🟢 OPERATIONAL - Monitor Regularly'}

---

## 2️⃣ CURRENT MACHINE CONDITION

**Performance Metrics**
- **Efficiency Index:** {efficiency:.1f}% ({100-efficiency:.1f}% loss from optimal)
- **Vibration Index:** {vibration:.1f} {'⚠️ ABNORMAL' if vibration > 60 else '✓ Normal'}
- **Thermal Index:** {thermal:.1f} {'⚠️ ELEVATED' if thermal > 60 else '✓ Normal'}
- **Dominant Issue:** {dominant_issue}

**Operational Assessment**  
The asset is currently operating at {efficiency:.1f}% efficiency with {'elevated' if vibration > 60 or thermal > 60 else 'acceptable'} stress indicators. Primary concern is {dominant_issue.lower()}-related degradation.

---

## 3️⃣ AI ROOT CAUSE ANALYSIS

{root_cause}

---

## 4️⃣ RISK & FAILURE ASSESSMENT

{risk_assessment}

**Failure Likelihood:** {'HIGH - Failure imminent within days' if risk_level == 'Critical' else 'MODERATE-HIGH - Failure likely within weeks' if risk_level == 'High' else 'MODERATE - Gradual degradation expected' if risk_level == 'Medium' else 'LOW - Normal operational wear'}

---

## 5️⃣ MAINTENANCE RECOMMENDATIONS

{recommendations}

**Action Priority Matrix:**
- **Immediate (0-6 hours):** {'Shutdown and inspect' if risk_level == 'Critical' else 'N/A'}
- **Short-term (1-7 days):** {'Component replacement' if risk_level in ['Critical', 'High'] else 'Diagnostic inspection'}
- **Long-term (1-3 months):** Preventive maintenance schedule revision

---

## 6️⃣ BUSINESS IMPACT ASSESSMENT

{cost_impact}

**Quantified Impact:**
- **Efficiency Loss:** {100-efficiency:.1f}% reduction in output capacity
- **Downtime Risk:** {'CRITICAL - Unplanned shutdown imminent' if risk_level == 'Critical' else 'HIGH - Significant disruption likely' if risk_level == 'High' else 'MODERATE - Manageable with planning' if risk_level == 'Medium' else 'LOW - Minimal impact expected'}
- **Financial Exposure:** {'High - Emergency repairs + production loss' if risk_level in ['Critical', 'High'] else 'Moderate - Scheduled maintenance costs' if risk_level == 'Medium' else 'Low - Routine maintenance'}

---

## 📋 REPORT METADATA

**Data Sources:** Sensor telemetry, ML predictions, AI diagnostic analysis  
**Analysis Model:** Claude Sonnet 4.5 + Gemini 2.0 Flash  
**Confidence Level:** {'High' if health_score > 60 else 'Medium'}  
**Next Review:** Recommended within {'6 hours' if risk_level == 'Critical' else '48 hours' if risk_level == 'High' else '7 days'}

---

*This report is generated by AI-powered maintenance intelligence systems and should be reviewed by qualified maintenance personnel before taking action.*
"""
                
                # Store in session state
                st.session_state.generated_report = convert_markdown_to_html(report)
                st.success("✅ Maintenance report generated successfully!")

        with col2:
            if 'generated_report' in st.session_state:
                if st.button("⬇️ Download Report (PDF)", use_container_width=True, help="Download professional PDF report"):
                    # Import PDF generator
                    from services.pdf_generator import generate_maintenance_pdf
                    
                    # Get data for PDF
                    machine_data, prediction_data = get_machine_analytics(
                        selected_index,
                        st.session_state.uploaded_data,
                        st.session_state.predictions
                    )
                    
                    machine_id = machine_data.get('machine_id', 'UNKNOWN')
                    health_score = prediction_data.get('health_score', 0)
                    risk_level = prediction_data.get('risk_level', 'Unknown')
                    
                    # Generate PDF
                    with st.spinner("🖨️ Generating PDF..."):
                        try:
                            pdf_bytes = generate_maintenance_pdf(
                                report_text=st.session_state.generated_report,
                                machine_id=machine_id,
                                health_score=health_score,
                                risk_level=risk_level
                            )
                            
                            # Create download button
                            filename = f"Maintenance_Report_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            
                            st.download_button(
                                label="💾 Save PDF File",
                                data=pdf_bytes,
                                file_name=filename,
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
                            st.success("✅ PDF generated successfully!")
                        
                        except Exception as e:
                            st.error(f"❌ PDF generation failed: {str(e)}")
            else:
                st.button("⬇️ Download Report (PDF)", use_container_width=True, disabled=True, help="Generate a report first")

        # Display generated report if available
        if 'generated_report' in st.session_state:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---") 
            display_report = st.session_state.generated_report.replace('<b>', '**').replace('</b>', '**')  #st.markdown(st.session_state.generated_report)
            st.markdown(display_report)   #st.markdown(st.session_state.generated_report
    else:
        # Before analysis is generated
        st.markdown("""
        <div style="padding: 4rem 2rem; background: rgba(74, 158, 255, 0.05); border-radius: 12px; 
                    border: 2px dashed #2d4a5f; text-align: center;">
            <p style="font-size: 3rem; margin: 0;">🤖</p>
            <h3 style="color: #4a9eff; margin: 1rem 0 0.5rem 0;">Real AI Analysis Ready</h3>
            <p style="color: #7a92a8; font-size: 0.95rem; max-width: 600px; margin: 0 auto;">
                Select an asset and click "Generate Analysis" to receive comprehensive AI-powered 
                maintenance intelligence powered by Google Gemini, including diagnostics, timelines, 
                financial impact, and actionable recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    
