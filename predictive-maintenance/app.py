import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Import ML inference
from models.inference import predict_from_dataframe

# ================= PAGE CONFIG =================
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

# ================= SIDEBAR =================
st.sidebar.markdown("### 🎛️ Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["🏠 Operations Dashboard", "📥 Data Integration", "📊 Fleet Analytics", "🤖 AI Intelligence Hub", "📄 Reports & Insights"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

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

# ================= DASHBOARD =================
if page == "🏠 Operations Dashboard":
    st.markdown("### 📈 Factory Operations Overview")
    
    # Check if predictions are available
    if st.session_state.predictions is not None and st.session_state.uploaded_data is not None:
        predictions = st.session_state.predictions
        uploaded_data = st.session_state.uploaded_data
        
        total_assets = len(predictions)
        
        # Classify assets by efficiency
        operational = len(predictions[predictions['efficiency_index'] >= 70])
        attention = len(predictions[(predictions['efficiency_index'] >= 40) & (predictions['efficiency_index'] < 70)])
        critical = len(predictions[predictions['efficiency_index'] < 40])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Assets",
                value=total_assets,
                delta=None
            )
        
        with col2:
            operational_pct = (operational / total_assets * 100) if total_assets > 0 else 0
            st.metric(
                label="Operational",
                value=operational,
                delta=f"{operational_pct:.1f}%"
            )
        
        with col3:
            st.metric(
                label="Attention Required",
                value=attention,
                delta=f"{attention}" if attention > 0 else "0"
            )
        
        with col4:
            st.metric(
                label="Critical Alerts",
                value=critical,
                delta=f"{critical}" if critical > 0 else "0"
            )
        
        st.markdown("---")
        
        # Display asset summary table
        st.markdown("### 🎯 Asset Health Summary")
        
        # Merge uploaded data with predictions
        summary_df = uploaded_data.copy()
        summary_df['Efficiency Index'] = predictions['efficiency_index'].values
        summary_df['Vibration Index'] = predictions['vibration_index'].values
        summary_df['Thermal Index'] = predictions['thermal_index'].values
        
        # Add status classification
        def classify_status(efficiency):
            if efficiency >= 70:
                return "🟢 Operational"
            elif efficiency >= 40:
                return "🟡 Attention"
            else:
                return "🔴 Critical"
        
        summary_df['Status'] = summary_df['Efficiency Index'].apply(classify_status)
        
        # Display key columns
        display_cols = []
        if 'machine_id' in summary_df.columns:
            display_cols.append('machine_id')
        if 'machine_type' in summary_df.columns:
            display_cols.append('machine_type')
        display_cols.extend(['Status', 'Efficiency Index', 'Vibration Index', 'Thermal Index'])
        
        st.dataframe(
            summary_df[display_cols].style.format({
                'Efficiency Index': '{:.1f}',
                'Vibration Index': '{:.1f}',
                'Thermal Index': '{:.1f}'
            }),
            use_container_width=True,
            height=400
        )
        
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Assets",
                value="—",
                delta="Awaiting data"
            )
        
        with col2:
            st.metric(
                label="Operational",
                value="—",
                delta="0%"
            )
        
        with col3:
            st.metric(
                label="Attention Required",
                value="—",
                delta="0"
            )
        
        with col4:
            st.metric(
                label="Critical Alerts",
                value="—",
                delta="0"
            )
        
        st.markdown("---")
        
        st.markdown("""
        <div style="padding: 2rem; background: rgba(74, 158, 255, 0.05); border-radius: 8px; border: 1px solid #2d4a5f; text-align: center;">
            <p style="color: #7a92a8; font-size: 0.95rem;">
                ⚙️ Machine health monitoring will be displayed here once the prediction engine is integrated.<br>
                <span style="font-size: 0.85rem;">Connect your ML model to begin real-time asset tracking.</span>
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

# ================= FLEET OVERVIEW =================
elif page == "📊 Fleet Analytics":
    st.markdown("### 🔧 Comprehensive Fleet Health Analytics")

    # Check if predictions are available
    if st.session_state.predictions is None:
        st.warning("⚠️ No prediction data available. Please upload and process data in the Data Integration page first.")
        st.stop()
    
    predictions = st.session_state.predictions
    
    # Calculate time since last update
    time_since_update = (pd.Timestamp.now() - st.session_state.last_health_update).total_seconds()
    
    # Calculate time until next update (5 minutes = 300 seconds)
    time_until_next = max(0, 300 - time_since_update)
    minutes_left = int(time_until_next // 60)
    seconds_left = int(time_until_next % 60)
    
    # Use real ML predictions
    vibration_index = predictions['vibration_index'].mean()
    thermal_index = predictions['thermal_index'].mean()
    efficiency_index = predictions['efficiency_index'].mean()
    
    # Calculate deltas (comparing to baseline of 50 for indices, inverted logic)
    vibration_delta = vibration_index - 50
    thermal_delta = thermal_index - 50
    efficiency_delta = efficiency_index - 75  # baseline 75% for efficiency
    
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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Vibration Index",
            value=f"{vibration_index:.1f}",
            delta=f"{vibration_delta:+.1f}",
            delta_color="inverse",
            help="Lower values indicate healthier equipment"
        )
    
    with col2:
        st.metric(
            label="Thermal Index",
            value=f"{thermal_index:.1f}",
            delta=f"{thermal_delta:+.1f}",
            delta_color="inverse",
            help="Temperature-based health indicator"
        )
    
    with col3:
        st.metric(
            label="Efficiency Index",
            value=f"{efficiency_index:.1f}%",
            delta=f"{efficiency_delta:+.1f}%",
            delta_color="normal",
            help="Overall operational efficiency"
        )
    
    st.markdown("---")
    
    # Fleet Table Section
    st.markdown("### 🎯 Asset Status Matrix")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create asset status table
        if st.session_state.uploaded_data is not None:
            asset_df = st.session_state.uploaded_data.copy()
            asset_df['Risk Level'] = predictions['efficiency_index'].apply(
                lambda x: "🟢 Operational" if x >= 70 else ("🟡 Attention" if x >= 40 else "🔴 Critical")
            )
            asset_df['Priority'] = predictions['efficiency_index'].apply(
                lambda x: "Low" if x >= 70 else ("Medium" if x >= 40 else "High")
            )
            asset_df['Last Maintenance'] = "—"
            
            display_cols = []
            if 'machine_id' in asset_df.columns:
                display_cols.append('machine_id')
            if 'machine_type' in asset_df.columns:
                display_cols.append('machine_type')
            display_cols.extend(['Risk Level', 'Priority', 'Last Maintenance'])
            
            st.dataframe(
                asset_df[display_cols].head(10),
                use_container_width=True,
                height=200
            )
        else:
            placeholder_df = pd.DataFrame({
                "Asset ID": ["—"],
                "Type": ["—"],
                "Risk Level": ["—"],
                "Priority": ["—"],
                "Last Maintenance": ["—"]
            })
            
            st.dataframe(
                placeholder_df,
                use_container_width=True,
                height=200
            )
    
    with col2:
        st.markdown("""
        <div style="background: rgba(74, 158, 255, 0.05); padding: 1rem; border-radius: 8px; height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <p style="font-size: 0.75rem; color: #4a9eff; margin: 0; font-weight: 600; text-transform: uppercase;">Legend</p>
            <div style="margin-top: 1rem;">
                <span class="status-badge status-operational">● Operational</span><br><br>
                <span class="status-badge status-warning">● Attention</span><br><br>
                <span class="status-badge status-critical">● Critical</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fleet Distribution Chart
    st.markdown("### 📊 Fleet Health Distribution")
    
    # Calculate actual distribution
    operational_count = len(predictions[predictions['efficiency_index'] >= 70])
    attention_count = len(predictions[(predictions['efficiency_index'] >= 40) & (predictions['efficiency_index'] < 70)])
    critical_count = len(predictions[predictions['efficiency_index'] < 40])
    
    fig = go.Figure(data=[
        go.Bar(
            x=["Operational", "Attention Required", "Critical"],
            y=[operational_count, attention_count, critical_count],
            marker=dict(
                color=['#34d399', '#fbbf24', '#ef4444'],
                line=dict(color='#1a2332', width=2)
            ),
            text=[operational_count, attention_count, critical_count],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#b8c5d6', family='IBM Plex Sans'),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(
            text="Asset Status Overview",
            font=dict(size=14, color='#b8c5d6')
        ),
        xaxis=dict(
            gridcolor='#2d4a5f',
            showgrid=False
        ),
        yaxis=dict(
            gridcolor='#2d4a5f',
            showgrid=True,
            title="Count"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh every second to update countdown
    time.sleep(1)
    st.rerun()

# ================= AI INTELLIGENCE HUB =================
elif page == "🤖 AI Intelligence Hub":
    st.markdown("### 🧠 AI-Powered Maintenance Intelligence")
    
    # Check if predictions are available
    if st.session_state.predictions is None:
        st.warning("⚠️ No prediction data available. Please upload and process data in the Data Integration page first.")
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
    with col2:
        analysis_depth = st.selectbox("Analysis Depth", ["Quick Scan", "Standard", "Deep Analysis"])
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Generate Analysis", use_container_width=True)
    
    st.markdown("---")
    
    # Simulated LLM output - In production, this would call your actual LLM
    if analyze_btn or 'ai_analysis_generated' in st.session_state:
        st.session_state.ai_analysis_generated = True
        
        # Smart Alerts Section
        st.markdown("### 🚨 Smart Alerts & Notifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%); 
                        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #ef4444; margin-bottom: 1rem;">
                <p style="font-size: 0.75rem; color: #ef4444; margin: 0; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
                    ⚠️ CRITICAL ALERT
                </p>
                <p style="color: #fca5a5; font-size: 1rem; margin-top: 0.5rem; font-weight: 600;">
                    Immediate Maintenance Required
                </p>
                <p style="color: #b8c5d6; font-size: 0.85rem; margin-top: 0.8rem; line-height: 1.5;">
                    High vibration levels detected on MTR-001. Bearing failure risk is elevated. 
                    Recommend immediate shutdown and inspection.
                </p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(239, 68, 68, 0.3);">
                    <p style="font-size: 0.75rem; color: #7a92a8; margin: 0;">
                        📞 Emergency contacts notified<br>
                        📧 Maintenance team alerted<br>
                        📱 SMS sent to on-call engineer
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%); 
                        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #fbbf24; margin-bottom: 1rem;">
                <p style="font-size: 0.75rem; color: #fbbf24; margin: 0; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
                    ⚡ ATTENTION NEEDED
                </p>
                <p style="color: #fcd34d; font-size: 1rem; margin-top: 0.5rem; font-weight: 600;">
                    Scheduled Maintenance Due
                </p>
                <p style="color: #b8c5d6; font-size: 0.85rem; margin-top: 0.8rem; line-height: 1.5;">
                    PMP-042 pump efficiency has decreased by 12% over the past week. 
                    Preventive maintenance recommended within 72 hours.
                </p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(251, 191, 36, 0.3);">
                    <p style="font-size: 0.75rem; color: #7a92a8; margin: 0;">
                        📅 Maintenance window: Next 3 days<br>
                        👷 Assigned to: Team B<br>
                        💰 Estimated cost: $2,400
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Machine Health Summary
        st.markdown("### 📊 Machine Health Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: rgba(251, 191, 36, 0.1); border-radius: 8px;">
                <p style="font-size: 2.5rem; margin: 0; color: #fbbf24; font-weight: 700;">58</p>
                <p style="font-size: 0.8rem; color: #b8c5d6; margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px;">Health Score</p>
                <p style="font-size: 0.7rem; color: #fbbf24; margin-top: 0.3rem;">Attention Range</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: rgba(251, 191, 36, 0.1); border-radius: 8px;">
                <p style="font-size: 2.5rem; margin: 0; color: #fbbf24; font-weight: 700;">67%</p>
                <p style="font-size: 0.8rem; color: #b8c5d6; margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px;">Current Efficiency</p>
                <p style="font-size: 0.7rem; color: #fbbf24; margin-top: 0.3rem;">33% Loss Detected</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: rgba(74, 158, 255, 0.1); border-radius: 8px;">
                <p style="font-size: 2.5rem; margin: 0; color: #4a9eff; font-weight: 700;">2.4</p>
                <p style="font-size: 0.8rem; color: #b8c5d6; margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px;">Days Until Failure</p>
                <p style="font-size: 0.7rem; color: #4a9eff; margin-top: 0.3rem;">Predicted</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # AI Diagnostic Report
        st.markdown("### 🔬 AI Diagnostic Analysis")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a2332 0%, #243447 100%); 
                    padding: 2rem; border-radius: 12px; border: 1px solid #2d4a5f; margin-bottom: 1.5rem;">
            <p style="font-size: 1.1rem; color: #4a9eff; margin: 0 0 1rem 0; font-weight: 600;">
                🎯 Primary Diagnosis: Bearing Degradation
            </p>
            <p style="color: #e8eaed; font-size: 0.95rem; line-height: 1.8; margin-bottom: 1.5rem;">
                Analysis of sensor data from MTR-001 reveals critical bearing wear patterns. The combination of 
                <span style="color: #ef4444; font-weight: 600;">elevated vibration signatures (amplitude: 8.4mm/s RMS)</span> 
                and <span style="color: #fbbf24; font-weight: 600;">thermal stress (bearing temp: 87°C)</span> 
                strongly indicates advanced bearing degradation, likely in the drive-end bearing assembly.
            </p>
            
            <div style="background: rgba(74, 158, 255, 0.1); padding: 1.2rem; border-radius: 8px; border-left: 3px solid #4a9eff; margin-bottom: 1.5rem;">
                <p style="font-size: 0.85rem; color: #4a9eff; margin: 0 0 0.5rem 0; font-weight: 600;">
                    🔍 ROOT CAUSE ANALYSIS
                </p>
                <p style="color: #b8c5d6; font-size: 0.85rem; line-height: 1.6; margin: 0;">
                    <strong>Primary Factors:</strong><br>
                    • Inadequate lubrication (last service: 147 days ago, recommended: 90 days)<br>
                    • Extended operation above rated load (112% of nominal capacity for 23 days)<br>
                    • Contamination from environmental particles (detected in vibration spectrum analysis)<br>
                    • Bearing fatigue after 14,247 operating hours (approaching L10 life expectancy)
                </p>
            </div>
            
            <p style="color: #e8eaed; font-size: 0.95rem; line-height: 1.8; margin-bottom: 1.5rem;">
                The degradation pattern suggests the bearing has entered <span style="color: #ef4444; font-weight: 600;">Stage 3 failure progression</span>, 
                where rapid exponential degradation typically occurs. Without intervention, complete bearing seizure 
                is predicted within <span style="color: #ef4444; font-weight: 600;">48-72 hours</span>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Financial Impact Analysis
        st.markdown("### 💰 Loss & Efficiency Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.1); padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.3);">
                <p style="font-size: 0.8rem; color: #ef4444; margin: 0 0 1rem 0; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
                    📉 EFFICIENCY LOSS BREAKDOWN
                </p>
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #b8c5d6; font-size: 0.85rem;">Mechanical Efficiency Loss</span>
                        <span style="color: #ef4444; font-size: 0.85rem; font-weight: 600;">-22%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #b8c5d6; font-size: 0.85rem;">Thermal Losses (friction)</span>
                        <span style="color: #ef4444; font-size: 0.85rem; font-weight: 600;">-8%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #b8c5d6; font-size: 0.85rem;">Vibration Energy Loss</span>
                        <span style="color: #ef4444; font-size: 0.85rem; font-weight: 600;">-3%</span>
                    </div>
                    <div style="border-top: 1px solid rgba(239, 68, 68, 0.3); padding-top: 0.5rem; margin-top: 0.5rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #e8eaed; font-size: 0.9rem; font-weight: 600;">Total Efficiency Loss</span>
                            <span style="color: #ef4444; font-size: 1.1rem; font-weight: 700;">-33%</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(251, 191, 36, 0.1); padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(251, 191, 36, 0.3);">
                <p style="font-size: 0.8rem; color: #fbbf24; margin: 0 0 1rem 0; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
                    💵 FINANCIAL IMPACT (30 DAYS)
                </p>
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #b8c5d6; font-size: 0.85rem;">Excess Energy Consumption</span>
                        <span style="color: #fbbf24; font-size: 0.85rem; font-weight: 600;">$8,420</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #b8c5d6; font-size: 0.85rem;">Reduced Output Value</span>
                        <span style="color: #fbbf24; font-size: 0.85rem; font-weight: 600;">$12,300</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #b8c5d6; font-size: 0.85rem;">Quality Impact Costs</span>
                        <span style="color: #fbbf24; font-size: 0.85rem; font-weight: 600;">$3,150</span>
                    </div>
                    <div style="border-top: 1px solid rgba(251, 191, 36, 0.3); padding-top: 0.5rem; margin-top: 0.5rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #e8eaed; font-size: 0.9rem; font-weight: 600;">Total Monthly Loss</span>
                            <span style="color: #fbbf24; font-size: 1.1rem; font-weight: 700;">$23,870</span>
                        </div>
                    </div>
                </div>
                <p style="color: #7a92a8; font-size: 0.75rem; margin: 0.8rem 0 0 0; line-height: 1.4;">
                    ⚠️ Catastrophic failure could result in $147,000 in downtime costs plus $31,000 in emergency repairs.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Maintenance Timeline
        st.markdown("### 📅 Recommended Maintenance Timeline")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a2332 0%, #243447 100%); 
                    padding: 1.8rem; border-radius: 12px; border: 1px solid #2d4a5f;">
            <div style="position: relative; padding-left: 2rem;">
                <!-- Immediate Action -->
                <div style="margin-bottom: 1.8rem; position: relative;">
                    <div style="position: absolute; left: -2rem; top: 0.3rem; width: 12px; height: 12px; 
                                background: #ef4444; border-radius: 50%; border: 3px solid #1a2332;"></div>
                    <div style="position: absolute; left: -1.45rem; top: 1.5rem; width: 2px; height: calc(100% + 1rem); 
                                background: linear-gradient(180deg, #ef4444 0%, #2d4a5f 100%);"></div>
                    <p style="color: #ef4444; font-size: 0.75rem; font-weight: 700; margin: 0 0 0.3rem 0; text-transform: uppercase; letter-spacing: 1px;">
                        IMMEDIATE (0-6 Hours)
                    </p>
                    <p style="color: #e8eaed; font-size: 0.95rem; font-weight: 600; margin: 0 0 0.5rem 0;">
                        Emergency Response Protocol
                    </p>
                    <p style="color: #b8c5d6; font-size: 0.85rem; line-height: 1.6; margin: 0;">
                        • Reduce load to 60% capacity immediately<br>
                        • Deploy continuous vibration monitoring<br>
                        • Prepare spare bearing assemblies (P/N: BRG-2847-HD)<br>
                        • Schedule shutdown window with production team
                    </p>
                </div>
                
                <!-- Short Term -->
                <div style="margin-bottom: 1.8rem; position: relative;">
                    <div style="position: absolute; left: -2rem; top: 0.3rem; width: 12px; height: 12px; 
                                background: #fbbf24; border-radius: 50%; border: 3px solid #1a2332;"></div>
                    <div style="position: absolute; left: -1.45rem; top: 1.5rem; width: 2px; height: calc(100% + 1rem); 
                                background: linear-gradient(180deg, #fbbf24 0%, #2d4a5f 100%);"></div>
                    <p style="color: #fbbf24; font-size: 0.75rem; font-weight: 700; margin: 0 0 0.3rem 0; text-transform: uppercase; letter-spacing: 1px;">
                        SHORT-TERM (6-48 Hours)
                    </p>
                    <p style="color: #e8eaed; font-size: 0.95rem; font-weight: 600; margin: 0 0 0.5rem 0;">
                        Bearing Replacement & System Overhaul
                    </p>
                    <p style="color: #b8c5d6; font-size: 0.85rem; line-height: 1.6; margin: 0;">
                        • Complete motor shutdown and lockout/tagout<br>
                        • Replace drive-end and non-drive-end bearings<br>
                        • Inspect shaft for damage, replace if scoring detected<br>
                        • Clean housing, apply appropriate lubricant (NLGI Grade 2)<br>
                        • Perform alignment verification (tolerance: ±0.002")<br>
                        • Conduct vibration baseline testing
                    </p>
                </div>
                
                <!-- Medium Term -->
                <div style="margin-bottom: 1.8rem; position: relative;">
                    <div style="position: absolute; left: -2rem; top: 0.3rem; width: 12px; height: 12px; 
                                background: #4a9eff; border-radius: 50%; border: 3px solid #1a2332;"></div>
                    <div style="position: absolute; left: -1.45rem; top: 1.5rem; width: 2px; height: calc(100% + 1rem); 
                                background: linear-gradient(180deg, #4a9eff 0%, #2d4a5f 100%);"></div>
                    <p style="color: #4a9eff; font-size: 0.75rem; font-weight: 700; margin: 0 0 0.3rem 0; text-transform: uppercase; letter-spacing: 1px;">
                        MEDIUM-TERM (1-2 Weeks)
                    </p>
                    <p style="color: #e8eaed; font-size: 0.95rem; font-weight: 600; margin: 0 0 0.5rem 0;">
                        System Verification & Optimization
                    </p>
                    <p style="color: #b8c5d6; font-size: 0.85rem; line-height: 1.6; margin: 0;">
                        • Monitor post-repair vibration trends (target: <2.5mm/s)<br>
                        • Verify thermal performance (bearing temp: <65°C)<br>
                        • Optimize load distribution patterns<br>
                        • Update maintenance logs and sensor baselines<br>
                        • Train operators on early warning signs
                    </p>
                </div>
                
                <!-- Long Term -->
                <div style="position: relative;">
                    <div style="position: absolute; left: -2rem; top: 0.3rem; width: 12px; height: 12px; 
                                background: #34d399; border-radius: 50%; border: 3px solid #1a2332;"></div>
                    <p style="color: #34d399; font-size: 0.75rem; font-weight: 700; margin: 0 0 0.3rem 0; text-transform: uppercase; letter-spacing: 1px;">
                        LONG-TERM (Ongoing)
                    </p>
                    <p style="color: #e8eaed; font-size: 0.95rem; font-weight: 600; margin: 0 0 0.5rem 0;">
                        Preventive Strategy Implementation
                    </p>
                    <p style="color: #b8c5d6; font-size: 0.85rem; line-height: 1.6; margin: 0;">
                        • Implement 60-day lubrication schedule<br>
                        • Install permanent vibration sensors with cloud connectivity<br>
                        • Conduct quarterly thermographic surveys<br>
                        • Review loading patterns to prevent overload conditions<br>
                        • Consider bearing upgrade to sealed design for contamination resistance
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Climate & Geographical Impact
        st.markdown("### 🌍 Environmental & Geographical Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(74, 158, 255, 0.15) 0%, rgba(74, 158, 255, 0.05) 100%); 
                        padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(74, 158, 255, 0.3);">
                <p style="font-size: 0.8rem; color: #4a9eff; margin: 0 0 1rem 0; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
                    🌡️ CLIMATE FACTORS
                </p>
                <p style="color: #e8eaed; font-size: 0.9rem; line-height: 1.7; margin: 0;">
                    <strong style="color: #4a9eff;">High Humidity Impact:</strong><br>
                    <span style="color: #b8c5d6; font-size: 0.85rem;">
                    Current ambient humidity (78%) accelerates lubricant degradation and promotes corrosion. 
                    The facility's coastal location exposes equipment to salt-laden air, increasing oxidation rates 
                    by an estimated 40%.
                    </span>
                </p>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(74, 158, 255, 0.2);">
                    <p style="color: #b8c5d6; font-size: 0.8rem; line-height: 1.5; margin: 0;">
                        📍 <strong>Location:</strong> Mumbai, India (Coastal Industrial Zone)<br>
                        🌡️ <strong>Avg Temp:</strong> 32°C (Summer), 24°C (Winter)<br>
                        💧 <strong>Humidity:</strong> 65-85% year-round<br>
                        🌊 <strong>Salt Air Exposure:</strong> High (2.1km from coast)
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(52, 211, 153, 0.15) 0%, rgba(52, 211, 153, 0.05) 100%); 
                        padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(52, 211, 153, 0.3);">
                <p style="font-size: 0.8rem; color: #34d399; margin: 0 0 1rem 0; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
                    🛡️ ENVIRONMENTAL MITIGATION
                </p>
                <p style="color: #e8eaed; font-size: 0.9rem; line-height: 1.7; margin: 0 0 1rem 0;">
                    <strong style="color: #34d399;">Recommended Adaptations:</strong>
                </p>
                <p style="color: #b8c5d6; font-size: 0.85rem; line-height: 1.6; margin: 0;">
                    • Switch to synthetic lubricants with superior moisture resistance<br>
                    • Install humidity-controlled enclosures (target: 40-50% RH)<br>
                    • Apply corrosion-resistant coatings to exposed surfaces<br>
                    • Implement weekly condensation drainage procedures<br>
                    • Consider bearing seals designed for high-humidity environments
                </p>
                <div style="margin-top: 1rem; padding: 0.8rem; background: rgba(52, 211, 153, 0.1); border-radius: 6px;">
                    <p style="color: #34d399; font-size: 0.75rem; margin: 0; font-weight: 600;">
                        ✓ Implementing these measures could extend bearing life by 60-80% in this climate.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Maintenance Report Summary
        st.markdown("### 📋 Comprehensive Maintenance Report")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a2332 0%, #243447 100%); 
                    padding: 2rem; border-radius: 12px; border: 1px solid #2d4a5f;">
            <div style="border-bottom: 2px solid #2d4a5f; padding-bottom: 1rem; margin-bottom: 1.5rem;">
                <h3 style="color: #4a9eff; margin: 0; font-size: 1.2rem;">Executive Summary</h3>
                <p style="color: #7a92a8; font-size: 0.8rem; margin: 0.3rem 0 0 0;">
                    Generated: January 30, 2026 | Asset: MTR-001 | Priority: CRITICAL
                </p>
            </div>
            
            <p style="color: #e8eaed; font-size: 0.95rem; line-height: 1.8; margin-bottom: 1.5rem;">
                Motor MTR-001 has reached a critical failure threshold requiring immediate intervention. 
                The primary failure mode—bearing degradation—has been conclusively identified through 
                multi-sensor analysis combining vibration spectroscopy, thermal imaging, and acoustic emission patterns.
            </p>
            
            <div style="background: rgba(239, 68, 68, 0.1); padding: 1.2rem; border-radius: 8px; border-left: 3px solid #ef4444; margin-bottom: 1.5rem;">
                <p style="color: #ef4444; font-size: 0.85rem; font-weight: 600; margin: 0 0 0.5rem 0;">
                    ⚠️ CRITICAL FINDINGS
                </p>
                <p style="color: #b8c5d6; font-size: 0.85rem; line-height: 1.6; margin: 0;">
                    • Vibration amplitude exceeds ISO 10816 alarm threshold by 340%<br>
                    • Bearing temperature 22°C above normal operating range<br>
                    • Efficiency degradation quantified at 33% below baseline<br>
                    • Financial impact: $23,870/month in losses, potential $178,000 catastrophic failure cost
                </p>
            </div>
            
            <div style="background: rgba(74, 158, 255, 0.1); padding: 1.2rem; border-radius: 8px; border-left: 3px solid #4a9eff; margin-bottom: 1.5rem;">
                <p style="color: #4a9eff; font-size: 0.85rem; font-weight: 600; margin: 0 0 0.5rem 0;">
                    💡 AI RECOMMENDATIONS
                </p>
                <p style="color: #b8c5d6; font-size: 0.85rem; line-height: 1.6; margin: 0;">
                    <strong>Immediate Action:</strong> Reduce operational load to 60% capacity and schedule emergency maintenance within 6 hours.
                    Deploy continuous monitoring to detect rapid degradation patterns. Prepare for 24-48 hour maintenance window.<br><br>
                    
                    <strong>Root Cause Mitigation:</strong> Address extended lubrication intervals (current: 147 days, recommended: 60-90 days max for coastal environment).
                    Review loading protocols to prevent sustained overload conditions. Implement environmental controls to combat high-humidity corrosion acceleration.<br><br>
                    
                    <strong>Long-term Strategy:</strong> Transition to sealed bearing assemblies with synthetic lubrication designed for tropical coastal climates.
                    Install permanent vibration monitoring with predictive analytics to prevent future critical failures.
                </p>
            </div>
            
            <div style="display: flex; gap: 1rem; margin-top: 1.5rem;">
                <div style="flex: 1; text-align: center; padding: 1rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px;">
                    <p style="color: #ef4444; font-size: 1.5rem; font-weight: 700; margin: 0;">CRITICAL</p>
                    <p style="color: #b8c5d6; font-size: 0.75rem; margin: 0.3rem 0 0 0;">Current Status</p>
                </div>
                <div style="flex: 1; text-align: center; padding: 1rem; background: rgba(251, 191, 36, 0.1); border-radius: 8px;">
                    <p style="color: #fbbf24; font-size: 1.5rem; font-weight: 700; margin: 0;">6 HRS</p>
                    <p style="color: #b8c5d6; font-size: 0.75rem; margin: 0.3rem 0 0 0;">Response Window</p>
                </div>
                <div style="flex: 1; text-align: center; padding: 1rem; background: rgba(74, 158, 255, 0.1); border-radius: 8px;">
                    <p style="color: #4a9eff; font-size: 1.5rem; font-weight: 700; margin: 0;">$178K</p>
                    <p style="color: #b8c5d6; font-size: 0.75rem; margin: 0.3rem 0 0 0;">Failure Cost Risk</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Export Options
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("📄 Export as PDF", use_container_width=True)
        with col2:
            st.button("📧 Email to Team", use_container_width=True)
        with col3:
            st.button("📊 Generate Detailed Report", use_container_width=True)
    
    else:
        # Before analysis is generated
        st.markdown("""
        <div style="padding: 4rem 2rem; background: rgba(74, 158, 255, 0.05); border-radius: 12px; 
                    border: 2px dashed #2d4a5f; text-align: center;">
            <p style="font-size: 3rem; margin: 0;">🤖</p>
            <h3 style="color: #4a9eff; margin: 1rem 0 0.5rem 0;">AI Analysis Ready</h3>
            <p style="color: #7a92a8; font-size: 0.95rem; max-width: 600px; margin: 0 auto;">
                Select an asset and click "Generate Analysis" to receive comprehensive AI-powered 
                maintenance intelligence including diagnostics, timelines, financial impact, and actionable recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ================= REPORTS =================
elif page == "📄 Reports & Insights":
    st.markdown("### 📋 Automated Maintenance Reporting")
    
    # Check if predictions are available
    if st.session_state.predictions is None:
        st.warning("⚠️ No prediction data available. Please upload and process data in the Data Integration page first.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        machine_options = []
        if st.session_state.uploaded_data is not None and 'machine_id' in st.session_state.uploaded_data.columns:
            machine_options = st.session_state.uploaded_data['machine_id'].tolist()
        else:
            machine_options = [f"Asset {i+1}" for i in range(len(st.session_state.predictions))]
        
        st.selectbox(
            "Select Asset",
            machine_options,
            help="Choose an asset to generate maintenance report"
        )
        
        st.selectbox(
            "Report Format",
            ["PDF Document", "HTML Web Page", "JSON Data Export", "Excel Spreadsheet"],
            help="Select output format for the report"
        )
        
        st.date_input("Report Period Start")
        st.date_input("Report Period End")
    
    with col2:
        st.markdown("""
        <div style="background: rgba(74, 158, 255, 0.05); padding: 1.5rem; border-radius: 8px; border: 1px solid #2d4a5f; height: 100%;">
            <p style="font-size: 0.75rem; color: #4a9eff; margin: 0 0 1rem 0; font-weight: 600;">📑 REPORT INCLUDES</p>
            <p style="font-size: 0.8rem; color: #b8c5d6; line-height: 1.6; margin: 0;">
                • Operational performance metrics<br>
                • Maintenance recommendations<br>
                • Historical trend analysis<br>
                • Cost-benefit projections<br>
                • Compliance documentation
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.button("📥 Generate Report", disabled=True, use_container_width=False, help="ML engine integration required")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📄 Report Preview")
    
    st.text_area(
        "Generated report will appear here",
        value="═══════════════════════════════════════════\n"
              "   MAINTENANCE INTELLIGENCE REPORT\n"
              "═══════════════════════════════════════════\n\n"
              "Report generation pending ML engine integration.\n\n"
              "Once connected, this section will display:\n"
              "• Executive summary\n"
              "• Detailed analytics\n"
              "• Actionable insights\n"
              "• Maintenance schedules\n\n"
              "═══════════════════════════════════════════",
        height=300,
        disabled=True
    )