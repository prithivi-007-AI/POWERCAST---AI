import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from core.preprocessor import LoadPreprocessor
from core.model import ForecastModel
from core.decision_support import DecisionSupport
import os

# Page Config
st.set_page_config(page_title="Powercast-AI Dashboard", layout="wide", page_icon="⚡")

# Sidebar - User Inputs
with st.sidebar:
    st.title("⚡ Powercast-AI")
    st.markdown("---")
    st.header("⚙️ Configuration")

    with st.expander("📁 Data Source", expanded=True):
        uploaded_file = st.file_uploader("Upload Historical Load CSV", type=["csv"])

    with st.expander("🧠 Model Parameters", expanded=True):
        look_back = st.slider("Look-back Window (Hours)", 6, 48, 24, help="Number of past hours to use for predicting the next hour.")

        st.write("Forecast Horizon")
        horizon_mode = st.radio("Mode", ["Hours", "Days"], horizontal=True)

        if horizon_mode == "Hours":
            horizon = st.slider("Horizon (Hours)", 1, 72, 24)
        else:
            days = st.number_input("Horizon (Days)", min_value=1, max_value=365, value=7)
            horizon = days * 24

        if horizon > 72:
            st.warning("⚠️ Accuracy may decrease for long horizons (>72h) due to recursive error accumulation.")

    with st.expander("🏭 Generator Capacities (MW)", expanded=False):
        unit1 = st.number_input("Unit 1", value=300)
        unit2 = st.number_input("Unit 2", value=250)
        unit3 = st.number_input("Unit 3", value=200)
        capacities = [unit1, unit2, unit3]

    st.markdown("---")
    st.info("Powercast-AI v1.1\nProfessional Edition")

# Initialize modules
preprocessor = LoadPreprocessor()
model = ForecastModel(look_back=look_back)
decision_support = DecisionSupport(capacities)

# Main Content
st.title("⚡ Powercast-AI: Intelligent Load Management")
st.markdown("### Advanced Load Forecasting & Decision Support System")

if uploaded_file is not None or os.path.exists('data/historical_load.csv'):
    # Load Data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv('data/historical_load.csv')
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Preprocessing
    df['smoothed_load'] = preprocessor.smooth_data(df['load'].values)
    
    # KPIs at the top
    current_load = df['load'].iloc[-1]
    avg_load_24h = df['load'].iloc[-24:].mean()
    peak_load_hist = df['load'].max()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Current Load", f"{current_load:.2f} MW", delta=f"{current_load - df['load'].iloc[-2]:.2f}")
    kpi2.metric("24h Avg Load", f"{avg_load_24h:.2f} MW")
    kpi3.metric("Historical Peak", f"{peak_load_hist:.2f} MW")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "🔮 Forecast", "💡 Decision Support"])

    with tab1:
        st.subheader("Historical Load Analysis")
        st.markdown("Comparison of raw historical load versus smoothed signal used for training.")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['load'], name="Actual Load", line=dict(color='gray', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['smoothed_load'], name="Smoothed Load", line=dict(color='#0068C9', width=2)))
        fig.update_layout(height=500, xaxis_title="Timestamp", yaxis_title="Load (MW)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"Load Forecast ({horizon} hours)")

        # Prepare Data for Training
        data_values = df['smoothed_load'].values.reshape(-1, 1)
        normalized_data = preprocessor.normalize_data(data_values)
        X, y = preprocessor.prepare_sliding_window(normalized_data, look_back)

        # Train Model
        with st.spinner("Training forecasting model..."):
            model.train(X, y.flatten())

        # Forecasting
        last_window = normalized_data[-look_back:]

        with st.spinner(f"Generating forecast for {horizon} hours..."):
            predictions_norm = model.multi_step_forecast(last_window, horizon)
            predictions = preprocessor.inverse_transform(predictions_norm.reshape(-1, 1)).flatten()

        # Future timestamps
        last_ts = df['timestamp'].iloc[-1]
        future_ts = [last_ts + timedelta(hours=i+1) for i in range(horizon)]

        # Visualization
        fig_pred = go.Figure()
        # Show last 7 days of history if available, else all
        history_plot_points = min(len(df), 24*7)
        fig_pred.add_trace(go.Scatter(x=df['timestamp'].iloc[-history_plot_points:], y=df['load'].iloc[-history_plot_points:], name="Historical (Last 7 Days)", line=dict(color='gray', width=1)))
        fig_pred.add_trace(go.Scatter(x=future_ts, y=predictions, name="Forecast", line=dict(color='#FF4B4B', width=2)))

        fig_pred.update_layout(
            height=600,
            xaxis_title="Timestamp",
            yaxis_title="Load (MW)",
            template="plotly_white",
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right")
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Download Forecast
        forecast_df = pd.DataFrame({'Timestamp': future_ts, 'Predicted_Load': predictions})
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name='forecast.csv',
            mime='text/csv',
        )

    with tab3:
        st.subheader("Generation Planning & Maintenance")

        # Unit Commitment for Peak Load in horizon
        peak_load = np.max(predictions)
        avg_load_forecast = np.mean(predictions)

        on_units, off_units, total_cap = decision_support.recommend_units(peak_load)
        maintenance_indices, threshold = decision_support.identify_maintenance_windows(predictions)

        # Metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Peak Predicted Load", f"{peak_load:.2f} MW")
        m_col2.metric("Total Online Capacity", f"{total_cap} MW")
        m_col3.metric("Maintenance Threshold", f"{threshold:.2f} MW")

        st.markdown("#### Unit Commitment Recommendation")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.success(f"**✅ Suggested ON Units (Total: {len(on_units)})**")
            for unit in on_units:
                st.write(f"- {unit}")

        with d_col2:
            st.error(f"**❌ Suggested OFF Units (Total: {len(off_units)})**")
            for unit in off_units:
                st.write(f"- {unit}")

        # Maintenance Planning
        st.markdown("#### 🛠️ Maintenance Planning Opportunities")
        maintenance_times = [future_ts[i] for i in maintenance_indices]
        if maintenance_times:
            st.info(f"Recommended maintenance windows identified during low-load periods (< {threshold:.2f} MW)")

            # Group consecutive times
            windows = []
            if maintenance_times:
                current_window = [maintenance_times[0]]
                for i in range(1, len(maintenance_times)):
                    if maintenance_times[i] - maintenance_times[i-1] == timedelta(hours=1):
                        current_window.append(maintenance_times[i])
                    else:
                        windows.append(current_window)
                        current_window = [maintenance_times[i]]
                windows.append(current_window)

            for win in windows:
                if len(win) > 1:
                    st.write(f"📅 **Window:** {win[0].strftime('%Y-%m-%d %H:%M')} to {win[-1].strftime('%Y-%m-%d %H:%M')} ({len(win)} hours)")
                else:
                    st.write(f"📅 **Slot:** {win[0].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.write("No specific maintenance windows identified for this horizon.")

else:
    st.info("👋 Welcome to Powercast-AI. Please upload a CSV file or ensure `data/historical_load.csv` exists to begin.")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Generate Sample Data"):
            import utils.generate_sample_data as gsd
            gsd.generate_sample_load_data()
            st.rerun()
