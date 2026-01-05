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

def load_data(uploaded_file):
    """
    Loads data from various file formats (CSV, Excel, Parquet).
    Returns a DataFrame.
    """
    filename = uploaded_file.name
    if filename.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(uploaded_file)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(uploaded_file)
    else:
        st.error("Unsupported file format.")
        return None
    return df

def infer_frequency(df, timestamp_col='timestamp'):
    """
    Infers the frequency of the time series data.
    Returns the frequency string (e.g., '1h', '15min') and the timedelta.
    """
    # Try to find the timestamp column if not provided or incorrect
    if timestamp_col not in df.columns:
        # Heuristic: look for columns with 'date', 'time', or 'ts' in the name
        possible_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'ts' in c.lower()]
        if possible_cols:
            timestamp_col = possible_cols[0]
        else:
            return None, None, None

    # Parse datetime
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except Exception:
        return None, None, None

    df = df.sort_values(by=timestamp_col)

    # Infer frequency
    if len(df) > 1:
        diffs = df[timestamp_col].diff().dropna()
        # Get the mode of the time differences
        if not diffs.empty:
            common_diff = diffs.mode()[0]

            # Determine a human readable label
            seconds = common_diff.total_seconds()
            if seconds == 3600:
                freq_label = "Hourly"
                freq_str = "1h"
            elif seconds == 900:
                freq_label = "15-min"
                freq_str = "15min"
            elif seconds == 86400:
                freq_label = "Daily"
                freq_str = "1d"
            else:
                freq_label = f"{int(seconds)}s"
                freq_str = f"{int(seconds)}S"

            return timestamp_col, common_diff, freq_label

    return timestamp_col, timedelta(hours=1), "Hourly (Default)"

# Sidebar - User Inputs
with st.sidebar:
    st.title("⚡ Powercast-AI")
    st.markdown("---")
    st.header("⚙️ Configuration")

    with st.expander("📁 Data Source", expanded=True):
        uploaded_file = st.file_uploader("Upload Historical Load (CSV, Excel, Parquet)", type=["csv", "xlsx", "xls", "parquet"])

    # Placeholder variables for later updates
    freq_label = "Hourly"
    time_delta = timedelta(hours=1)

    # We need to load data early to adjust sliders dynamically,
    # but Streamlit runs top-to-bottom. We can handle defaults first.

    # Default capacities
    with st.expander("🏭 Generator Capacities (MW)", expanded=False):
        unit1 = st.number_input("Unit 1", value=300)
        unit2 = st.number_input("Unit 2", value=250)
        unit3 = st.number_input("Unit 3", value=200)
        capacities = [unit1, unit2, unit3]

    st.markdown("---")
    st.info("Powercast-AI v1.2\nProfessional Edition")

# Initialize modules
preprocessor = LoadPreprocessor()
decision_support = DecisionSupport(capacities)

# Main Content
st.title("⚡ Powercast-AI: Intelligent Load Management")
st.markdown("### Advanced Load Forecasting & Decision Support System")

df = None
if uploaded_file is not None:
    df = load_data(uploaded_file)
elif os.path.exists('data/historical_load.csv'):
    df = pd.read_csv('data/historical_load.csv')

if df is not None:
    # Intelligent Analysis
    ts_col, time_delta, freq_label = infer_frequency(df)
    
    if ts_col is None:
        st.error("Could not identify a timestamp column. Please ensure your data has a column with date/time information.")
        st.stop()

    df.rename(columns={ts_col: 'timestamp'}, inplace=True)
    
    # Find load column (assume numerical and not timestamp)
    load_col = None
    for col in df.columns:
        if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
            load_col = col
            break

    if load_col is None:
        st.error("Could not identify a numerical load column.")
        st.stop()

    df.rename(columns={load_col: 'load'}, inplace=True)

    # Sort just in case
    df = df.sort_values('timestamp')

    # Sidebar - Dynamic Model Parameters
    with st.sidebar:
        with st.expander("🧠 Model Parameters", expanded=True):
            st.info(f"Detected Frequency: **{freq_label}**")

            # Adjust defaults based on frequency
            default_lookback = 24
            default_horizon = 24

            if freq_label == "Daily":
                default_lookback = 7 # 1 week
                default_horizon = 7
            elif freq_label == "15-min":
                default_lookback = 96 # 1 day
                default_horizon = 96

            look_back = st.slider(f"Look-back Window ({freq_label} Steps)", 6, max(48, default_lookback*2), default_lookback)

            st.write("Forecast Horizon")
            horizon_mode = st.radio("Mode", ["Steps", "Time"], horizontal=True)

            if horizon_mode == "Steps":
                horizon = st.slider(f"Horizon ({freq_label} Steps)", 1, max(72, default_horizon*3), default_horizon)
            else:
                # Time mode
                if freq_label == "Daily":
                    days = st.number_input("Horizon (Days)", min_value=1, max_value=365, value=7)
                    horizon = days
                elif freq_label == "Hourly":
                    days = st.number_input("Horizon (Days)", min_value=1, max_value=365, value=7)
                    horizon = days * 24
                elif freq_label == "15-min":
                     hours = st.number_input("Horizon (Hours)", min_value=1, max_value=720, value=24)
                     horizon = hours * 4
                else:
                    steps = st.number_input("Horizon (Steps)", min_value=1, max_value=1000, value=24)
                    horizon = steps

            if horizon > 1000: # Generic warning threshold
                st.warning("⚠️ Accuracy may decrease for very long horizons.")

    # Initialize model with updated look_back
    model = ForecastModel(look_back=look_back)

    # Preprocessing
    df['smoothed_load'] = preprocessor.smooth_data(df['load'].values)
    
    # KPIs at the top
    current_load = df['load'].iloc[-1]
    # Calculate avg based on ~24 hours worth of data if possible
    steps_24h = int(timedelta(hours=24) / time_delta)
    if steps_24h < 1: steps_24h = 1

    avg_load_24h = df['load'].iloc[-steps_24h:].mean()
    peak_load_hist = df['load'].max()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Current Load", f"{current_load:.2f} MW", delta=f"{current_load - df['load'].iloc[-2]:.2f}")
    kpi2.metric(f"Avg Load (Last {steps_24h} steps)", f"{avg_load_24h:.2f} MW")
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
        st.subheader(f"Load Forecast ({horizon} steps)")

        # Prepare Data for Training
        data_values = df['smoothed_load'].values.reshape(-1, 1)
        normalized_data = preprocessor.normalize_data(data_values)
        X, y = preprocessor.prepare_sliding_window(normalized_data, look_back)

        # Train Model
        with st.spinner("Training forecasting model..."):
            model.train(X, y.flatten())

        # Forecasting
        last_window = normalized_data[-look_back:]

        with st.spinner(f"Generating forecast for {horizon} steps..."):
            predictions_norm = model.multi_step_forecast(last_window, horizon)
            predictions = preprocessor.inverse_transform(predictions_norm.reshape(-1, 1)).flatten()

        # Future timestamps
        last_ts = df['timestamp'].iloc[-1]
        future_ts = [last_ts + time_delta * (i+1) for i in range(horizon)]

        # Visualization
        fig_pred = go.Figure()
        # Show last ~7 days of history if available
        steps_7days = int(timedelta(days=7) / time_delta)
        history_plot_points = min(len(df), steps_7days)

        fig_pred.add_trace(go.Scatter(x=df['timestamp'].iloc[-history_plot_points:], y=df['load'].iloc[-history_plot_points:], name="Historical", line=dict(color='gray', width=1)))
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
                    # Check if consecutive based on time_delta
                    if maintenance_times[i] - maintenance_times[i-1] == time_delta:
                        current_window.append(maintenance_times[i])
                    else:
                        windows.append(current_window)
                        current_window = [maintenance_times[i]]
                windows.append(current_window)

            for win in windows:
                if len(win) > 1:
                    duration_str = f"{len(win)} steps"
                    # Try to give a time duration
                    total_seconds = len(win) * time_delta.total_seconds()
                    if total_seconds >= 3600:
                        duration_str = f"{total_seconds/3600:.1f} hours"
                    else:
                        duration_str = f"{total_seconds/60:.1f} minutes"

                    st.write(f"📅 **Window:** {win[0].strftime('%Y-%m-%d %H:%M')} to {win[-1].strftime('%Y-%m-%d %H:%M')} ({duration_str})")
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
