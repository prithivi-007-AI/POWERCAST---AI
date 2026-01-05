# ⚡ Powercast-AI: Load Forecasting & Decision Support

An AI-based electrical load forecasting and decision support application designed for power system planning and control-room style visualization.

## 🌟 Key Features

- **Signal Preprocessing**: Savitzky–Golay smoothing (window=11, poly=2) to preserve trends while removing noise.
- **AI-Powered Forecasting**: Multi-horizon load forecasting using advanced Neural Networks (MLP/LSTM).
- **Decision Support**: 
    - **Unit Commitment**: Automated ON/OFF recommendations for generator units based on predicted demand + 10% spinning reserve.
    - **Maintenance Planning**: Identification of low-load windows for optimal maintenance scheduling.
- **Interactive Visualization**: Real-time plots of historical data, smoothed trends, and future forecasts.

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/Powercast-AI.git
cd Powercast-AI
pip install -r requirements.txt
```

### 2. Run the App
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

## 📂 Project Structure

- `app.py`: Main Streamlit dashboard.
- `core/`: Core logic modules.
  - `preprocessor.py`: Signal smoothing and normalization.
  - `model.py`: AI forecasting engine (scikit-learn fallback for compatibility).
  - `decision_support.py`: Rule-based generation scheduling logic.
- `utils/`: Utility scripts (e.g., sample data generator).
- `data/`: Folder for historical load datasets.

##📂 Dataset Requirements
-Upload a CSV file containing the following mandatory columns:
-timestamp → Date & time in a consistent format
-load → Load demand values (e.g., MW)
-For testing and reference, use the dataset provided in the GitHub repository:
-File name: mock data(2).csv
-Follow this file strictly for column names and formatting.

##⚙️ Application Workflow
1. Upload Data
-Use the sidebar to upload the CSV file.
-Ensure the dataset follows the same structure as mock data(2).csv.

2. Set Parameters
-Look-back Window:
  -Number of past time steps used by the model to learn patterns.
-Forecast Horizon:
  -Number of future steps the model will predict.
   
3. Choose Forecast Horizon (Recommended Setting)
-Select 36 steps (daily) as the forecast horizon for accurate and stable predictions.
-This configuration balances short-term fluctuations and long-term trends effectively.

4. Dataset Size for Accurate Prediction
-To achieve reliable forecasting performance:
  -Recommended historical data size = 5× to 10× the forecast horizon
-For a 36-step daily forecast:
  -Minimum required data: 180–360 days
  -More data improves seasonal learning and reduces prediction error.

5. Configure Generators
-Input individual generator capacities in the sidebar.
-The system provides:
  -Real-time unit commitment recommendations
  -Optimal generator usage based on predicted load

6. Analyze & Monitor
View results on the interactive dashboard, including:
Load demand predictions
Generator utilization insights
Maintenance and efficiency suggestions

##📊 Output Features
-Time-series load forecasting
-Generator-wise power allocation
-Decision-support insights for operational planning

##🧪 Testing & Validation
-Use mock data(2).csv from the GitHub repository for:
-Model testing
-UI validation
-Forecast verification

## 📄 License
MIT License
