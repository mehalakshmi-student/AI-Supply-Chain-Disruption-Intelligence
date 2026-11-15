# AI-Powered Supply Chain Disruption Intelligence & Early Warning System

## Project Overview
This project is a machine learning-based dashboard designed to analyze supply chain shipment data and classify shipment disruption risk levels.

It provides shipment insights, supplier performance analysis, weather impact analysis, and risk predictions through an interactive Streamlit dashboard.

## Objectives
- Analyze shipment performance.
- Identify high-risk shipments.
- Analyze supplier performance.
- Understand weather impact on deliveries.
- Provide risk-based business recommendations.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Streamlit
- Joblib

## Key Features
- Interactive shipment dashboard.
- Shipment risk distribution.
- Supplier performance analysis.
- Transportation cost analysis.
- Weather impact analysis.
- Machine learning risk classification.
- AI risk probability visualization.
- Business recommendations.

## Machine Learning Model
Random Forest Classifier is used to classify shipment disruption risk into:
- Low Risk
- Medium Risk
- High Risk

**Dataset Note:** This project currently uses a synthetic dataset. Risk labels are generated using predefined rules, so model predictions demonstrate the project workflow and are not validated real-world forecasts.

## Project Structure
- `app.py` – Streamlit dashboard
- `supply_chain_dataset.csv` – Shipment dataset
- `supply_chain_risk_model.pkl` – Trained model
- `model_columns.pkl` – Model feature columns
- `requirements.txt` – Required libraries
- `supplier_performance_report.csv`
- `category_analysis_report.csv`
- `delivery_delay_report.csv`
- `weather_impact_report.csv`

## Run the Project
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed using Streamlit Community Cloud.

## Author
Mehalakshmi M.

B.Tech Artificial Intelligence and Data Science
