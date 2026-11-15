
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="🚚",
    layout="wide"
)

# Load project files
df = pd.read_csv("supply_chain_dataset.csv")
model = joblib.load("supply_chain_risk_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Dashboard Title
st.title("🚚 AI-Powered Supply Chain Disruption Intelligence")
st.caption("Supply Chain Risk Monitoring & Early Warning System")

st.divider()

# KPI Metrics
total_shipments = len(df)
total_suppliers = df["Supplier_ID"].nunique()
delayed = (df["Delivery_Delay_Days"] > 0).sum()
high_risk = (df["Disruption_Risk"] == "High").sum()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Shipments", total_shipments)
c2.metric("Total Suppliers", total_suppliers)
c3.metric("Delayed Shipments", int(delayed))
c4.metric("High Risk Shipments", int(high_risk))

st.divider()

# Risk Analysis
st.subheader("📊 Shipment Risk Analysis")

risk_counts = df["Disruption_Risk"].value_counts().reset_index()
risk_counts.columns = ["Risk Level", "Shipments"]

fig1 = px.pie(
    risk_counts,
    names="Risk Level",
    values="Shipments",
    title="Disruption Risk Distribution"
)

st.plotly_chart(fig1, use_container_width=True)

# Transportation Analysis
st.subheader("🚛 Transportation Cost Analysis")

cost_data = df.groupby(
    "Transportation_Mode"
)["Transportation_Cost"].mean().reset_index()

fig2 = px.bar(
    cost_data,
    x="Transportation_Mode",
    y="Transportation_Cost",
    title="Average Transportation Cost"
)

st.plotly_chart(fig2, use_container_width=True)

# Supplier Analysis
st.subheader("🏭 Supplier Performance")

supplier_data = df.groupby("Supplier_ID").agg(
    Total_Shipments=("Shipment_ID", "count"),
    Average_Rating=("Supplier_Rating", "mean"),
    Average_Delay=("Delivery_Delay_Days", "mean")
).reset_index()

st.dataframe(supplier_data, use_container_width=True)

# Weather Analysis
st.subheader("🌦️ Weather Impact")

weather_data = df.groupby(
    "Weather_Condition"
)["Delivery_Delay_Days"].mean().reset_index()

fig3 = px.bar(
    weather_data,
    x="Weather_Condition",
    y="Delivery_Delay_Days",
    title="Average Delay by Weather"
)

st.plotly_chart(fig3, use_container_width=True)

# AI Prediction
st.divider()
st.subheader("🤖 AI Shipment Risk Prediction")

with st.form("prediction_form"):

    supplier = st.number_input(
        "Supplier ID", 101, 120, 105
    )

    category = st.selectbox(
        "Product Category",
        ["Electronics", "Clothing", "Food", "Automotive"]
    )

    transport = st.selectbox(
        "Transportation Mode",
        ["Air", "Sea", "Road", "Rail"]
    )

    distance = st.number_input(
        "Distance (KM)", 100, 5000, 1000
    )

    cost = st.number_input(
        "Transportation Cost", 500, 10000, 5000
    )

    weather = st.selectbox(
        "Weather Condition",
        ["Normal", "Rainy", "Stormy", "Extreme"]
    )

    rating = st.slider(
        "Supplier Rating", 1, 5, 3
    )

    inventory = st.number_input(
        "Inventory Level", 10, 500, 100
    )

    delay = st.number_input(
        "Delivery Delay Days", 0, 15, 2
    )

    submitted = st.form_submit_button(
        "Predict Disruption Risk"
    )

if submitted:

    new_data = pd.DataFrame([{
        "Supplier_ID": supplier,
        "Product_Category": category,
        "Transportation_Mode": transport,
        "Distance_KM": distance,
        "Transportation_Cost": cost,
        "Weather_Condition": weather,
        "Supplier_Rating": rating,
        "Inventory_Level": inventory,
        "Delivery_Delay_Days": delay
    }])

    encoded = pd.get_dummies(new_data)
    encoded = encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(encoded)[0]
    probabilities = model.predict_proba(encoded)[0]

    st.subheader("Prediction Result")
    st.metric("Predicted Risk", prediction)

    probability_data = pd.DataFrame({
        "Risk": model.classes_,
        "Probability": probabilities
    })

    st.plotly_chart(
        px.bar(
            probability_data,
            x="Risk",
            y="Probability",
            title="AI Risk Probability"
        ),
        use_container_width=True
    )

    if prediction == "High":
        st.error("HIGH RISK: Immediate action required!")
    elif prediction == "Medium":
        st.warning("MEDIUM RISK: Monitor shipment carefully.")
    else:
        st.success("LOW RISK: Normal monitoring.")

    st.subheader("💡 Business Recommendations")

    if rating <= 2:
        st.write("- Review supplier performance.")

    if delay >= 7:
        st.write("- Contact logistics team immediately.")

    if inventory <= 50:
        st.write("- Increase safety stock.")

    if weather in ["Stormy", "Extreme"]:
        st.write("- Consider alternative transportation routes.")

    if prediction == "High":
        st.write("- Arrange backup suppliers.")

# Early Warning Alerts
st.divider()
st.subheader("🚨 Early Warning Alerts")

alerts = df[df["Disruption_Risk"] == "High"]

st.write("High-risk shipment records:")
st.dataframe(alerts, use_container_width=True)

st.caption("Developed using Python, Machine Learning and Streamlit.")
