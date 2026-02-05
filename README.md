# 🚗 Colorado Motor Vehicle Sales Analysis & Forecasting

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![EDA](https://img.shields.io/badge/EDA-Advanced-green)
![ML](https://img.shields.io/badge/Machine%20Learning-RandomForest-orange)
![TimeSeries](https://img.shields.io/badge/Time%20Series-ARIMA-red)

---

## 📌 Project Overview

This project performs an **advanced financial and statistical analysis** on Colorado Motor Vehicle Sales data.  
The objective is to **understand sales trends**, **perform exploratory data analysis**, and **forecast future vehicle sales** using time-series modeling.

The project is suitable for:
- Finance Analysts
- Data Analysts
- Business Intelligence Roles
- Academic & Client Submissions

---

## 🎯 Objectives

- Analyze quarterly vehicle sales trends across Colorado counties
- Perform deep Exploratory Data Analysis (EDA)
- Identify seasonal and economic patterns
- Build Machine Learning models for prediction
- Forecast future sales using ARIMA time-series modeling
- Generate actionable insights for business decisions

---

## 🗂 Dataset Information

**Dataset Name:** Colorado Motor Vehicle Sales Data  
**Format:** CSV  
**Granularity:** Quarterly  
**Key Columns:**
- `year`
- `quarter`
- `county`
- `sales`

---

## 🛠️ Tools & Technologies Used

- **Python**
- **Pandas & NumPy** – Data Processing
- **Matplotlib & Seaborn** – Visualization
- **Scikit-learn** – Machine Learning
- **Statsmodels** – Time Series (ARIMA)
- **Streamlit** – Dashboard (Optional Extension)

---

## 📊 Exploratory Data Analysis (EDA)

### Key Insights:
- Sedan cars with Petrol engines dominate sales
- Sales show clear seasonal behavior across quarters
- Price is negatively correlated with mileage
- Registered vehicles have significantly higher prices
- Engine value positively impacts vehicle price

Visualizations include:
- Time-series plots
- Correlation heatmaps
- Distribution plots
- Boxplots with median labels
- Interactive county-wise sales charts

---

## 🤖 Machine Learning Model

- **Algorithm:** Random Forest Regressor
- **Features Used:** Year, Quarter, County
- **Metric:** RMSE
- **Purpose:** Sales prediction (demonstration-level)

---

## ⏳ Time Series Forecasting

- **Model Used:** ARIMA
- **Frequency:** Quarterly
- **Output:** Forecast for next 8–12 periods
- **Findings:** Consistent trend with seasonal variation

---

## 📈 Results Summary

- Strong quarterly seasonality observed
- Sales concentrated in major counties
- Forecasting model provides reliable trend estimation
- Insights useful for policy planning and market strategy

---

## ▶️ How to Run the Project

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd project-folder
```

## Step 2: Install Dependencies
``` bash
pip install -r requirements.txt
```
## Step 3: Run the Application
``` bash
streamlit run app.py
```

---

## 📁 Project Structure

📦 Colorado-Motor-Vehicle-Sales<br>
 ┣ 📜 app.py <br>
 ┣ 📜 requirements.txt <br>
 ┣ 📜 colorado_motor_vehicle_sales.csv <br>
 ┣ 📜 README.md <br>

 ---
 ## 📌 Future Enhancements

- Full Streamlit dashboard
- SARIMA model for improved seasonality
- County-wise forecasting
- Power BI / Tableau integration
- Cloud deployment

 ---

## 📚 Reference

GitHub Reference Project:
https://github.com/Subham2S/EDA-Car-Sales

---

## 👤 Author  
**Pranuth HM**  
🔗 [GitHub Profile](https://github.com/PranuthHM)
🔗 [LinedIn Profile](https://www.linkedin.com/in/pranuth-hm)
🔗 [Portfolio Profile](https://pranuth.netlify.app/)
