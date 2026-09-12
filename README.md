# Intelligent Multi-Modal Weather Intelligence and Climate Decision Support Platform

## 🚀 Live Demo

[Open the Weather Intelligence Platform](https://weather-intelligence-platform-joacfobwrvbtwsjuojmhuz.streamlit.app/)

An end-to-end weather intelligence platform combining **historical weather analysis, machine learning, FastAPI, Streamlit, Retrieval-Augmented Generation (RAG), and Generative AI**.

The project uses historical hourly weather data from **Kochi, Kerala (2015–2025)** to analyze weather patterns and predict **next-hour temperature** using machine-learning models. The trained model is exposed through a FastAPI backend and integrated into an interactive Streamlit dashboard. A local GenAI assistant using **Gemma 3 4B with Ollama** provides grounded answers and explanations using project-specific information.

---

## 1. Project Overview

Weather data contains valuable temporal and environmental patterns that can be used for prediction and decision support.

This project develops a complete weather intelligence workflow:

**Data Collection → Preprocessing → EDA → Feature Engineering → ML Prediction → API → Dashboard → RAG → GenAI**

The primary machine-learning task is to predict the **temperature for the next hour** using current weather conditions, temporal features, and historical lag features.

---

## 2. Objectives

- Collect and analyze historical weather data.
- Perform data cleaning and preprocessing.
- Identify weather patterns through exploratory data analysis.
- Engineer temporal and lag-based features.
- Compare multiple machine-learning models.
- Develop a next-hour temperature prediction system.
- Deploy the prediction model through FastAPI.
- Build an interactive Streamlit dashboard.
- Integrate a local Generative AI assistant.
- Implement a lightweight RAG pipeline for grounded responses.
- Combine ML predictions with GenAI-generated explanations.

---

## 3. System Architecture

```text
                Historical Weather Data
                         |
                         v
                Open-Meteo API
                         |
                         v
              Data Preprocessing
                         |
                         v
             Feature Engineering
                         |
              +----------+----------+
              |                     |
              v                     v
             EDA              ML Dataset
                                    |
                                    v
                           Model Training
                                    |
                 +------------------+------------------+
                 |                  |                  |
                 v                  v                  v
          Linear Regression    Random Forest       XGBoost
                 |                  |                  |
                 +------------------+------------------+
                                    |
                                    v
                              Best Model
                              XGBoost
                                    |
                     +--------------+--------------+
                     |                             |
                     v                             v
                  FastAPI                    RAG + GenAI
                     |                             |
                     +--------------+--------------+
                                    |
                                    v
                            Streamlit Dashboard
                                    |
                                    v
                       Weather Intelligence
                            Assistant
4. Dataset

Historical hourly weather data was collected for Kochi, Kerala, India using the Open-Meteo Historical Weather API.

Dataset Information
Property	Value
Location	Kochi, Kerala, India
Period	2015–2025
Frequency	Hourly
Records	96,432
Original Variables	12
Missing Values	0
Duplicate Records	0
Weather Variables
Temperature
Relative Humidity
Apparent Temperature
Precipitation
Rain
Weather Code
Surface Pressure
Cloud Cover
Wind Speed
Wind Direction
Wind Gusts
5. Data Preprocessing

The raw weather data was processed before machine-learning development.

The preprocessing workflow included:

Converting timestamps into datetime format.
Checking missing values.
Checking duplicate records.
Extracting temporal features.
Creating year, month, day, hour and day-of-year features.
Classifying observations into seasons.
Creating historical lag features.
Creating the next-hour temperature target.
Maintaining chronological order for model evaluation.

The processed machine-learning dataset contains:

96,407 observations
17 input features
1 target variable
Target
target_temperature_next_hour
6. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the behaviour of weather variables and identify useful patterns.

Analyses Performed
Temperature distribution
Monthly temperature variation
Monthly rainfall variation
Seasonal temperature analysis
Yearly temperature trends
Correlation analysis
Key Findings
Temperature generally remained around 25–28°C.
Higher monthly temperatures were observed around March–April.
Monsoon periods showed comparatively lower temperatures.
Rainfall showed strong seasonal variation.
Temperature and relative humidity showed a strong negative correlation.
Temperature and apparent temperature showed a strong positive correlation.
Wind speed and wind gusts showed a strong positive relationship.

EDA visualizations are available in:

screenshots/EDA/
7. Machine Learning
Prediction Problem

The model predicts:

The temperature for the next hour.

The input consists of current weather conditions, temporal features and historical lag features.

Models Compared

Three models were evaluated:

Linear Regression
Random Forest
XGBoost

A chronological train-test split was used instead of a random split because weather observations are time-dependent.

Model Results
Model	MAE	RMSE	R²
Linear Regression	0.4029°C	0.5616°C	0.9396
Random Forest	0.3338°C	0.4836°C	0.9552
XGBoost	0.3303°C	0.4763°C	0.9566
Best Model

XGBoost produced the best results among the evaluated models.

Test-set performance:

MAE: 0.3303°C
RMSE: 0.4763°C
R²: 0.9566

The trained model is stored at:

models/xgboost_weather_model.joblib
8. Feature Importance

Feature-importance analysis was performed using the trained XGBoost model.

The most influential features were:

Rank	Feature
1	Current Temperature
2	Temperature Lag 24 Hours
3	Temperature Lag 1 Hour
4	Hour
5	Temperature Lag 3 Hours
6	Relative Humidity
7	Precipitation

The results show that current temperature and recent historical temperature patterns are particularly important for predicting the next-hour temperature.

9. FastAPI Backend

The trained XGBoost model is exposed through a FastAPI REST API.

Endpoints
GET  /
GET  /health
POST /predict
Health Check
{
  "status": "healthy",
  "model": "XGBoost"
}
Prediction Response
{
  "predicted_temperature": 26.83,
  "unit": "°C"
}
API Documentation

FastAPI provides interactive Swagger documentation at:

http://127.0.0.1:8000/docs
10. Streamlit Dashboard

The Streamlit frontend provides an interactive interface for the weather intelligence system.

Dashboard Components
Historical temperature visualization
Monthly temperature analysis
Monthly precipitation analysis
Next-hour temperature prediction
AI-generated prediction explanation
Weather Intelligence Assistant
Retrieved RAG context

The dashboard communicates with the FastAPI backend for machine-learning predictions.

11. Retrieval-Augmented Generation

A lightweight Retrieval-Augmented Generation (RAG) system was implemented to provide project-grounded answers.

RAG Workflow
User Question
      |
      v
Project Knowledge Base
      |
      v
Relevant Information Retrieval
      |
      v
Retrieved Context
      |
      v
Gemma 3 4B
      |
      v
Grounded Response

The knowledge base contains information about:

Project overview
Dataset
EDA
Correlations
Machine-learning methodology
Model performance
Feature importance
FastAPI
Streamlit
GenAI integration
Project limitations

RAG components:

rag/
├── weather_knowledge.txt
├── retriever.py
└── generate_answer.py
12. Generative AI

The project uses:

Gemma 3 4B

through:

Ollama

The model runs locally and is used for:

Answering questions about the project.
Explaining machine-learning results.
Explaining model predictions.
Generating grounded natural-language responses.

The assistant is designed to avoid inventing project-specific statistics and to abstain when the required information is not available in the project knowledge base.

13. ML + GenAI Integration

The project combines machine-learning prediction with Generative AI explanation.

Weather Input
     |
     v
XGBoost
     |
     v
Next-Hour Temperature
     |
     v
Prediction Context
     |
     v
RAG + Gemma 3 4B
     |
     v
AI Explanation

The XGBoost model performs the numerical prediction, while Gemma 3 4B generates a natural-language explanation using the available project context.

This separates prediction from explanation and prevents the language model from being responsible for the numerical forecasting task.

14. Project Structure
Weather-Intelligence-Platform/
│
├── api/
│   └── main.py
│
├── app/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── Implementation_Plan.md
│   └── Project_Architecture.md
│
├── frontend/
│   └── dashboard.py
│
├── models/
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   └── xgboost_weather_model.joblib
│
├── notebooks/
│
├── rag/
│   ├── weather_knowledge.txt
│   ├── retriever.py
│   └── generate_answer.py
│
├── research/
│   ├── References/
│   ├── Research Papers/
│   └── Weather APIs/
│
├── screenshots/
│   ├── API/
│   ├── Dashboard/
│   ├── Dataset/
│   ├── EDA/
│   ├── Environment Setup/
│   ├── Models/
│   └── RAG/
│
├── src/
│   ├── download_weather.py
│   ├── inspect_dataset.py
│   ├── preprocess_weather.py
│   ├── eda_weather.py
│   ├── prepare_ml_data.py
│   ├── train_models.py
│   ├── feature_importance.py
│   └── test_prediction.py
│
├── .gitignore
├── README.md
└── requirements.txt
15. Installation
Clone the Repository
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Weather-Intelligence-Platform
Create a Virtual Environment
python -m venv venv
Activate the Environment

Windows PowerShell:

.\venv\Scripts\Activate.ps1
Install Dependencies
pip install -r requirements.txt
16. Running the Application
Start FastAPI

Run:

python -m uvicorn api.main:app --reload

The API will be available at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs
Start Streamlit

Open another terminal and run:

streamlit run frontend\dashboard.py
Start Ollama

Make sure Ollama is installed and Gemma 3 4B is available:

ollama pull gemma3:4b

The GenAI component communicates with the local Ollama service.

17. Testing and Validation

The project was tested across the major components.

Dataset
Missing-value checks
Duplicate checks
Date-range validation
Chronological-order validation
Machine Learning
Model comparison
MAE evaluation
RMSE evaluation
R² evaluation
Individual prediction testing
Dynamic prediction testing
FastAPI
Root endpoint
Health endpoint
Prediction endpoint
Swagger testing
RAG and GenAI
Project-related questions
Retrieved-context inspection
Model-performance questions
Feature-importance questions
Out-of-scope questions
ML prediction explanations
18. Example Prediction

One individual prediction test produced:

Current Temperature:       27.40°C
Actual Next-Hour Temperature: 26.90°C
Predicted Temperature:     26.86°C
Absolute Error:             0.04°C

This is an individual test example and is separate from the overall test-set MAE of 0.3303°C.

19. Limitations
The current ML task focuses on next-hour temperature prediction.
It does not represent a complete multi-variable weather forecasting system.
The model is trained using historical weather data for Kochi.
Prediction performance may vary during unusual or extreme weather conditions.
GenAI responses depend on the information available in the project knowledge base.
Gemma 3 4B is a relatively small local language model.
Model predictions are estimates and are not guaranteed future observations.
20. Future Scope

Future development could include:

Multi-location weather prediction
Longer-horizon forecasting
Prediction of additional weather variables
Real-time weather data integration
Weather-image and satellite-data analysis
Advanced time-series forecasting
Weather anomaly detection
Advanced RAG retrieval
Climate-risk analysis
Weather-based decision-support features
21. Technologies Used
Category	Technologies
Programming	Python
Data Processing	Pandas, NumPy
Visualization	Matplotlib, Seaborn, Plotly
Machine Learning	Scikit-learn, XGBoost
Backend	FastAPI, Uvicorn
Frontend	Streamlit
GenAI	Gemma 3 4B, Ollama
RAG	Custom lightweight retrieval pipeline
Development	VS Code, Git, GitHub
Data Source	Open-Meteo Historical Weather API
22. Conclusion

The Intelligent Multi-Modal Weather Intelligence and Climate Decision Support Platform demonstrates an end-to-end integration of data analysis, machine learning, API development, interactive visualization and Generative AI.

The evaluated models showed that XGBoost achieved the strongest performance, with an MAE of 0.3303°C, RMSE of 0.4763°C, and R² of 0.9566 on the chronological test set.

The RAG and GenAI components extend the system beyond numerical prediction by providing a natural-language interface for project information and model-prediction explanations.

The project provides a foundation for developing a broader weather intelligence and climate decision-support platform.