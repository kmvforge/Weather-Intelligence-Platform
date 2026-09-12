import streamlit as st
import pandas as pd
import os
import plotly.express as px
import sys
import joblib
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


# ============================================================
# RAG IMPORTS
# ============================================================

from rag.generate_answer import generate_answer
from rag.retriever import retrieve_context


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Kochi Weather Intelligence",
    page_icon="🌦️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "kochi_weather_processed.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_weather_model.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

weather_model = joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_weather_data():

    df = pd.read_csv(DATA_PATH)

    df["time"] = pd.to_datetime(
        df["time"]
    )

    return df


df = load_weather_data()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🌦️ Kochi Weather Intelligence Platform"
)

st.markdown(
    """
    **AI-powered historical weather analysis and
    next-hour temperature prediction**
    """
)

st.divider()


# ============================================================
# SUMMARY METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "📊 Weather Records",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "📅 Data Period",
        "2015–2025"
    )


with col3:

    st.metric(
        "🌡️ Average Temperature",
        f"{df['temperature_2m'].mean():.2f} °C"
    )


with col4:

    st.metric(
        "🌧️ Total Precipitation",
        f"{df['precipitation'].sum():,.1f} mm"
    )


st.divider()


# ============================================================
# HISTORICAL TEMPERATURE
# ============================================================

st.header(
    "🌡️ Historical Temperature"
)


daily_temperature = (
    df.set_index("time")["temperature_2m"]
    .resample("D")
    .mean()
    .reset_index()
)


daily_temperature.columns = [
    "Date",
    "Temperature"
]


fig_temp = px.line(
    daily_temperature,
    x="Date",
    y="Temperature",
    title="Daily Average Temperature in Kochi",
    labels={
        "Date": "Date",
        "Temperature": "Temperature (°C)"
    }
)


fig_temp.update_layout(
    height=450,
    hovermode="x unified"
)


st.plotly_chart(
    fig_temp,
    use_container_width=True
)


# ============================================================
# MONTHLY TEMPERATURE
# ============================================================

st.header(
    "📈 Average Monthly Temperature"
)


monthly_temperature = (
    df.groupby("month")["temperature_2m"]
    .mean()
    .reset_index()
)


monthly_temperature.columns = [
    "Month",
    "Average Temperature (°C)"
]


fig_monthly_temp = px.line(
    monthly_temperature,
    x="Month",
    y="Average Temperature (°C)",
    markers=True,
    title="Average Monthly Temperature in Kochi",
    labels={
        "Month": "Month",
        "Average Temperature (°C)": "Temperature (°C)"
    }
)


fig_monthly_temp.update_layout(
    height=400,
    xaxis=dict(
        tickmode="linear",
        dtick=1
    )
)


st.plotly_chart(
    fig_monthly_temp,
    use_container_width=True
)


# ============================================================
# MONTHLY PRECIPITATION
# ============================================================

st.header(
    "🌧️ Average Monthly Precipitation"
)


monthly_rain = (
    df.groupby("month")["precipitation"]
    .mean()
    .reset_index()
)


monthly_rain.columns = [
    "Month",
    "Average Precipitation (mm)"
]


monthly_rain = monthly_rain.set_index(
    "Month"
)


st.bar_chart(
    monthly_rain,
    height=350
)


st.divider()


# ============================================================
# AI PREDICTION SECTION
# ============================================================

st.header(
    "🤖 AI Next-Hour Temperature Prediction"
)


st.write(
    "Enter weather conditions to obtain a next-hour "
    "temperature prediction from the XGBoost model."
)


# ============================================================
# CURRENT WEATHER INPUTS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    temperature = st.number_input(
        "Current Temperature (°C)",
        value=27.4
    )


    humidity = st.number_input(
        "Relative Humidity (%)",
        min_value=0,
        max_value=100,
        value=82
    )


    precipitation = st.number_input(
        "Precipitation (mm)",
        min_value=0.0,
        value=0.0
    )


    pressure = st.number_input(
        "Surface Pressure (hPa)",
        value=1010.0
    )


with col2:

    cloud_cover = st.number_input(
        "Cloud Cover (%)",
        min_value=0,
        max_value=100,
        value=50
    )


    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        value=10.0
    )


    wind_direction = st.number_input(
        "Wind Direction (°)",
        min_value=0.0,
        max_value=360.0,
        value=180.0
    )


    wind_gusts = st.number_input(
        "Wind Gusts (km/h)",
        min_value=0.0,
        value=15.0
    )


with col3:

    year = st.number_input(
        "Year",
        min_value=2015,
        max_value=2030,
        value=2025
    )


    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=1
    )


    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=1
    )


    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=0
    )


# ============================================================
# HISTORICAL FEATURES
# ============================================================

st.subheader(
    "Historical Features"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    temperature_lag_1 = st.number_input(
        "Temperature Lag 1h",
        value=27.6
    )


with col2:

    temperature_lag_3 = st.number_input(
        "Temperature Lag 3h",
        value=27.8
    )


with col3:

    temperature_lag_24 = st.number_input(
        "Temperature Lag 24h",
        value=25.9
    )


with col4:

    humidity_lag_1 = st.number_input(
        "Humidity Lag 1h",
        value=81.0
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Next-Hour Temperature",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # Calculate day of year
        # ----------------------------------------------------

        selected_date = pd.Timestamp(
            year=int(year),
            month=int(month),
            day=int(day)
        )

        day_of_year = selected_date.dayofyear


        # ----------------------------------------------------
        # Prepare model input
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "temperature_2m": temperature,

            "relative_humidity_2m": humidity,

            "precipitation": precipitation,

            "surface_pressure": pressure,

            "cloud_cover": cloud_cover,

            "wind_speed_10m": wind_speed,

            "wind_direction_10m": wind_direction,

            "wind_gusts_10m": wind_gusts,

            "year": int(year),

            "month": int(month),

            "day": int(day),

            "hour": int(hour),

            "day_of_year": int(day_of_year),

            "temperature_lag_1": temperature_lag_1,

            "temperature_lag_3": temperature_lag_3,

            "temperature_lag_24": temperature_lag_24,

            "humidity_lag_1": humidity_lag_1

        }])


        # ----------------------------------------------------
        # XGBoost prediction
        # ----------------------------------------------------

        prediction = weather_model.predict(
            input_data
        )[0]


        prediction = round(
            float(prediction),
            2
        )


        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.success(
            "Prediction generated successfully!"
        )


        st.metric(
            "🌡️ Predicted Next-Hour Temperature",
            f"{prediction:.2f} °C"
        )


        # ====================================================
        # GENAI PREDICTION EXPLANATION
        # ====================================================

        st.subheader(
            "🤖 AI Explanation"
        )


        prediction_question = (
            "What does the model predict for the next hour, "
            "and what information is relevant to this prediction?"
        )


        prediction_context = f"""

The XGBoost model predicted the next-hour temperature as
{prediction:.2f} °C.

Current input conditions:

- Current temperature: {temperature:.2f} °C
- Relative humidity: {humidity:.0f}%
- Precipitation: {precipitation:.2f} mm
- Surface pressure: {pressure:.2f} hPa
- Cloud cover: {cloud_cover:.0f}%
- Wind speed: {wind_speed:.2f} km/h
- Wind direction: {wind_direction:.0f}°
- Wind gusts: {wind_gusts:.2f} km/h

Historical temperature features:

- Temperature lag 1h: {temperature_lag_1:.2f} °C
- Temperature lag 3h: {temperature_lag_3:.2f} °C
- Temperature lag 24h: {temperature_lag_24:.2f} °C
- Humidity lag 1h: {humidity_lag_1:.0f}%

This is an XGBoost next-hour temperature prediction.
It is a model prediction, not a guaranteed future observation.

"""

        # ----------------------------------------------------
        # Generate GenAI explanation
        # ----------------------------------------------------

        with st.spinner(
            "Generating AI explanation..."
        ):

            try:

                explanation = generate_answer(
                    prediction_question,
                    additional_context=prediction_context
                )

                st.write(
                    explanation
                )

                with st.expander(
                    "🔎 View Prediction Context"
                ):

                    st.text(
                        prediction_context
                    )

            except Exception:

                st.info(
                    "📊 Model Interpretation"
                )

                st.write(
                    f"The XGBoost model predicts a next-hour "
                    f"temperature of **{prediction:.2f} °C** based on "
                    "the current weather conditions and historical "
                    "temperature features provided above."
                )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )
       
# ============================================================
# WEATHER INTELLIGENCE ASSISTANT
# ============================================================

st.divider()


st.header(
    "🤖 Weather Intelligence Assistant"
)


st.write(
    "Ask questions about the project's weather data, "
    "machine-learning models, predictions, and analysis."
)


question = st.text_input(
    "Ask a question:",
    placeholder="Example: Which model performed best?"
)


if st.button(
    "Ask AI"
):

    if question.strip():

        with st.spinner(
            "Retrieving information and generating answer..."
        ):

            try:

                context = retrieve_context(
                    question
                )

                answer = generate_answer(
                    question
                )

                st.success(
                    "Answer generated"
                )

                st.subheader(
                    "Answer"
                )

                st.write(
                    answer
                )

                with st.expander(
                    "🔎 View Retrieved Context"
                ):

                    if context:

                        st.text(
                            context
                        )

                    else:

                        st.info(
                            "No relevant project information "
                            "was retrieved."
                        )

            except Exception:

                st.info(
                    "📚 Project Knowledge Response"
                )

                if context:

                    st.write(
                        "Based on the project's documented knowledge:"
                    )

                    st.write(
                        context
                    )

                else:

                    st.write(
                        "I don't have enough information in the "
                        "project knowledge base to answer that."
                    )

    else:

        st.warning(
            "Please enter a question."
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Weather Intelligence Platform | "
    "Historical Analysis + Machine Learning + RAG"
)
