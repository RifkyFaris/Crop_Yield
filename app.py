import streamlit as st
import pickle
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


__crop = None
__data_columns = None
__model = None

def load_saved_artifacts():
    global __data_columns, __crop, __model
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    columns_path = os.path.join(BASE_DIR, "artifacts", "columns.json")
    model_path = os.path.join(BASE_DIR, "artifacts", "yield_prediction_model.pickle")

    with open(columns_path, "r") as f:
        __data_columns = json.load(f)['data_columns']
        __crop = __data_columns[12:18]  # adjust if needed

    with open(model_path, 'rb') as f:
        __model = pickle.load(f)
    


def get_rainfall_bin(rainfall):
    if rainfall < 300:
        return "Rainfall_Bin_Low"
    elif rainfall < 700:
        return "Rainfall_Bin_Medium"
    elif rainfall < 1000:
        return "Rainfall_Bin_High"
    else:
        return "Rainfall_Bin_Very High"

def get_temp_bin(temp):
    if temp < 15:
        return "Temp_Bin_Cold"
    elif temp < 25:
        return "Temp_Bin_Mild"
    elif temp < 30:
        return "Temp_Bin_Warm"
    else:
        return "Temp_Bin_Hot"

def get_harvest_bin(days):
    if days < 90:
        return "Harvest_Bin_Short"
    elif days < 120:
        return "Harvest_Bin_Medium"
    elif days < 150:
        return "Harvest_Bin_Long"
    else:
        return "Harvest_Bin_Very Long"


def get_estimated_yield(fertilizer, irrigation, region, soil, crop, weather,
                        rainfall, temperature, harvest_days):
    global __data_columns, __model
    x = np.zeros(len(__data_columns))

    
    x[__data_columns.index('Fertilizer_Used')] = int(fertilizer)
    x[__data_columns.index('Irrigation_Used')] = int(irrigation)

    
    features = [
        region,
        soil,
        crop,
        weather,
        get_rainfall_bin(rainfall),
        get_temp_bin(temperature),
        get_harvest_bin(harvest_days)
    ]

    for feature in features:
        if feature in __data_columns:
            x[__data_columns.index(feature)] = 1

    x_df = pd.DataFrame([x], columns=__data_columns)
    return round(__model.predict(x_df)[0], 2)

def get_crop_names():
    global __crop
    return __crop


load_saved_artifacts()

st.set_page_config(page_title="Crop Yield Prediction", layout="wide")
st.title("🌾 Crop Yield Prediction")



col1, col2 = st.columns(2)

with col1:
    fertilizer = st.radio("Fertilizer Used", ["Yes", "No"], index=None,)
    fertilizer_val = 1 if fertilizer == "Yes" else 0

with col2:
    irrigation = st.radio("Irrigation Used", ["Yes", "No"], index=None,)
    irrigation_val = 1 if irrigation == "Yes" else 0



with col1:
    crops = ["Select Crop"] + get_crop_names()
    crop = st.selectbox("Crop", crops)

    
    if crop == "Select Crop":
        crop_val = None
    else:
        crop_val = crop

with col2:
    regions = ["Select Region","Region_North", "Region_South", "Region_East", "Region_West"]
    region = st.selectbox("Select Region", regions)
    if region == "Select Region":
        region_val = None
    else:
        region_val = region


with col1:
    soil_types = ["Select Soil Type","Soil_Type_Chalky", "Soil_Type_Clay", "Soil_Type_Loam","Soil_Type_Peaty", "Soil_Type_Sandy", "Soil_Type_Silt"]
    soil = st.selectbox("Select Soil Type", soil_types)
    if soil == "Select Soil Type":
        soil_val = None
    else:
        soil_val = soil
   

with col2:
    weathers = ["Select Weather","Weather_Condition_Sunny", "Weather_Condition_Rainy","Weather_Condition_Cloudy", "Weather_Condition_Stormy"]
    weather = st.selectbox("Select Weather Condition", weathers)
    if weather == "Select Weather":
        weather_val = None
    else:
        weather_val = weather


col3, col4,col5 = st.columns(3)

with col3:
   rainfall_input = st.text_input("Rainfall (mm)", placeholder="Enter rainfall in mm")
   rainfall = float(rainfall_input) if rainfall_input else None


with col4:
    
    temperature_input = st.text_input("Temperature (°C)", placeholder="Enter temperature")
    temperature = float(temperature_input) if temperature_input else None
with col5:
    
    harvest_days_input = st.text_input("Days to Harvest", placeholder="Enter no of days")
    harvest_days = float(harvest_days_input) if harvest_days_input else None



if st.button("Estimate Yield"):
    if fertilizer is None:
        st.error("❌ Please select Fertilizer Used")
    elif irrigation is None:
        st.error("❌ Please select Irrigation Used")
    elif crop == "Select Crop" or crop is None:
        st.error("❌ Please select a Crop")
    elif region_val is None or region_val.strip() == "":
        st.error("❌ Please select a Region")
    elif soil_val is None or soil_val.strip() == "":
        st.error("❌ Please select a Soil Type")
    elif weather_val is None or weather_val.strip() == "":
        st.error("❌ Please select a Weather condition")
    elif rainfall is None or rainfall <= 0:
        st.error("❌ Please enter valid Rainfall (mm)")
    elif temperature is None:
        st.error("❌ Please enter valid Temperature (°C)")
    elif harvest_days is None or harvest_days <= 0:
        st.error("❌ Please enter valid Days to Harvest")
    else:
        try:
            estimated_yield = get_estimated_yield(
                fertilizer=fertilizer_val,
                irrigation=irrigation_val,
                region=region,
                soil=soil,
                crop=crop,
                weather=weather,
                rainfall=rainfall,
                temperature=temperature,
                harvest_days=harvest_days
            )
            st.success(f"✅ Estimated Yield: {estimated_yield} tons/hectare")

            
            fert_options = [1, 0]
            fert_labels = ["Yes", "No"]
            fert_yields = [
                get_estimated_yield(fert, irrigation_val, region, soil, crop, weather, rainfall, temperature, harvest_days)
                for fert in fert_options
            ]
            fig1 = go.Figure(data=[go.Bar(x=fert_labels, y=fert_yields, marker_color='green')])
            fig1.update_layout(title=f"Fertilizer vs Yield for {crop}", yaxis_title="Yield (tons/hectare)")
            st.plotly_chart(fig1, use_container_width=True)

            
            rainfall_range = [50, 100, 200, 300, 400, 500]
            rain_yields = [
                get_estimated_yield(fertilizer_val, irrigation_val, region, soil, crop, weather, rf, temperature, harvest_days)
                for rf in rainfall_range
            ]
            fig2 = go.Figure(data=[go.Scatter(x=rainfall_range, y=rain_yields, mode='lines+markers')])
            fig2.update_layout(title=f"Rainfall vs Yield for {crop}", xaxis_title="Rainfall (mm)", yaxis_title="Yield (tons/hectare)")
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Error estimating yield: {str(e)}")

