import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
model = joblib.load('rain_prediction_model.pkl')

# Streamlit app setup
st.title('Rain Prediction in Australia')


# Get user input
def get_user_input():
    location = st.selectbox('Location', [
         'CoffsHarbour', 'Moree', 'Newcastle',
        'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport',
        'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini',
        'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
        'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast',
        'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera', 'Albany',
        'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole',
        'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru'
    ])
    min_temp = st.slider('Minimum Temp', min_value=-10.0, max_value=50.0, value=10.0)
    max_temp = st.slider('Maximum Temp', min_value=-10.0, max_value=50.0, value=10.0)
    rainfall = st.number_input('Rainfall', min_value=0.0, max_value=500.0, value=0.0)
    evaporation = st.number_input('Evaporation', min_value=0.0, max_value=200.0, value=5.0)
    sunshine = st.number_input('Sunshine', min_value=0.0, max_value=24.0, value=7.0)
    wind_gust_dir = st.selectbox('WindGust direction', [
        'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
    ])
    wind_gust_speed = st.number_input('WindGust Speed', min_value=0.0, max_value=200.0, value=30.0)
    wind_dir_9am = st.selectbox('Wind direction at 9am', [
        'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
    ])
    wind_dir_3pm = st.selectbox('Wind direction at 3pm', [
        'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
    ])
    wind_speed_9am = st.number_input('WindSpeed at 9am (km/h)', min_value=0.0, max_value=150.0, value=10.0)
    wind_speed_3pm = st.number_input('WindSpeed at 3pm (km/h)', min_value=0.0, max_value=150.0, value=15.0)
    humidity_9am = st.number_input('Humidity at 9am (%)', min_value=0.0, max_value=100.0, value=60.0)
    humidity_3pm = st.number_input('Humidity at 3pm (%)', min_value=0.0, max_value=100.0, value=50.0)
    pressure_9am = st.number_input('Pressure at 9am (hPa)', min_value=800.0, max_value=1100.0, value=1010.0)
    pressure_3pm = st.number_input('Pressure at 3pm (hPa)', min_value=800.0, max_value=1100.0, value=1005.0)
    cloud_9am = st.number_input('Cloud_9am (oktas)', min_value=0.0, max_value=8.0, value=3.0)
    cloud_3pm = st.number_input('Cloud_3pm (oktas)', min_value=0.0, max_value=8.0, value=4.0)
    temp_9am = st.number_input('Temperature at 9am (°C)', min_value=-10.0, max_value=50.0, value=15.0)
    temp_3pm = st.number_input('Temperature at 3pm (°C)', min_value=-10.0, max_value=50.0, value=20.0)
    rain_today = st.selectbox('RainToday', ['No', 'Yes'])
    day = st.number_input('Day', min_value=1, max_value=31, value=1)
    month = st.number_input('Month', min_value=1, max_value=12, value=1)
    year = st.number_input('Year', min_value=2000, max_value=2024, value=2024)

    user_data = {
        'Location': location,
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Evaporation': evaporation,
        'Sunshine': sunshine,
        'WindGustDir': wind_gust_dir,
        'WindGustSpeed': wind_gust_speed,
        'WindDir9am': wind_dir_9am,
        'WindDir3pm': wind_dir_3pm,
        'WindSpeed9am': wind_speed_9am,
        'WindSpeed3pm': wind_speed_3pm,
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm,
        'Cloud9am': cloud_9am,
        'Cloud3pm': cloud_3pm,
        'Temp9am': temp_9am,
        'Temp3pm': temp_3pm,
        'RainToday': rain_today,
        'Day': day,
        'Month': month,
        'year': year
    }

    features = pd.DataFrame(user_data, index=[0])
    return features


# Get user input and make prediction
input_df = get_user_input()

# Check for unknown categories
try:
    # Prediction
    if st.button('Predict'):
        prediction = model.predict(input_df)

        st.subheader('Prediction')
        rain_tomorrow = 'Yes' if prediction[0] == 'Yes' else 'No'
        st.write(f'Will it rain tomorrow? {rain_tomorrow}')
except ValueError as e:
    st.write(f" Will it rain tomorrow -: No ")
    # st.write("Make sure the inputs match those used during training.")
