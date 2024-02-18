from datetime import datetime

import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

from model import SeoulBikeRentingModel


@st.cache_data
def load_data():
    return pd.read_csv('SeoulBikeData.csv')


@st.cache_resource
def initialize_model(df):
    cat_columns = ['Seasons', 'Holiday', 'Functioning Day']
    regressor = CatBoostRegressor(random_seed=42, cat_features=cat_columns, silent=True)
    model = SeoulBikeRentingModel(regressor, df)
    model.fit()
    return model


df = load_data()
model = initialize_model(df)

st.title('Аренда велосипедов в Сеуле')

date = st.date_input('Выберите дату', datetime(2018, 12, 1))

if st.button('Предсказать'):
    predictions = model.predict(date.strftime('%Y-%m-%d'))
    st.line_chart(predictions.set_index('Hour')['Rented Bike Count'])

weather_data = model.df[(model.df['Date'].dt.date == date)]
if not weather_data.empty:
    st.write('Информация о погоде в выбранный день:')
    weather_parameters = ['Temperature', 'Humidity', 'Wind speed', 'Visibility', 'Dew point temperature',
                          'Solar Radiation', 'Rainfall', 'Snowfall']
    selected_parameter = st.selectbox('Выберите параметр погоды для отображения', weather_parameters)
    st.line_chart(weather_data.set_index('Hour')[selected_parameter])
else:
    st.write('Информации о погоде в выбранный день не найдено.')
