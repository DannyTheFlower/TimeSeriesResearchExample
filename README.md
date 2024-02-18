# Short description
As part of the training practice at ITMO University, the task of forecasting the demand for bicycle rentals based on weather conditions was taken for analysis. This repository provides an example of how this problem can be solved.

# Content
This repository contains 3 py-files:
* _weather.py_ — `WeatherAPI` class which provides methods for getting historical weather data and weather forecast from [World Weather API](https://www.worldweatheronline.com/).
* _model.py_ — `SeoulBikeRentingModel` class which combines data, it's processing, and ML-model.
* _app.py_ — Streamlit-based web application for demonstrating the performance of the model.

and 2 csv-files:
* _SeoulBikeData.csv_ — main data from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/seoul-bike-sharing-demand-prediction/) for training the model.
* _cache.csv_ — the result of parsing weather in Seoul from 01/12/2018 to 10/02/2024.

# Quick start
Clone this repo, install packages from _requirements.txt_ and enter the command `streamlit run app.py` from the folder in your console.
