import datetime as dt
import json
import logging
import os
import typing as tp

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class WeatherAPI:
    def __init__(self, api_key: str = None, config_file: str = None) -> None:
        """Initializing World Weather Online API with API key (directly or from a json-file)."""
        self.history_url = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx'
        self.forecast_url = 'http://api.worldweatheronline.com/premium/v1/weather.ashx'
        if api_key:
            self.api_key = api_key
        elif config_file:
            with open(config_file) as config_file:
                config = json.load(config_file)
            self.api_key = config['api_key']
        else:
            raise ValueError('No API-key has been founded.')
    
    def get_month_history(self, location: str, date: str, end_date: str = None) -> pd.DataFrame:
        """Getting historical data about the weather during given period."""
        params = {
            'key': self.api_key,
            'q': location,
            'date': date,
            'tp': 1,
            'format': 'json'
        }
        if end_date:
            params['enddate'] = end_date

        response = requests.get(self.history_url, params=params)
        if response.status_code != 200:
            response.raise_for_status()
        data = response.json()
        result = self.parse_response(data)
        
        return pd.DataFrame.from_records(result)

    def get_forecast(self, location: str, date: str) -> pd.DataFrame:
        current_date = dt.datetime.now()
        future_date = dt.datetime.strptime(date, '%Y-%m-%d')
        difference = (future_date - current_date).days
        if difference < 0:
            raise ValueError(f'The date {date} has already passed. Use get_history() instead.')

        params = {
            'key': self.api_key,
            'q': location,
            'num_of_days': difference,
            'tp': 1,
            'format': 'json'
        }

        response = requests.get(self.forecast_url, params=params)
        if response.status_code != 200:
            response.raise_for_status()
        data = response.json()
        result = self.parse_response(data)

        return pd.DataFrame.from_records(result)

    @staticmethod
    def parse_response(data: tp.Dict) -> tp.List[tp.Dict]:
        """Processing response for the dataset."""
        result = []

        for day in data['data']['weather']:
            for hour in day['hourly']:
                result.append({
                    'Date': day['date'],
                    'Hour': int(hour['time']) // 100,
                    'Temperature': float(hour['tempC']),
                    'Humidity': float(hour['humidity']),
                    'Wind speed': float(hour['windspeedKmph']),
                    'Visibility': float(hour['visibility']) * 100,
                    'Dew point temperature': float(hour['DewPointC']),
                    'Solar Radiation': round(float(hour['uvIndex']) * 3.52 / 6, 2),  # a simple approximation
                    # uv-index to MJ/m2 based on info from the dataset
                    'Rainfall': float(hour['precipMM']),
                    'Snowfall': 0.0,
                })
                if result[-1]['Temperature'] < 0.0:  # this api does not provide information about precipitation type
                    result[-1]['Rainfall'], result[-1]['Snowfall'] = result[-1]['Snowfall'], result[-1]['Rainfall'] / 10

        return result

    def get_history(self, location: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Getting history without limitations of API."""
        date_format = '%Y-%m-%d'
        start_date = dt.datetime.strptime(start_date, date_format)
        end_date = dt.datetime.strptime(end_date, date_format)

        frames = []

        while start_date < end_date:
            last_day = (start_date.replace(day=28) + dt.timedelta(days=4)).replace(
                day=1) - dt.timedelta(days=1)  # calculating the last day of the month
            if last_day > end_date:
                last_day = end_date

            try:
                df = self.get_month_history(location, start_date.strftime(date_format), last_day.strftime(date_format))
            except requests.HTTPError as e:
                logger.error(f'During the next request, an error {e} occurred. Returned data is not full.')
                break

            frames.append(df)
            start_date = last_day + dt.timedelta(days=1)

        result_df = pd.concat(frames, ignore_index=True)
        return result_df

    def cache_period(self, location: str, start_date: str, end_date: str, filename: str = 'cache.csv') -> None:
        """Caching weather data for the period to a csv-file."""
        file_exists = os.path.isfile(filename)
        result_df = self.get_history(location, start_date, end_date)
        result_df.to_csv(filename, mode='a', index=False, header=not file_exists)


if __name__ == '__main__':
    weather_api = WeatherAPI(config_file='weather_config.json')
    weather_api.cache_period('Seoul', '2018-12-01', '2024-02-10')
