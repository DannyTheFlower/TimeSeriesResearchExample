import datetime as dt
import logging
import os.path
import typing as tp

import pandas as pd
from catboost import CatBoostRegressor
from workalendar.asia import SouthKorea

from weather import WeatherAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class SeoulBikeRentingModel:
    def __init__(self, model: tp.Any, data: pd.DataFrame) -> None:
        """Initializing model and dataframe."""
        self.model = model
        self.df = data
        self.preprocess()
        self.last_date = self.df['Date'].max()
        logger.info(f'Model has been initialized. Max datetime is {self.last_date}.')

    def preprocess(self) -> None:
        """First preprocessing: adding new features."""
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y') + pd.to_timedelta(self.df['Hour'],
                                                                                               unit='hours')
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Week'] = self.df['Date'].dt.isocalendar().week

        n_lags = 3
        for lag in range(1, n_lags + 1):
            self.df[f'lag_{lag}'] = self.df['Rented Bike Count'].shift(lag)

        self.df.dropna(inplace=True)
        logger.info('First preprocessing has been done.')

    def add_data(self, from_csv: str = None, from_api: WeatherAPI = None, last_date: str = None) -> None:
        """Adding new rows if necessary and preprocessing."""
        def get_season(date):
            month = date.month
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'

        date_format = '%Y-%m-%d'
        cal = SouthKorea()
        city = 'Seoul'

        if from_csv:
            new_df = pd.read_csv(from_csv)
        elif from_api and last_date:
            if dt.datetime.strptime(last_date, date_format) <= dt.datetime.now():
                new_df = from_api.get_history(city, dt.datetime.strftime(self.last_date, date_format), last_date)
            else:
                new_df_1 = from_api.get_history(city, dt.datetime.strftime(self.last_date, date_format),
                                                dt.datetime.strftime(dt.datetime.now(), date_format))
                new_df_2 = from_api.get_forecast(city, last_date)
                new_df = pd.concat([new_df_1, new_df_2], ignore_index=True)
        else:
            raise ValueError('Not enough data. You must provide csv-file or WeatherAPI instance with date.')

        new_df['Date'] = pd.to_datetime(new_df['Date'], format=date_format) + pd.to_timedelta(new_df['Hour'],
                                                                                              unit='hours')
        new_df['Seasons'] = new_df['Date'].apply(get_season)
        new_df['Holiday'] = new_df['Date'].apply(lambda x: 'Holiday' if cal.is_holiday(x) else 'No Holiday')
        new_df['Functioning Day'] = 'Yes'
        new_df['Year'] = new_df['Date'].dt.year
        new_df['Month'] = new_df['Date'].dt.month
        new_df['Week'] = new_df['Date'].dt.isocalendar().week

        new_df = new_df[new_df['Date'] > self.df['Date'].max()]

        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def fit(self, target_col: str = 'Rented Bike Count', exclude: tp.List[str] = None) -> None:
        """Fitting the model."""
        if exclude is None:
            exclude = ['Date']
        X, y = self.df.drop(columns=exclude + [target_col]), self.df[target_col]
        self.model.fit(X, y)
        logger.info('Model has been fitted.')

    def predict(self, date: str, cachefile: str = 'cache.csv', config_file: str = 'weather_config.json'):
        """Making prediction for the date."""
        date_format = '%Y-%m-%d'
        dt_date = dt.datetime.strptime(date, date_format) + dt.timedelta(hours=23)

        if dt_date < self.df['Date'].min():
            raise ValueError('Prediction date is before the first known date.')
        if dt_date <= self.last_date:
            return self.df[self.df['Date'].dt.date == dt_date.date()][['Hour', 'Rented Bike Count']]

        if os.path.isfile(cachefile):
            self.add_data(from_csv=cachefile)
        if self.df['Date'].max() < dt_date:
            weather = WeatherAPI(config_file=config_file)
            self.add_data(from_api=weather, last_date=date)
        logger.info('Data has been added.')

        start_index = self.df[self.df['Date'] > self.last_date].index.min()
        end_index = self.df[self.df['Date'] <= dt_date].index.max()

        for i in range(start_index, end_index + 1):
            for lag in range(1, 4):
                if i - lag >= 0:
                    self.df.at[i, f'lag_{lag}'] = self.df.iloc[i - lag]['Rented Bike Count']
                else:
                    self.df.at[i, f'lag_{lag}'] = 0
            X_pred = self.df.iloc[i].drop(['Rented Bike Count', 'Date'])
            y_pred = self.model.predict(X_pred)
            self.df.at[i, 'Rented Bike Count'] = round(y_pred)

        self.last_date = dt_date
        predictions = self.df[self.df['Date'].dt.date == dt_date.date()]
        return predictions[['Hour', 'Rented Bike Count']]


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    reg = CatBoostRegressor(random_seed=42, cat_features=['Seasons', 'Holiday', 'Functioning Day'], silent=True)
    data = pd.read_csv('SeoulBikeData.csv')
    model = SeoulBikeRentingModel(reg, data)

    model.fit()
    print(model.predict('2018-12-01'))
    print(model.df[(model.df['Date'].dt.date >= dt.datetime(2018, 11, 30).date()) & (model.df['Date'].dt.date <=
                   dt.datetime(2018, 12, 2).date())])
