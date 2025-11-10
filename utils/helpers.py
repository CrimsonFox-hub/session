# utils/helpers.py
import pandas as pd
import numpy as np

class CurrencyConverter:
    """Конвертер валют"""
    
    def __init__(self, usd_to_rub_rate):
        self.usd_to_rub_rate = usd_to_rub_rate
    
    def to_rub(self, usd_amount):
        return usd_amount * self.usd_to_rub_rate
    
    def to_usd(self, rub_amount):
        return rub_amount / self.usd_to_rub_rate

class FeatureEngineer:
    """признаки для временных рядов"""
    
    @staticmethod
    def create_features(data, window=30):
        df = data.copy()
        df['Price'] = df['Close']
        
        for i in range(1, window + 1):
            df[f'Lag_{i}'] = df['Price'].shift(i)
        
        # Технические индикаторы на исторических данных
        df['SMA_10'] = df['Price'].shift(1).rolling(window=10).mean()
        df['SMA_30'] = df['Price'].shift(1).rolling(window=30).mean()
        df['EMA_12'] = df['Price'].shift(1).ewm(span=12).mean()
        df['EMA_26'] = df['Price'].shift(1).ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['RSI'] = FeatureEngineer.calculate_rsi(df['Price'].shift(1))
        df['Volatility'] = df['Price'].shift(1).rolling(window=20).std()
        
        # Добавляем сезонные признаки
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        return df.dropna()
    
    @staticmethod
    def calculate_rsi(prices, window=14):
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi