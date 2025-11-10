# utils/data_loader.py
import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Загрузка данных об акциях"""
    
    @staticmethod
    def download_stock_data(ticker, period='2y'):
        try:
            print(f" Загружаем данные для тикера: {ticker}")
            
            if ticker.endswith('.ME'):
                stock = yf.Ticker(ticker)
            else:
                stock = yf.Ticker(ticker)
                
            data = stock.history(period=period)
            print(f"Получено строк данных: {len(data)}")
            
            if data.empty:
                print(f"Данные пустые для тикера: {ticker}")
                # Пробуем альтернативный период
                print(" Пробуем загрузить данные за 1 год...")
                data = stock.history(period='1y')
                if data.empty:
                    print(" Не удалось загрузить данные даже за 1 год")
                    return None
            
            print(f" Успешно загружены данные для: {ticker}")
            return data
            
        except Exception as e:
            print(f" Ошибка загрузки для {ticker}: {e}")
            return None
    
    @staticmethod
    def get_current_usd_rub_rate():
        """Получение текущего курса USD/RUB"""
        try:
            rate_data = yf.download("USDRUB=X", period="1d", progress=False)
            if not rate_data.empty:
                rate = float(rate_data['Close'].iloc[-1])
                print(f"💰 Текущий курс USD/RUB: {rate:.2f}")
                return rate
        except Exception as e:
            print(f" Не удалось получить курс USD/RUB: {e}")
        
        fallback_rate = 90.0
        print(f" Используем курс по умолчанию: {fallback_rate} RUB/USD")
        return fallback_rate
    
    @staticmethod
    def validate_ticker(ticker):
        """Проверка валидности тикера"""
        popular_tickers = {
            # Американские акции
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft',
            'GOOGL': 'Alphabet (Google)',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta Platforms',
            'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan Chase',
            'JNJ': 'Johnson & Johnson',
            'V': 'Visa',
            'WMT': 'Walmart',
            'PG': 'Procter & Gamble',
            'DIS': 'Disney',
            'NFLX': 'Netflix',
            'ADBE': 'Adobe',
            'PYPL': 'PayPal',
            'INTC': 'Intel',
            'CSCO': 'Cisco',
            'PFE': 'Pfizer',
            'XOM': 'Exxon Mobil',
            
            # ETF и индексы
            'SPY': 'S&P 500 ETF',
            'QQQ': 'Nasdaq 100 ETF',
            'VOO': 'Vanguard S&P 500 ETF',
            'IVV': 'iShares Core S&P 500 ETF'
        }
        
        return ticker in popular_tickers, popular_tickers.get(ticker, "Неизвестная компания")