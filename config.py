import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    LOG_FILE = 'trading_logs.txt'
    
    # Настройки моделей
    TRAINING_PERIOD = '2y'  # Период данных для обучения
    FORECAST_DAYS = 30      # Дней для прогноза
    TEST_SIZE = 0.2         # Доля тестовой выборки
    
    # Параметры моделей
    RIDGE_ALPHA = 1.0
    RF_ESTIMATORS = 100
    ARIMA_ORDER = (5, 1, 0)
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32