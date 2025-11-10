import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not TELEGRAM_TOKEN:
        print(" ошибка: TELEGRAM_BOT_TOKEN не найден в .env файле!")
        
    LOG_FILE = 'trading_logs.txt'
    TRAINING_PERIOD = '2y'
    FORECAST_DAYS = 30
    TEST_SIZE = 0.2
    RIDGE_ALPHA = 1.0
    RF_ESTIMATORS = 100
    ARIMA_ORDER = (5, 1, 0)
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32