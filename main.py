# main.py
import os
import logging
from bot.core import StockAnalysisBot
from config import Config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    if not Config.TELEGRAM_TOKEN:
        print("Ошибка: TELEGRAM_TOKEN не найден!")
        exit(1)
    
    bot = StockAnalysisBot(Config.TELEGRAM_TOKEN)
    bot.run()