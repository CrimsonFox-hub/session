#!/usr/bin/env python3
import sys
import os

# Добавляем текущую директорию в путь Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import StockAnalysisBot
from config import Config

def main():
    
    print(f"Токен из Config: {' Установлен' if Config.TELEGRAM_TOKEN else ' Отсутствует'}")
    print(f"Токен из окружения: {' Установлен' if os.getenv('TELEGRAM_BOT_TOKEN') else ' Отсутствует'}")
    print(f"Текущая директория: {os.path.abspath(os.path.curdir)}")
    
    # Проверяем наличие .env файла
    env_file = '.env'
    if os.path.exists(env_file):
        print(f"Файл .env: Найден ({os.path.abspath(env_file)})")
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                if 'TELEGRAM_BOT_TOKEN' in content:
                    print("TELEGRAM_BOT_TOKEN в .env: Найден")
                else:
                    print("TELEGRAM_BOT_TOKEN в .env: Отсутствует")
        except Exception as e:
            print(f"Ошибка чтения .env: {e}")
    else:
        print(f"Файл .env: Не найден")
    
    
    print("\nТокен загружен. Запуск бота...")
    
    try:
        bot = StockAnalysisBot(Config.TELEGRAM_TOKEN)
        bot.run()
    except Exception as e:
        print(f"Ошибка при запуске бота: {e}")
        print("Проверьте правильность токена и доступ к Telegram API")

if __name__ == "__main__":
    main()