#!/usr/bin/env python3
import sys
import os

# Добавляем текущую директорию в путь Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import StockAnalysisBot
from config import Config

def main():
    # Проверяем наличие токена
    if not Config.TELEGRAM_TOKEN or Config.TELEGRAM_TOKEN == 'your_actual_bot_token_here':
        print("Ошибка: Токен бота не настроен!")
        print("Пожалуйста, откройте файл .env и установите ваш TELEGRAM_BOT_TOKEN")
        return
    
    print(" Запуск телеграм-бота для анализа акций...")
    print(" Функциональность:")
    print("   • Анализ акций по тикеру")
    print("   • Прогноз на 30 дней")
    print("   • Торговые рекомендации")
    print("   • Расчет прибыли")
    print("   • Логирование всех операций")
    
    bot = StockAnalysisBot(Config.TELEGRAM_TOKEN)
    bot.run()

if __name__ == "__main__":
    main()