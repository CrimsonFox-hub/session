# bot/core.py
import logging
import signal
import sys
import asyncio
import numpy as np
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

from .models import ModelManager, ModelTrainer
from .trading import TradingEngine
from .visualization import ChartBuilder
from utils.data_loader import DataLoader
from utils.helpers import CurrencyConverter, FeatureEngineer

logger = logging.getLogger(__name__)

class StockAnalysisBot:
    def __init__(self, telegram_token):
        self.telegram_token = telegram_token
        self.user_sessions = {}
        
        # Инициализация компонентов
        self.data_loader = DataLoader()
        self.usd_to_rub_rate = self.data_loader.get_current_usd_rub_rate()
        self.converter = CurrencyConverter(self.usd_to_rub_rate)
        self.model_manager = ModelManager()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(self.model_manager, self.feature_engineer)
        self.trading_engine = TradingEngine(self.converter)
        self.chart_builder = ChartBuilder(self.trading_engine)
        
        self.application = None

    async def start(self, update: Update, context: CallbackContext):
        """Обработчик команды /start"""
        welcome_text = """
🤖 Добро пожаловать в бот для анализа акций!

🎯 **Полностью интегрированная версия!**

Я могу:
•  Анализировать акции по тикеру (AAPL, TSLA, GOOGL, etc.)
•  Использовать 4 модели машинного обучения
•  Строить реалистичные прогнозы на 30 дней
•  Находить оптимальные точки для покупки и продажи
•  Рассчитать потенциальную прибыль в рублях

**Доступные команды:**
/start - показать это сообщение
/tickers - показать примеры тикеров
/help - помощь

**Для анализа отправьте в формате:**
`ТИКЕР СУММА`

💡 Можно использовать любую сумму от 100 рублей!

**Примеры:**
`AAPL 1000` - 1 тысяча рублей
`TSLA 50000` - 50 тысяч рублей
`MSFT 100000` - 100 тысяч рублей

💰 Текущий курс: {:.2f} RUB/USD
        """.format(self.usd_to_rub_rate)
        await update.message.reply_text(welcome_text, parse_mode='Markdown')

    async def tickers_command(self, update: Update, context: CallbackContext):
        """Обработчик команды /tickers"""
        tickers_info = """
    📊 **Доступные тикеры для анализа:**

    **🇺🇸 Американские акции:**
    • `AAPL` - Apple Inc.
    • `MSFT` - Microsoft
    • `GOOGL` - Alphabet (Google)
    • `AMZN` - Amazon
    • `TSLA` - Tesla
    • `META` - Meta Platforms
    • `NVDA` - NVIDIA
    • `JPM` - JPMorgan Chase
    • `JNJ` - Johnson & Johnson
    • `V` - Visa
    • `WMT` - Walmart
    • `PG` - Procter & Gamble
    • `DIS` - Disney
    • `NFLX` - Netflix
    • `ADBE` - Adobe
    • `PYPL` - PayPal
    • `INTC` - Intel

    **💎 ETF и индексы:**
    • `SPY` - S&P 500 ETF
    • `QQQ` - Nasdaq 100 ETF
    • `VOO` - Vanguard S&P 500 ETF
    • `IVV` - iShares Core S&P 500 ETF

    **💡 Советы:**
    • Все суммы указываются в рублях
    • Минимальная сумма: 100 рублей
    • Прогноз строится на 30 дней

    **Пример использования:**
    `AAPL 50000` - анализ Apple с инвестицией 50,000 рублей
    `SIBN.ME 100000` - анализ Газпрома с инвестицией 100,000 рублей
        """
        await update.message.reply_text(tickers_info, parse_mode='Markdown')

    async def help_command(self, update: Update, context: CallbackContext):
        """Обработчик команды /help"""
        help_text = """
🆘 **Помощь по использованию бота**

**Основные команды:**
/start - начать работу с ботом
/tickers - показать доступные тикеры
/help - эта справка

**Как использовать:**
1. Выберите тикер компании из списка /tickers
2. Отправьте сообщение в формате: `ТИКЕР СУММА`
3. Дождитесь анализа (обычно 1-3 минуты)
4. Получите подробный отчет с прогнозом

**Примеры запросов:**
`AAPL 10000` - анализ Apple с 10,000 рублей
`TSLA 50000` - анализ Tesla с 50,000 рублей
`SBER.ME 100000` - анализ Сбербанка с 100,000 рублей

**Что вы получите:**
•  График с прогнозом на 30 дней
•  Рекомендации по покупке/продаже
•  Расчет потенциальной прибыли
•  Информацию о лучшей модели

**Важно:**
• Это учебный проект, не используйте для реальных инвестиций
• Прогнозы основаны на исторических данных
• Рынки волатильны, прошлые результаты не гарантируют будущие
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def handle_message(self, update: Update, context: CallbackContext):
        """Обработка сообщений пользователя"""
        user_id = update.effective_user.id
        text = update.message.text.strip().upper()
        
        try:
            parts = text.split()
            if len(parts) < 2:
                await update.message.reply_text(
                    "Пожалуйста, укажите тикер и сумму инвестиции в рублях.\n"
                    "Например: `AAPL 1000`\n\n"
                    "Используйте /tickers чтобы посмотреть доступные тикеры", 
                    parse_mode='Markdown'
                )
                return
            
            ticker = parts[0]
            investment_rub = float(parts[1])
            
            # Валидация тикера
            is_valid, company_name = self.data_loader.validate_ticker(ticker)
            if not is_valid:
                await update.message.reply_text(
                    f" Тикер `{ticker}` не найден в списке поддерживаемых.\n"
                    f"Используйте /tickers чтобы посмотреть доступные тикеры.\n\n"
                    f" Попробуйте популярные тикеры: AAPL, TSLA, MSFT, GOOGL",
                    parse_mode='Markdown'
                )
                return
            
            if investment_rub <= 0:
                await update.message.reply_text("Сумма инвестиции должна быть положительной.")
                return
            
            if investment_rub < 100:
                await update.message.reply_text("Минимальная сумма инвестиции: 100 рублей")
                return
            
            wait_message = await update.message.reply_text(
                f" Анализирую {ticker} ({company_name})...\n"
                f" Инвестиция: {investment_rub:.0f}₽\n"
                f" Загружаю данные..."
            )
            
            data = self.data_loader.download_stock_data(ticker)
            if data is None or data.empty:
                await update.message.reply_text(
                    f" Не удалось загрузить данные по тикеру {ticker}\n"
                    f" Возможные причины:\n"
                    f"• Тикер указан неверно\n"
                    f"• Нет данных за выбранный период\n"
                    f"• Проблемы с подключением к бирже\n\n"
                    f"Попробуйте другой тикер из списка /tickers"
                )
                await wait_message.delete()
                return
            
            # Проверяем достаточно ли данных
            if len(data) < 30:
                await update.message.reply_text(
                    f" Недостаточно данных для анализа: всего {len(data)} дней\n"
                    f" Нужно минимум 30 дней исторических данных\n"
                    f"Попробуйте другой тикер"
                )
                await wait_message.delete()
                return

            
            # Обучение или загрузка моделей
            best_model_data, best_model_name, model_info = self.model_trainer.train_or_load_models(ticker, data)
            
            if best_model_data is None or best_model_data.get('model') is None:
                await update.message.reply_text(" Не удалось обучить или загрузить модели для анализа.")
                await wait_message.delete()
                return
            
            await wait_message.edit_text(
                f" Анализирую {ticker}...\n"
                f" Инвестиция: {investment_rub:.0f}₽\n"
                f" Модель: {best_model_name}\n"
                f" Строю прогноз на 30 дней..."
            )
            
            # Генерация прогноза
            forecast = self.model_trainer.generate_forecast(best_model_data, best_model_name, data)
            
            await wait_message.edit_text(
                f" Анализирую {ticker}...\n"
                f" Инвестиция: {investment_rub:.0f}₽\n"
                f" Модель: {best_model_name}\n"
                f" Прогноз готов\n"
                f" Ищу торговые точки..."
            )
            
            # Поиск торговых точек и расчет прибыли
            buy_points, sell_points, method_used = self.trading_engine.find_trading_points(forecast)
            profit_rub, profit_percentage, transactions = self.trading_engine.calculate_profit(
                forecast, buy_points, sell_points, investment_rub
            )
            
            current_price = data['Close'].iloc[-1]
            forecast_price = forecast[-1] if forecast else current_price
            price_change = forecast_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            current_price_rub = self.converter.to_rub(current_price)
            forecast_price_rub = self.converter.to_rub(forecast_price)
            price_change_rub = self.converter.to_rub(price_change)
            
            # Анализ качества прогноза
            forecast_quality = " Прогноз: "
            if abs(price_change_percent) < 2:
                forecast_quality += "нейтральный"
            elif price_change_percent > 5:
                forecast_quality += " сильный рост"
            elif price_change_percent > 2:
                forecast_quality += " умеренный рост"
            elif price_change_percent < -5:
                forecast_quality += " сильное падение"
            else:
                forecast_quality += " умеренное падение"
            
            # Основная информация
            main_response = f"""
 **Анализ акций {ticker}**

{model_info}
{forecast_quality}
            
 **Текущая цена:** ${current_price:.2f} ({current_price_rub:.0f}₽)
 **Прогноз через 30 дней:** ${forecast_price:.2f} ({forecast_price_rub:.0f}₽)
 **Изменение:** {price_change_percent:+.2f}% ({price_change_rub:+.0f}₽)

 **Инвестиция:** {investment_rub:.0f}₽
 **Потенциальная прибыль:** {profit_rub:+.0f}₽ ({profit_percentage:+.2f}%)

 **Торговые точки:** {method_used}
 Покупок: {len(buy_points)} | Продаж: {len(sell_points)}
"""

            # Детали операций
            operations_response = ""
            if transactions:
                operations_response = "** Детали торговых операций:**\n\n"
                for transaction in transactions:
                    operations_response += f"• {transaction}\n"
            else:
                operations_response = "**Рекомендация:** Держательная стратегия - ожидать лучших условий\n"

            # Создаем и отправляем график
            plot_buf = self.chart_builder.create_plot(data, forecast, ticker)
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=plot_buf,
                caption=f" Прогноз цен акций {ticker} на 30 дней\n🔺 Покупка | 🔻 Продажа"
            )
            
            # Отправляем основную информацию
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=main_response,
                parse_mode='Markdown'
            )
            
            # Отправляем детали операций
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=operations_response,
                parse_mode='Markdown'
            )
            
            await wait_message.delete()
            
            # Логирование
            metric_value = "0.00"
            try:
                if "RMSE: " in model_info:
                    metric_value = model_info.split("RMSE: ")[1].split(",")[0]
            except:
                metric_value = best_model_data.get('metrics', {}).get('rmse', "0.00")
            
            self.log_session(user_id, ticker, investment_rub, best_model_name, metric_value, profit_rub)
            
        except ValueError:
            await update.message.reply_text(" Неверный формат суммы. Укажите число после тикера.")
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            await update.message.reply_text(" Произошла ошибка при обработке запроса. Попробуйте позже.")

    async def error_handler(self, update: Update, context: CallbackContext):
        """Обработчик ошибок"""
        logger.error(f"Ошибка: {context.error}")

    def log_session(self, user_id, ticker, investment_rub, best_model, metric, profit_rub):
        """Логирование сессии"""
        log_entry = {
            'user_id': user_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': ticker,
            'investment_rub': investment_rub,
            'best_model': best_model,
            'metric': metric,
            'profit_rub': profit_rub
        }
        
        log_line = (f"{log_entry['timestamp']} | User: {log_entry['user_id']} | "
                   f"Ticker: {log_entry['ticker']} | Investment: {log_entry['investment_rub']:.0f}₽ | "
                   f"Model: {log_entry['best_model']} | Metric: {log_entry['metric']} | "
                   f"Profit: {log_entry['profit_rub']:.0f}₽\n")
        
        with open('trading_logs.txt', 'a', encoding='utf-8') as f:
            f.write(log_line)
        
        logger.info(f"Запись в лог: {log_line.strip()}")

    def setup_signal_handlers(self):
        """Настройка обработчиков сигналов для graceful shutdown"""
        def signal_handler(sig, frame):
            print(f"\n Получен сигнал {sig}. Завершаем работу...")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def shutdown(self):
        """Корректное завершение работы бота"""
        print(" Останавливаем бота...")
        if self.application:
            await self.application.stop()
            await self.application.shutdown()
        print(" Бот успешно остановлен")
        sys.exit(0)

    def run(self):
        """Запуск бота с обработкой ошибок"""
        try:
            self.setup_signal_handlers()
            
            print(" Запускаем бота...")
            self.application = Application.builder().token(self.telegram_token).build()
            
            # Регистрируем обработчики команд
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("tickers", self.tickers_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            self.application.add_error_handler(self.error_handler)
            
            logger.info("Бот запущен...")
            print(" Бот для анализа акций запущен!")
            print(f" Курс USD/RUB: {self.usd_to_rub_rate:.2f}")
            print(f" Модели сохраняются в: {self.model_manager.models_dir}")
            print(" Для остановки нажмите Ctrl+C")
            
            self.application.run_polling()
            
        except Exception as e:
            logger.error(f"Критическая ошибка при запуске бота: {e}")
            print(f" Критическая ошибка: {e}")
            sys.exit(1)