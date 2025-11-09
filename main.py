import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import io
import asyncio
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_logs.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockAnalysisBot:
    def __init__(self, telegram_token):
        self.telegram_token = telegram_token
        self.user_sessions = {}
        
    def download_stock_data(self, ticker, period='2y'):
        """Загрузка данных об акциях"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if data.empty:
                return None
            return data
        except Exception as e:
            logger.error(f"Ошибка загрузки данных для {ticker}: {e}")
            return None
    
    def create_features(self, data, window=30):
        """Создание признаков для ML моделей"""
        df = data.copy()
        df['Price'] = df['Close']
        
        # Создание лаговых признаков
        for i in range(1, window + 1):
            df[f'Lag_{i}'] = df['Price'].shift(i)
        
        # Технические индикаторы
        df['SMA_10'] = df['Price'].rolling(window=10).mean()
        df['SMA_30'] = df['Price'].rolling(window=30).mean()
        df['EMA_12'] = df['Price'].ewm(span=12).mean()
        df['EMA_26'] = df['Price'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['RSI'] = self.calculate_rsi(df['Price'])
        df['Volatility'] = df['Price'].rolling(window=20).std()
        
        df = df.dropna()
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_ml_model(self, X_train, y_train, model_type='ridge'):
        """Обучение классической ML модели"""
        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        return model
    
    def train_arima_model(self, train_data):
        """Обучение ARIMA модели"""
        try:
            model = ARIMA(train_data, order=(5,1,0))
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            logger.error(f"Ошибка ARIMA: {e}")
            return None
    
    def create_lstm_model(self, input_shape):
        """Создание LSTM модели"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error')
        return model
    
    def train_models(self, data):
        """Обучение всех моделей и выбор лучшей"""
        # Подготовка данных
        feature_data = self.create_features(data)
        
        if len(feature_data) < 100:
            return None, None, "Недостаточно данных для обучения"
        
        # Разделение на train/test
        split_idx = int(len(feature_data) * 0.8)
        train_data = feature_data[:split_idx]
        test_data = feature_data[split_idx:]
        
        # Признаки и целевая переменная
        feature_cols = [col for col in feature_data.columns if col not in ['Price', 'Close', 'Open', 'High', 'Low', 'Volume']]
        X_train = train_data[feature_cols]
        y_train = train_data['Price']
        X_test = test_data[feature_cols]
        y_test = test_data['Price']
        
        # Масштабирование
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Обучение ML моделей
        ridge_model = self.train_ml_model(X_train_scaled, y_train, 'ridge')
        rf_model = self.train_ml_model(X_train_scaled, y_train, 'random_forest')
        
        # Прогнозы ML моделей
        ridge_pred = ridge_model.predict(X_test_scaled)
        rf_pred = rf_model.predict(X_test_scaled)
        
        # Обучение ARIMA
        arima_model = self.train_arima_model(y_train)
        if arima_model:
            arima_pred = arima_model.forecast(steps=len(y_test))
        else:
            arima_pred = np.zeros(len(y_test))
        
        # Подготовка данных для LSTM
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
        
        # LSTM данные
        lstm_data = data['Close'].values.reshape(-1, 1)
        lstm_scaler = StandardScaler()
        lstm_data_scaled = lstm_scaler.fit_transform(lstm_data)
        
        seq_length = 30
        X_lstm, y_lstm = create_sequences(lstm_data_scaled, seq_length)
        
        split_idx_lstm = int(len(X_lstm) * 0.8)
        X_train_lstm, X_test_lstm = X_lstm[:split_idx_lstm], X_lstm[split_idx_lstm:]
        y_train_lstm, y_test_lstm = y_lstm[:split_idx_lstm], y_lstm[split_idx_lstm:]
        
        # Обучение LSTM
        lstm_model = self.create_lstm_model((seq_length, 1))
        lstm_model.fit(X_train_lstm, y_train_lstm, 
                      batch_size=32, 
                      epochs=50, 
                      validation_data=(X_test_lstm, y_test_lstm),
                      verbose=0)
        
        # Прогноз LSTM
        lstm_pred_scaled = lstm_model.predict(X_test_lstm)
        lstm_pred = lstm_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
        
        # Оценка моделей
        models = {
            'Ridge': (ridge_pred, ridge_model),
            'Random Forest': (rf_pred, rf_model),
            'ARIMA': (arima_pred, arima_model),
            'LSTM': (lstm_pred[:len(y_test)], lstm_model)
        }
        
        best_model = None
        best_score = float('inf')
        best_name = None
        
        for name, (pred, model) in models.items():
            if len(pred) == len(y_test):
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                if rmse < best_score:
                    best_score = rmse
                    best_model = model
                    best_name = name
        
        return best_model, best_name, f"Лучшая модель: {best_name}, RMSE: {best_score:.2f}"
    
    def generate_forecast(self, model, model_name, data, days=30):
        """Генерация прогноза на 30 дней"""
        if model_name in ['Ridge', 'Random Forest']:
            return self.ml_forecast(model, data, days)
        elif model_name == 'ARIMA':
            return self.arima_forecast(model, days)
        else:  # LSTM
            return self.lstm_forecast(model, data, days)
    
    def ml_forecast(self, model, data, days):
        """Прогноз для ML моделей"""
        feature_data = self.create_features(data)
        feature_cols = [col for col in feature_data.columns if col not in ['Price', 'Close', 'Open', 'High', 'Low', 'Volume']]
        
        current_features = feature_data[feature_cols].iloc[-1:].values
        scaler = StandardScaler()
        scaler.fit(feature_data[feature_cols])
        
        forecast = []
        current_features_scaled = scaler.transform(current_features)
        
        for _ in range(days):
            pred = model.predict(current_features_scaled)[0]
            forecast.append(pred)
            
            # Обновление features для следующего прогноза
            # (упрощенная логика - в реальном проекте нужно более сложное обновление)
            new_features = current_features[0][1:]  # Сдвиг лагов
            new_features = np.append(new_features, pred)  # Добавление нового прогноза
            
            if len(new_features) < len(current_features[0]):
                new_features = np.append(new_features, [pred] * (len(current_features[0]) - len(new_features)))
            
            current_features = new_features.reshape(1, -1)
            current_features_scaled = scaler.transform(current_features)
        
        return forecast
    
    def arima_forecast(self, model, days):
        """Прогноз для ARIMA"""
        try:
            forecast = model.forecast(steps=days)
            return forecast
        except Exception as e:
            logger.error(f"Ошибка прогноза ARIMA: {e}")
            return [0] * days
    
    def lstm_forecast(self, model, data, days):
        """Прогноз для LSTM"""
        try:
            lstm_data = data['Close'].values.reshape(-1, 1)
            lstm_scaler = StandardScaler()
            lstm_data_scaled = lstm_scaler.fit_transform(lstm_data)
            
            seq_length = 30
            current_sequence = lstm_data_scaled[-seq_length:].reshape(1, seq_length, 1)
            
            forecast_scaled = []
            for _ in range(days):
                pred = model.predict(current_sequence, verbose=0)[0][0]
                forecast_scaled.append(pred)
                # Обновление последовательности
                current_sequence = np.append(current_sequence[0][1:], [[pred]], axis=0).reshape(1, seq_length, 1)
            
            forecast = lstm_scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
            return forecast
        except Exception as e:
            logger.error(f"Ошибка прогноза LSTM: {e}")
            return [0] * days
    
    def find_trading_points(self, prices):
        """Поиск точек покупки и продажи"""
        buy_points = []
        sell_points = []
        
        for i in range(1, len(prices)-1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:  # Локальный минимум
                buy_points.append(i)
            elif prices[i] > prices[i-1] and prices[i] > prices[i+1]:  # Локальный максимум
                sell_points.append(i)
        
        return buy_points, sell_points
    
    def calculate_profit(self, prices, buy_points, sell_points, investment):
        """Расчет потенциальной прибыли"""
        cash = investment
        shares = 0
        transactions = []
        
        all_points = sorted([(i, 'buy') for i in buy_points] + [(i, 'sell') for i in sell_points])
        
        for day, action in all_points:
            price = prices[day]
            if action == 'buy' and cash > 0:
                shares_bought = cash / price
                shares += shares_bought
                cash = 0
                transactions.append(f"День {day+1}: ПОКУПКА по цене {price:.2f}")
            elif action == 'sell' and shares > 0:
                cash = shares * price
                shares = 0
                transactions.append(f"День {day+1}: ПРОДАЖА по цене {price:.2f}")
        
        # Продажа в последний день, если остались акции
        if shares > 0:
            final_cash = shares * prices[-1]
            cash += final_cash
            transactions.append(f"День {len(prices)}: ФИНАЛЬНАЯ ПРОДАЖА по цене {prices[-1]:.2f}")
        
        profit = cash - investment
        profit_percentage = (profit / investment) * 100
        
        return profit, profit_percentage, transactions
    
    def create_plot(self, historical_data, forecast, ticker):
        """Создание графика"""
        plt.figure(figsize=(12, 6))
        
        # Исторические данные
        historical_dates = historical_data.index[-100:]  # Последние 100 дней
        historical_prices = historical_data['Close'][-100:]
        
        # Прогноз
        forecast_dates = [historical_dates[-1] + timedelta(days=i+1) for i in range(len(forecast))]
        
        plt.plot(historical_dates, historical_prices, label='Исторические данные', linewidth=2)
        plt.plot(forecast_dates, forecast, label='Прогноз на 30 дней', linewidth=2, color='red')
        
        plt.title(f'Прогноз цен акций {ticker}', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel('Цена ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Сохранение в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()
        
        return buf
    
    def log_session(self, user_id, ticker, investment, best_model, metric, profit):
        """Логирование сессии"""
        log_entry = {
            'user_id': user_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ticker': ticker,
            'investment': investment,
            'best_model': best_model,
            'metric': metric,
            'profit': profit
        }
        
        log_line = (f"{log_entry['timestamp']} | User: {log_entry['user_id']} | "
                   f"Ticker: {log_entry['ticker']} | Investment: ${log_entry['investment']} | "
                   f"Model: {log_entry['best_model']} | Metric: {log_entry['metric']} | "
                   f"Profit: ${log_entry['profit']:.2f}\n")
        
        with open('trading_logs.txt', 'a', encoding='utf-8') as f:
            f.write(log_line)
        
        logger.info(f"Запись в лог: {log_line.strip()}")

    async def start(self, update: Update, context: CallbackContext):
        """Обработчик команды /start"""
        welcome_text = """
🤖 Добро пожаловать в бот для анализа акций!

Я могу:
• Проанализировать акции по тикеру (AAPL, TSLA, etc.)
• Построить прогноз на 30 дней
• Дать рекомендации по покупке/продаже
• Рассчитать потенциальную прибыль

Для начала отправьте тикер компании и сумму инвестиции в формате:
`ТИКЕР СУММА`

Например: `AAPL 1000`
        """
        await update.message.reply_text(welcome_text, parse_mode='Markdown')
    
    async def handle_message(self, update: Update, context: CallbackContext):
        """Обработка сообщений пользователя"""
        user_id = update.effective_user.id
        text = update.message.text.strip().upper()
        
        try:
            parts = text.split()
            if len(parts) < 2:
                await update.message.reply_text("Пожалуйста, укажите тикер и сумму инвестиции.\nНапример: `AAPL 1000`", parse_mode='Markdown')
                return
            
            ticker = parts[0]
            investment = float(parts[1])
            
            if investment <= 0:
                await update.message.reply_text("Сумма инвестиции должна быть положительной.")
                return
            
            # Сообщение о начале обработки
            wait_message = await update.message.reply_text("⏳ Загружаю данные и анализирую акции...")
            
            # Загрузка данных
            data = self.download_stock_data(ticker)
            if data is None or data.empty:
                await update.message.reply_text("❌ Не удалось загрузить данные по указанному тикеру. Проверьте правильность тикера.")
                return
            
            # Обучение моделей
            best_model, best_model_name, model_info = self.train_models(data)
            
            if best_model is None:
                await update.message.reply_text("❌ Не удалось обучить модели. Попробуйте другой тикер.")
                return
            
            # Генерация прогноза
            forecast = self.generate_forecast(best_model, best_model_name, data)
            
            # Создание графика
            plot_buf = self.create_plot(data, forecast, ticker)
            
            # Анализ торговых точек
            buy_points, sell_points = self.find_trading_points(forecast)
            profit, profit_percentage, transactions = self.calculate_profit(forecast, buy_points, sell_points, investment)
            
            # Текущая и прогнозируемая цена
            current_price = data['Close'].iloc[-1]
            forecast_price = forecast[-1]
            price_change = forecast_price - current_price
            price_change_percent = (price_change / current_price) * 100
            
            # Формирование ответа
            response = f"""
📊 **Анализ акций {ticker}**

{model_info}
            
💰 **Текущая цена:** ${current_price:.2f}
🎯 **Прогноз через 30 дней:** ${forecast_price:.2f}
📈 **Изменение:** ${price_change:.2f} ({price_change_percent:+.2f}%)

💼 **Инвестиция:** ${investment:.2f}
🎉 **Потенциальная прибыль:** ${profit:.2f} ({profit_percentage:+.2f}%)

🔄 **Рекомендации по торговле:**
"""
            
            for transaction in transactions[:10]:  # Показываем первые 10 транзакций
                response += f"• {transaction}\n"
            
            if len(transactions) > 10:
                response += f"• ... и еще {len(transactions) - 10} операций\n"
            
            response += f"\n📅 **Всего точек покупки:** {len(buy_points)}"
            response += f"\n📅 **Всего точек продажи:** {len(sell_points)}"
            
            # Отправка графика и текста
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=plot_buf,
                caption=response,
                parse_mode='Markdown'
            )
            
            # Удаление сообщения "ожидание"
            await wait_message.delete()
            
            # Логирование
            self.log_session(user_id, ticker, investment, best_model_name, 
                           model_info.split("RMSE: ")[1].split(",")[0], profit)
            
        except ValueError:
            await update.message.reply_text("❌ Неверный формат суммы. Убедитесь, что сумма - это число.")
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            await update.message.reply_text("❌ Произошла ошибка при обработке запроса. Попробуйте позже.")
    
    async def error_handler(self, update: Update, context: CallbackContext):
        """Обработчик ошибок"""
        logger.error(f"Ошибка: {context.error}", exc_info=context.error)
    
    def run(self):
        """Запуск бота"""
        application = Application.builder().token(self.telegram_token).build()
        
        # Обработчики команд
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        application.add_error_handler(self.error_handler)
        
        # Запуск бота
        logger.info("Бот запущен...")
        application.run_polling()

# Запуск бота
if __name__ == "__main__":
    TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"  # Замените на ваш токен
    
    bot = StockAnalysisBot(TELEGRAM_TOKEN)
    bot.run()