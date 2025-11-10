# bot/models.py
import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelManager:
    """Класс для управления моделями и их сохранением"""
    
    def __init__(self, models_dir="saved_models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
    def get_model_path(self, ticker, model_type):
        # Нормализуем имя файла для российских тикеров
        ticker_2 = ticker.replace('.', '_')
        return os.path.join(self.models_dir, f"{ticker_2}_{model_type}.pkl")
    
    def get_lstm_path(self, ticker):
        ticker_2 = ticker.replace('.', '_')
        return os.path.join(self.models_dir, f"{ticker_2}_lstm.h5")
    
    def should_retrain(self, ticker, model_type):
        """Проверяем, нужно ли переобучать модель"""
        model_path = self.get_model_path(ticker, model_type)
        if not os.path.exists(model_path):
            return True
            
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            last_updated = datetime.fromisoformat(model_data['last_updated'])
            # Переобучаем если модель старше 1 дня
            return (datetime.now() - last_updated) > timedelta(days=1)
        except:
            return True
    
    def save_model(self, ticker, model_type, model, scaler=None, metrics=None):
        try:
            model_data = {
                'model': model,
                'scaler': scaler,
                'metrics': metrics or {},
                'last_updated': datetime.now().isoformat(),
                'ticker': ticker,
                'model_type': model_type
            }
            
            with open(self.get_model_path(ticker, model_type), 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Модель {model_type} для {ticker} сохранена")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения модели {model_type} для {ticker}: {e}")
            return False
    
    def save_lstm_model(self, ticker, model, scaler, metrics=None):
        try:
            model.save(self.get_lstm_path(ticker))
            
            lstm_data = {
                'scaler': scaler,
                'metrics': metrics or {},
                'last_updated': datetime.now().isoformat(),
                'ticker': ticker,
                'model_type': 'LSTM'
            }
            
            with open(self.get_model_path(ticker, 'lstm_data'), 'wb') as f:
                pickle.dump(lstm_data, f)
                
            logger.info(f"LSTM модель для {ticker} сохранена")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения LSTM для {ticker}: {e}")
            return False
    
    def load_model(self, ticker, model_type):
        try:
            if model_type == 'LSTM':
                return self.load_lstm_model(ticker)
                
            model_path = self.get_model_path(ticker, model_type)
            if not os.path.exists(model_path):
                return None
                
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            last_updated = datetime.fromisoformat(model_data['last_updated'])
            if (datetime.now() - last_updated).days > 1:  # Уменьшили до 1 дня
                logger.info(f"Модель {model_type} для {ticker} устарела")
                return None
                
            logger.info(f"Модель {model_type} для {ticker} загружена")
            return model_data
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_type} для {ticker}: {e}")
            return None
    
    def load_lstm_model(self, ticker):
        try:
            lstm_path = self.get_lstm_path(ticker)
            data_path = self.get_model_path(ticker, 'lstm_data')
            
            if not os.path.exists(lstm_path) or not os.path.exists(data_path):
                return None
                
            model = load_model(lstm_path)
            with open(data_path, 'rb') as f:
                model_data = pickle.load(f)
                
            model_data['model'] = model
            
            last_updated = datetime.fromisoformat(model_data['last_updated'])
            if (datetime.now() - last_updated).days > 1:  # Уменьшили до 1 дня
                logger.info(f"LSTM модель для {ticker} устарела")
                return None
                
            logger.info(f"LSTM модель для {ticker} загружена")
            return model_data
        except Exception as e:
            logger.error(f"Ошибка загрузки LSTM для {ticker}: {e}")
            return None

class ModelTrainer:
    """Класс для обучения и работы с моделями"""
    
    def __init__(self, model_manager, feature_engineer):
        self.model_manager = model_manager
        self.feature_engineer = feature_engineer
    
    def train_ml_model(self, X_train, y_train, model_type='ridge'):
        """Обучение ML модели"""
        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)  # Уменьшили estimators для скорости
        
        model.fit(X_train, y_train)
        return model
    
    def train_arima_model(self, train_data):
        """Обучение ARIMA модели"""
        try:
            # Упрощенный ARIMA для скорости
            try:
                model = ARIMA(train_data, order=(1,1,1))
                fitted_model = model.fit()
                return fitted_model
            except:
                # Резервная модель
                model = ARIMA(train_data, order=(0,1,0))
                return model.fit()
        except Exception as e:
            logger.error(f"Ошибка ARIMA: {e}")
            return None
    
    def create_lstm_model(self, input_shape):
        """Создание LSTM модели"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            Dense(10),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error',
                     metrics=['mae'])
        return model
    
    def train_lstm_model(self, ticker, data):
        """Обучение LSTM модели"""
        try:
            if len(data) < 100:
                print(f" Недостаточно данных для LSTM: {len(data)}")
                return None
                
            lstm_data = data['Close'].values.reshape(-1, 1)
            lstm_scaler = StandardScaler()
            lstm_data_scaled = lstm_scaler.fit_transform(lstm_data)
            
            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(seq_length, len(data)):
                    X.append(data[i-seq_length:i, 0])
                    y.append(data[i, 0])
                return np.array(X), np.array(y)
            
            seq_length = 30
            if len(lstm_data_scaled) <= seq_length:
                print(f" Недостаточно данных для LSTM: {len(lstm_data_scaled)} <= {seq_length}")
                return None
                
            X, y = create_sequences(lstm_data_scaled, seq_length)
            
            if len(X) < 30:
                print(f" Слишком мало последовательностей для LSTM: {len(X)}")
                return None
                
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            model = self.create_lstm_model((seq_length, 1))
            
            print("🔄 Обучаем LSTM...")
            history = model.fit(
                X_train, y_train,
                batch_size=16,
                epochs=50,
                validation_data=(X_test, y_test),
                verbose=0,
                shuffle=False
            )
            
            self.model_manager.save_lstm_model(
                ticker=ticker,
                model=model,
                scaler=lstm_scaler,
                metrics={'loss': history.history['loss'][-1]}
            )
            
            return {'model': model, 'scaler': lstm_scaler}
            
        except Exception as e:
            logger.error(f"Ошибка обучения LSTM: {e}")
            print(f"❌ LSTM обучение не удалось: {e}")
            return None
    
    def train_or_load_models(self, ticker, data):
        """Обучение или загрузка моделей"""
        print(" Переобучаем модели...")
        return self.train_new_models(ticker, data)
    
    def train_new_models(self, ticker, data, existing_models=None):
        """Обучение новых моделей"""
        if len(data) < 60:  # Минимум 60 дней данных
            return None, None, f"Недостаточно данных для обучения: {len(data)} дней"
        
        feature_data = self.feature_engineer.create_features(data)
        
        if len(feature_data) < 30:
            return None, None, "Недостаточно данных для обучения после создания признаков"
        
        split_idx = int(len(feature_data) * 0.8)
        if split_idx < 10:
            return None, None, "Слишком мало данных для разделения на train/test"
            
        train_data = feature_data[:split_idx]
        test_data = feature_data[split_idx:]
        
        feature_cols = [col for col in feature_data.columns if col not in ['Price', 'Close', 'Open', 'High', 'Low', 'Volume']]
        X_train = train_data[feature_cols]
        y_train = train_data['Price']
        X_test = test_data[feature_cols]
        y_test = test_data['Price']
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models_data = {}
        
        # Ridge
        print(" Обучение Ridge...")
        try:
            ridge_model = self.train_ml_model(X_train_scaled, y_train, 'ridge')
            ridge_pred = ridge_model.predict(X_test_scaled)
            ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
            
            self.model_manager.save_model(ticker, 'ridge', ridge_model, scaler, 
                                        {'rmse': ridge_rmse})
            models_data['Ridge'] = {
                'model': ridge_model,
                'scaler': scaler,
                'metrics': {'rmse': ridge_rmse},
                'predictions': ridge_pred
            }
        except Exception as e:
            logger.error(f"Ошибка Ridge: {e}")
            models_data['Ridge'] = {'model': None, 'metrics': {'rmse': float('inf')}}
        
        # Random Forest
        print(" Обучение Random Forest...")
        try:
            rf_model = self.train_ml_model(X_train_scaled, y_train, 'random_forest')
            rf_pred = rf_model.predict(X_test_scaled)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            
            self.model_manager.save_model(ticker, 'random_forest', rf_model, scaler,
                                        {'rmse': rf_rmse})
            models_data['Random Forest'] = {
                'model': rf_model,
                'scaler': scaler,
                'metrics': {'rmse': rf_rmse},
                'predictions': rf_pred
            }
        except Exception as e:
            logger.error(f"Ошибка Random Forest: {e}")
            models_data['Random Forest'] = {'model': None, 'metrics': {'rmse': float('inf')}}
        
        # ARIMA
        print(" Обучение ARIMA...")
        try:
            arima_model = self.train_arima_model(y_train)
            if arima_model:
                arima_pred = arima_model.forecast(steps=len(y_test))
                arima_rmse = np.sqrt(mean_squared_error(y_test, arima_pred))
            else:
                arima_rmse = float('inf')
            
            self.model_manager.save_model(ticker, 'arima', arima_model, None,
                                        {'rmse': arima_rmse})
            models_data['ARIMA'] = {
                'model': arima_model,
                'scaler': None,
                'metrics': {'rmse': arima_rmse},
                'predictions': arima_pred if arima_model else np.zeros(len(y_test))
            }
        except Exception as e:
            logger.error(f"Ошибка ARIMA: {e}")
            models_data['ARIMA'] = {'model': None, 'metrics': {'rmse': float('inf')}}
        
        # LSTM (пробуем только если достаточно данных)
        print(" Обучение LSTM...")
        try:
            if len(data) >= 100:
                lstm_result = self.train_lstm_model(ticker, data)
                if lstm_result and lstm_result['model'] is not None:
                    lstm_pred = self.lstm_predict(lstm_result['model'], lstm_result['scaler'], data, len(y_test))
                    if len(lstm_pred) == len(y_test):
                        lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
                    else:
                        lstm_rmse = float('inf')
                else:
                    lstm_rmse = float('inf')
            else:
                lstm_rmse = float('inf')
                lstm_result = None
            
            models_data['LSTM'] = {
                'model': lstm_result['model'] if lstm_result else None,
                'scaler': lstm_result['scaler'] if lstm_result else None,
                'metrics': {'rmse': lstm_rmse},
                'predictions': lstm_pred if lstm_result and 'lstm_pred' in locals() else np.zeros(len(y_test))
            }
        except Exception as e:
            logger.error(f"Ошибка LSTM: {e}")
            models_data['LSTM'] = {'model': None, 'metrics': {'rmse': float('inf')}}
        
        # Выбор лучшей модели
        best_model_name = None
        best_score = float('inf')
        
        for name, data_dict in models_data.items():
            if (data_dict.get('model') is not None and 
                data_dict.get('metrics', {}).get('rmse', float('inf')) < best_score):
                best_score = data_dict['metrics']['rmse']
                best_model_name = name
        
        if best_model_name is None:
            return self.create_fallback_model(data), "Fallback", "Используется резервная модель"
        
        best_model_data = models_data[best_model_name]
        
        return best_model_data, best_model_name, f"Лучшая модель: {best_model_name}, RMSE: {best_score:.2f}"

    def create_fallback_model(self, data):
        """Создание резервной модели"""
        return {
            'model': None,
            'scaler': None,
            'metrics': {'rmse': float('inf')}
        }

    def lstm_predict(self, model, scaler, data, steps):
        """Прогноз LSTM"""
        try:
            lstm_data = data['Close'].values.reshape(-1, 1)
            lstm_data_scaled = scaler.transform(lstm_data)
            
            seq_length = 30
            if len(lstm_data_scaled) < seq_length:
                return [data['Close'].iloc[-1]] * steps
                
            current_sequence = lstm_data_scaled[-seq_length:].reshape(1, seq_length, 1)
            
            forecast_scaled = []
            for _ in range(steps):
                pred = model.predict(current_sequence, verbose=0)[0][0]
                forecast_scaled.append(pred)
                current_sequence = np.append(current_sequence[0][1:], [[pred]], axis=0).reshape(1, seq_length, 1)
            
            forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
            return forecast
        except Exception as e:
            logger.error(f"Ошибка прогноза LSTM: {e}")
            return [data['Close'].iloc[-1]] * steps

    def generate_forecast(self, model_data, model_name, data, days=30):
        """Генерация прогноза"""
        try:
            if model_data is None or model_data.get('model') is None:
                return self.trend_forecast(data, days)
                
            if model_name in ['Ridge', 'Random Forest']:
                forecast = self.ml_forecast(model_data, data, days)
            elif model_name == 'ARIMA':
                forecast = self.arima_forecast(model_data['model'], days)
            elif model_name == 'LSTM':
                forecast = self.lstm_forecast(model_data, data, days)
            else:
                forecast = self.trend_forecast(data, days)
            
            forecast = self.add_realistic_trend(data, forecast)
            return forecast
            
        except Exception as e:
            logger.error(f"Ошибка генерации прогноза для {model_name}: {e}")
            return self.trend_forecast(data, days)

    def trend_forecast(self, data, days):
        """Простой прогноз на основе тренда"""
        recent_data = data['Close'].iloc[-30:]
        if len(recent_data) < 10:
            return [data['Close'].iloc[-1]] * days
        
        x = np.arange(len(recent_data))
        y = recent_data.values
        z = np.polyfit(x, y, 1)
        trend_slope = z[0]
        
        last_price = data['Close'].iloc[-1]
        forecast = []
        
        for i in range(days):
            daily_change = trend_slope + np.random.normal(0, abs(trend_slope) * 0.3)  # Уменьшили шум
            new_price = last_price + daily_change
            forecast.append(max(new_price, last_price * 0.9))  # Максимум 10% падение
            last_price = new_price
        
        return forecast

    def add_realistic_trend(self, data, forecast):
        """Добавляет тренд к прогнозу"""
        if len(forecast) == 0:
            return forecast
            
        returns = data['Close'].pct_change().dropna()
        if len(returns) > 0:
            volatility = returns.std()
            mean_return = returns.mean()
        else:
            volatility = 0.01
            mean_return = 0.001
        
        improved_forecast = [forecast[0]]
        current_price = forecast[0]
        
        for i in range(1, len(forecast)):
            random_change = np.random.normal(mean_return, volatility * 0.5)
            new_price = current_price * (1 + random_change)
            
            max_daily_change = 0.03
            new_price = max(min(new_price, current_price * (1 + max_daily_change)), 
                           current_price * (1 - max_daily_change))
            
            improved_forecast.append(new_price)
            current_price = new_price
        
        return improved_forecast

    def ml_forecast(self, model_data, data, days):
        """Прогноз для ML моделей"""
        try:
            feature_data = self.feature_engineer.create_features(data)
            if len(feature_data) == 0:
                return self.trend_forecast(data, days)
            
            feature_cols = [col for col in feature_data.columns if col not in ['Price', 'Close', 'Open', 'High', 'Low', 'Volume']]
            current_features = feature_data[feature_cols].iloc[-1:].values
            
            scaler = model_data['scaler']
            model = model_data['model']
            
            current_features_scaled = scaler.transform(current_features)
            
            forecast = []
            features_sequence = current_features[0].copy()
            
            for i in range(days):
                pred = model.predict(current_features_scaled)[0]
                forecast.append(max(0, pred))
                
                features_sequence = np.roll(features_sequence, 1)
                features_sequence[0] = pred
                
                if i % 5 == 0:
                    features_sequence[0] *= (1 + np.random.normal(0, 0.005))
                
                current_features = features_sequence.reshape(1, -1)
                current_features_scaled = scaler.transform(current_features)
            
            return forecast
        except Exception as e:
            logger.error(f"Ошибка ML прогноза: {e}")
            return self.trend_forecast(data, days)

    def arima_forecast(self, model, days):
        """Прогноз для ARIMA"""
        try:
            if model is None:
                return [0] * days
            forecast = model.forecast(steps=days)
            return [max(0.1, x) for x in forecast]
        except Exception as e:
            logger.error(f"Ошибка прогноза ARIMA: {e}")
            return [0] * days

    def lstm_forecast(self, model_data, data, days):
        """Прогноз для LSTM"""
        try:
            if model_data.get('model') is None or model_data.get('scaler') is None:
                return self.trend_forecast(data, days)
                
            forecast = self.lstm_predict(model_data['model'], model_data['scaler'], data, days)
            return forecast
        except Exception as e:
            logger.error(f"Ошибка прогноза LSTM: {e}")
            return self.trend_forecast(data, days)