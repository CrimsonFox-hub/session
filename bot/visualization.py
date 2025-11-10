# bot/visualization.py
import matplotlib.pyplot as plt
import io
from datetime import timedelta

class ChartBuilder:
    """Построитель графиков"""
    
    def __init__(self, trading_engine):
        self.trading_engine = trading_engine
    
    def create_plot(self, historical_data, forecast, ticker):
        """Создание графика с торговыми точками"""
        plt.figure(figsize=(14, 8))
        
        # Берем последние 100 точек исторических данных
        historical_dates = historical_data.index[-100:]
        historical_prices = historical_data['Close'][-100:]
        
        # Генерируем даты для прогноза
        last_date = historical_dates[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast))]
        
        plt.plot(historical_dates, historical_prices, label='Исторические данные', linewidth=2, color='blue')
        plt.plot(forecast_dates, forecast, label='Прогноз на 30 дней', linewidth=2, color='red')
        
        # Находим и отмечаем торговые точки на прогнозе
        buy_points, sell_points, _ = self.trading_engine.find_trading_points(forecast)
        
        # Отмечаем точки покупки
        for bp in buy_points:
            if bp < len(forecast_dates):
                plt.plot(forecast_dates[bp], forecast[bp], 'g^', markersize=10, label='Покупка' if bp == buy_points[0] else "")
        
        # Отмечаем точки продажи
        for sp in sell_points:
            if sp < len(forecast_dates):
                plt.plot(forecast_dates[sp], forecast[sp], 'rv', markersize=10, label='Продажа' if sp == sell_points[0] else "")
        
        plt.title(f'Прогноз цен акций {ticker} с торговыми точками', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel('Цена ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()
        
        return buf