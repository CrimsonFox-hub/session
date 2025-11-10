# bot/trading.py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TradingEngine:
    """Движок торговли"""
    
    def __init__(self, converter):
        self.converter = converter
    
    def find_trading_points(self, prices):
        """поиск торговых точек"""
        if len(prices) < 10:
            return [], [], "недостаточно данных"
        
        buy_points = []
        sell_points = []
        
        # Стратегия: ищем четкие минимумы и максимумы
        for i in range(2, len(prices) - 2):
            # Явный минимум для покупки
            if (prices[i] < prices[i-1] and prices[i] < prices[i-2] and 
                prices[i] < prices[i+1] and prices[i] < prices[i+2]):
                buy_points.append(i)
            
            # Явный максимум для продажи
            elif (prices[i] > prices[i-1] and prices[i] > prices[i-2] and 
                  prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                sell_points.append(i)
        
        # убираем точки, которые слишком близко друг к другу
        min_gap = max(3, len(prices) // 10)
        
        filtered_buy = []
        filtered_sell = []
        
        last_action_day = -min_gap
        for buy in sorted(buy_points):
            if buy - last_action_day >= min_gap:
                filtered_buy.append(buy)
                last_action_day = buy
        
        last_action_day = -min_gap
        for sell in sorted(sell_points):
            if sell - last_action_day >= min_gap:
                filtered_sell.append(sell)
                last_action_day = sell
        
        # Создаем пары покупка-продажа
        trading_pairs = []
        used_sells = set()
        
        for buy in filtered_buy:
            possible_sells = [s for s in filtered_sell if s > buy and s not in used_sells]
            if possible_sells:
                sell = min(possible_sells)
                if prices[sell] > prices[buy] * 1.01:  # Минимум 1% прибыли
                    trading_pairs.append((buy, sell))
                    used_sells.add(sell)
        
        if trading_pairs:
            buy_points = [pair[0] for pair in trading_pairs]
            sell_points = [pair[1] for pair in trading_pairs]
            return buy_points, sell_points, f"найдено {len(trading_pairs)} торговых пар"
        
        # Резервная стратегия
        if len(prices) > 5:
            min_idx = np.argmin(prices)
            max_after_min = -1
            
            if min_idx < len(prices) - 1:
                max_after_min = np.argmax(prices[min_idx:]) + min_idx
            
            if max_after_min > min_idx and prices[max_after_min] > prices[min_idx] * 1.02:
                return [min_idx], [max_after_min], "резервная стратегия"
        
        return [], [], "не найдено прибыльных возможностей"
    
    def calculate_profit(self, prices, buy_points, sell_points, investment_rub):
        """Простой и понятный расчет прибыли"""
        if not buy_points:
            return 0, 0, [" Не найдено точек для покупки"]
        
        investment_usd = self.converter.to_usd(investment_rub)
        
        # Создаем последовательность действий
        actions = []
        for i, (buy, sell) in enumerate(zip(buy_points, sell_points)):
            if buy < len(prices) and sell < len(prices):
                actions.append((buy, 'buy', prices[buy], f"Покупка #{i+1}"))
                actions.append((sell, 'sell', prices[sell], f"Продажа #{i+1}"))
        
        # Сортируем по времени
        actions.sort(key=lambda x: x[0])
        
        cash_usd = investment_usd
        shares = 0.0
        transactions = []
        total_profit_usd = 0
        
        for day, action, price, description in actions:
            if action == 'buy' and cash_usd > 0:
                shares_bought = cash_usd / price
                shares += shares_bought
                buy_amount_usd = cash_usd
                cash_usd = 0
                transactions.append(f" День {day}: {description} - куплено {shares_bought:.2f} акций по ${price:.2f}")
                    
            elif action == 'sell' and shares > 0:
                revenue_usd = shares * price
                profit_usd = revenue_usd - (buy_amount_usd if 'buy_amount_usd' in locals() else 0)
                total_profit_usd += profit_usd
                profit_rub = self.converter.to_rub(profit_usd)
                cash_usd += revenue_usd
                
                profit_indicator = "🟢" if profit_usd > 0 else "🔴"
                profit_text = f"прибыль: {profit_rub:.0f}₽" if profit_usd > 0 else f"убыток: {abs(profit_rub):.0f}₽"
                
                transactions.append(f" День {day}: {description} - продано {shares:.2f} акций по ${price:.2f} {profit_indicator} ({profit_text})")
                shares = 0
        
        # Финальная продажа если остались акции
        if shares > 0 and len(prices) > 0:
            final_price = prices[-1]
            final_revenue_usd = shares * final_price
            final_profit_usd = final_revenue_usd - (buy_amount_usd if 'buy_amount_usd' in locals() else 0)
            total_profit_usd += final_profit_usd
            cash_usd += final_revenue_usd
            
            profit_indicator = "🟢" if final_profit_usd > 0 else "🔴"
            profit_text = f"прибыль: {self.converter.to_rub(final_profit_usd):.0f}₽" if final_profit_usd > 0 else f"убыток: {abs(self.converter.to_rub(final_profit_usd)):.0f}₽"
            
            transactions.append(f"В итоге: {shares:.2f} акций по ${final_price:.2f} {profit_indicator} ({profit_text})")
        
        final_cash_rub = self.converter.to_rub(cash_usd)
        profit_rub = final_cash_rub - investment_rub
        profit_percentage = (profit_rub / investment_rub) * 100 if investment_rub > 0 else 0
        
        return profit_rub, profit_percentage, transactions