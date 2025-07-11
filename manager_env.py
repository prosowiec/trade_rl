import pandas as pd
import numpy as np
import gym
from gym import spaces
import ta 

class PortfolioEnv(gym.Env):
    def __init__(self, close_data: pd.DataFrame, window_size=96, initial_cash=10000.0,
                 transaction_cost=1, max_allocation=0.5):
        """
        close_data: DataFrame z cenami wielu aktywów (każda kolumna to inne aktywo)
        """
        super(PortfolioEnv, self).__init__()
        self.close_data = close_data
        self.asset_names = close_data.columns.tolist()
        self.n_assets = len(self.asset_names)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_allocation = max_allocation

        self.action_space = spaces.Dict({
            'trader': spaces.MultiDiscrete([3] * self.n_assets),  # 0=hold, 1=buy, 2=sell per asset
            'portfolio_manager': spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        })

        # Przykład: okno cen (n_assets * window), pozycje, gotówka, akcje tradera, alokacja
        obs_len = self.n_assets * window_size + self.n_assets + 1 + self.n_assets + self.n_assets
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_len,), dtype=np.float32)

        self.indicators = PortfolioIndicators(window_size)
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.position = np.zeros(self.n_assets)  # ilość akcji każdego aktywa
        self.prev_trader_action = np.zeros(self.n_assets)
        self.prev_allocation = np.zeros(self.n_assets)
        self.trader_action = np.zeros(self.n_assets)
        self.portfolio_value_history = []
        return self._get_obs()

    def _get_obs(self):
        window_start = max(0, self.current_step - self.window_size)
        curr_prices = self.close_data.iloc[window_start:self.current_step].values  # shape: (window, n_assets)
        #curr_prices = self.close_data.iloc[self.current_step].values  # shape: (n_assets,)
        #print(curr_prices)
        # Normalizacja bieżących cen (min-max z całego dostępnego zakresu)
        min_vals = self.close_data.min().values
        max_vals = self.close_data.max().values
        norm_prices = (curr_prices - min_vals) / (max_vals - min_vals + 1e-8)  # shape: (n_assets,)

        # Zakładamy, że prev_trader_action ma shape (n_assets,) i zawiera 0/1/2
        # Możesz też zakodować jako one-hot, ale tutaj normalizujemy: 0.0 (hold), 0.5 (buy), 1.0 (sell)
        norm_actions = self.trader_action / 2.0
        norm_actions = norm_actions[:, np.newaxis]  
        #print(norm_prices.T)
        #print(norm_actions)
        # Łączenie cen i decyzji per aktywo
        obs = np.concatenate([norm_prices.T, norm_actions], axis=1)  # shape: (n_assets, 2)

        return np.round(obs, 4)
    
    def step(self, action):
        executed = False
        self.trader_action = np.array(action['trader'])
        allocation = np.array(action['portfolio_manager'])
        curr_prices = self.close_data.iloc[self.current_step].values
        prev_prices = self.close_data.iloc[self.current_step - 1].values
        prev_value = self.cash + np.sum(self.position * prev_prices)
        transaction_cost_total = 0

        if np.any(np.isnan(allocation)) or np.any(allocation < 0) or np.any(allocation > 1):
            return self._get_obs(), -1.0, False, {'error': 'Invalid allocation'}

        for i in range(self.n_assets):
            act = self.trader_action[i]
            alloc = allocation[i]
            price = curr_prices[i]
            if act == 1:  # BUY
                invest_amount = self.cash * alloc
                cost = invest_amount + self.transaction_cost
                shares = invest_amount / price
                if cost <= self.cash:
                    self.position[i] += shares
                    self.cash -= cost
                    transaction_cost_total += self.transaction_cost
                    executed = True
            elif act == 2:  # SELL
                shares_to_sell = self.position[i] * alloc
                if shares_to_sell > 0:
                    revenue = shares_to_sell * price
                    self.position[i] -= shares_to_sell
                    self.cash += revenue  # koszt transakcji pomijany przy sprzedaży
                    executed = True

        curr_value = self.cash + np.sum(self.position * curr_prices)
        portfolio_return = (curr_value - prev_value) / (prev_value + 1e-8)

        entropy_coeff = 0.01
        entropy = -np.sum(allocation * np.log(allocation + 1e-8) + (1 - allocation) * np.log(1 - allocation + 1e-8))
        reward = np.dot(allocation, portfolio_return ) + entropy_coeff * entropy #* np.ones(self.n_assets)
        reward = np.clip(reward, -1.0, 1.0)

        self.prev_trader_action = self.trader_action.copy()
        self.prev_allocation = allocation.copy()
        self.portfolio_value_history.append(curr_value)
        self.current_step += 1
        done = self.current_step >= len(self.close_data) - 1

        info = {
            'portfolio_value': curr_value,
            'cash': self.cash,
            'position': self.position.copy(),
            'transaction_cost': transaction_cost_total,
            'return': portfolio_return,
            'executed': executed
        }
        return self._get_obs(), reward, done, info

    def get_portfolio_allocation(self):
        price = self.close_data.iloc[self.current_step - 1].values
        total_value = self.cash + np.sum(self.position * price)
        return {
            'asset_allocation': (self.position * price) / (total_value + 1e-8),
            'cash_allocation': self.cash / (total_value + 1e-8),
            'total_value': total_value,
            'shares': self.position.copy()
        }

    def sample_action(self):
        return {
            'trader': self.action_space['trader'].sample(),
            'portfolio_manager': self.action_space['portfolio_manager'].sample()
        }

    def get_price_window(self):
        # window_start = max(0, self.current_step - self.window_size)
        # window = self.close_data.iloc[window_start:self.current_step].values.T
        # norm_prices = []
        # for prices in window:
        #     min_val = np.min(prices)
        #     max_val = np.max(prices)
        #     norm = (prices - min_val) / (max_val - min_val + 1e-8)
        #     norm_prices.append(norm)
        window_start = max(0, self.current_step - self.window_size)
        curr_prices = self.close_data.iloc[window_start:self.current_step].values  # shape: (window, n_assets)
        #curr_prices = self.close_data.iloc[self.current_step].values  # shape: (n_assets,)
        #print(curr_prices)
        # Normalizacja bieżących cen (min-max z całego dostępnego zakresu)
        min_vals = self.close_data.min().values
        max_vals = self.close_data.max().values
        norm_prices = (curr_prices - min_vals) / (max_vals - min_vals + 1e-8)  # shape: (n_assets,)
        
        #norm_actions = norm_actions[:, np.newaxis]  

        return np.array(norm_prices).T


class PortfolioIndicators:
    def __init__(self, window_size=96):
        self.window_size = window_size
        
    def calculate_all_indicators(self, price_data, current_step, portfolio_value_history, 
                               cash, position, current_price):
        """Calculate all portfolio indicators"""
        indicators = {}
        
        # Get price window
        window_start = max(0, current_step - self.window_size)
        price_window = price_data.iloc[window_start:current_step]

        if isinstance(price_window, pd.DataFrame):
            # podmień 'close' na właściwą nazwę kolumny z ceną
            price_window = price_window['close']
            
        price_window = price_window.values.flatten()        
        # 1. VOLATILITY INDICATORS
        indicators.update(self._volatility_indicators(price_window))
        
        # 2. MOMENTUM INDICATORS
        indicators.update(self._momentum_indicators(price_window))
        
        # 3. MEAN REVERSION INDICATORS
        indicators.update(self._mean_reversion_indicators(price_window))
        
        # 4. PORTFOLIO PERFORMANCE INDICATORS
        indicators.update(self._portfolio_performance_indicators(
            portfolio_value_history, cash, position, current_price))
        
        # 5. RISK INDICATORS
        indicators.update(self._risk_indicators(price_window, portfolio_value_history))
        
        # 6. MARKET REGIME INDICATORS
        indicators.update(self._market_regime_indicators(price_window))
        
        # Convert to array and normalize
        values = list(indicators.values())
        return np.array(values, dtype=np.float32)
    
    def _volatility_indicators(self, prices):
        """Volatility-based indicators"""
        returns = np.diff(prices) / prices[:-1]
        
        return {
            'vol_5': np.std(returns[-5:]) if len(returns) >= 5 else 0,
            'vol_20': np.std(returns[-20:]) if len(returns) >= 20 else 0,
            'vol_ratio': (np.std(returns[-5:]) / np.std(returns[-20:]) 
                         if len(returns) >= 20 and np.std(returns[-20:]) > 0 else 1),
            'parkinson_vol': self._parkinson_volatility(prices),
        }
    
    def _momentum_indicators(self, prices):
        """Momentum-based indicators"""
        if len(prices) < 20:
            return {f'momentum_{i}': 0 for i in [5, 10, 20]} | {'rsi': 0.5}
        
        current_price = prices[-1]
        return {
            'momentum_5': (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0,
            'momentum_10': (current_price - prices[-10]) / prices[-10] if len(prices) >= 10 else 0,
            'momentum_20': (current_price - prices[-20]) / prices[-20] if len(prices) >= 20 else 0,
            'rsi': self._rsi(prices),
        }
    
    def _mean_reversion_indicators(self, prices):
        """Mean reversion indicators"""
        if len(prices) < 20:
            return {'zscore': 0, 'distance_from_sma': 0, 'bollinger_position': 0.5}
        
        sma_20 = np.mean(prices[-20:])
        std_20 = np.std(prices[-20:])
        current_price = prices[-1]
        
        # Z-score (standardized distance from mean)
        zscore = (current_price - sma_20) / std_20 if std_20 > 0 else 0
        
        # Distance from SMA (normalized)
        distance_from_sma = (current_price - sma_20) / sma_20
        
        # Bollinger Band position (0 = lower band, 1 = upper band, 0.5 = middle)
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        bollinger_position = ((current_price - lower_band) / (upper_band - lower_band) 
                            if upper_band != lower_band else 0.5)
        
        return {
            'zscore': np.tanh(zscore),  # Normalize to [-1, 1]
            'distance_from_sma': np.tanh(distance_from_sma * 10),  # Scale and normalize
            'bollinger_position': np.clip(bollinger_position, 0, 1),
        }
    
    def _portfolio_performance_indicators(self, portfolio_history, cash, position, current_price):
        """Portfolio performance indicators"""
        if len(portfolio_history) < 2:
            return {
                'portfolio_momentum': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0.5,
                'profit_factor': 1,
            }
        
        portfolio_values = np.array(portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Portfolio momentum
        portfolio_momentum = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Sharpe ratio (annualized, assuming daily data)
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252) 
                       if np.std(returns) > 0 else 0)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.5
        
        # Profit factor
        profits = np.sum(returns[returns > 0])
        losses = np.sum(np.abs(returns[returns < 0]))
        profit_factor = profits / losses if losses > 0 else 1
        
        return {
            'portfolio_momentum': np.tanh(portfolio_momentum),
            'sharpe_ratio': np.tanh(sharpe_ratio),
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': np.tanh(profit_factor),
        }
    
    def _risk_indicators(self, prices, portfolio_history):
        """Risk assessment indicators"""
        if len(prices) < 20:
            return {'var_95': 0, 'skewness': 0, 'kurtosis': 0}
        
        returns = np.diff(prices) / prices[:-1]
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Skewness (asymmetry of returns)
        skewness = self._skewness(returns)
        
        # Kurtosis (tail risk)
        kurtosis = self._kurtosis(returns)
        
        return {
            'var_95': var_95,
            'skewness': np.tanh(skewness),
            'kurtosis': np.tanh(kurtosis / 10),
        }
    
    def _market_regime_indicators(self, prices):
        """Market regime detection"""
        if len(prices) < 50:
            return {'trend_strength': 0, 'market_regime': 0.5, 'trend_consistency': 0.5}
        
        # Trend strength using linear regression
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        trend_strength = slope / np.mean(prices)
        
        # Market regime: trending vs sideways
        sma_short = np.mean(prices[-10:])
        sma_long = np.mean(prices[-50:])
        market_regime = (sma_short - sma_long) / sma_long
        
        # Trend consistency
        short_trends = []
        for i in range(5):
            start_idx = -10 * (i + 1)
            end_idx = -10 * i if i > 0 else None
            segment = prices[start_idx:end_idx]
            if len(segment) >= 2:
                x_seg = np.arange(len(segment))
                slope_seg, _ = np.polyfit(x_seg, segment, 1)
                short_trends.append(slope_seg)
        
        if short_trends:
            trend_consistency = np.std(short_trends) / (np.mean(np.abs(short_trends)) + 1e-8)
        else:
            trend_consistency = 0.5
        
        return {
            'trend_strength': np.tanh(trend_strength * 100),
            'market_regime': np.tanh(market_regime * 10),
            'trend_consistency': 1 / (1 + trend_consistency),  # Lower std = higher consistency
        }
    
    def _parkinson_volatility(self, prices):
        """Parkinson volatility estimator"""
        if len(prices) < 20:
            return 0
        
        # Simple approximation using price range
        highs = np.maximum.accumulate(prices[-20:])
        lows = np.minimum.accumulate(prices[-20:])
        
        if len(highs) > 1 and len(lows) > 1:
            log_hl = np.log(highs / lows)
            return np.sqrt(np.mean(log_hl ** 2) / (4 * np.log(2)))
        else:
            return 0
    
    def _rsi(self, prices, period=14):
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 0.5
        
        deltas = np.diff(prices)
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 1
        rs = avg_gain / avg_loss
        rsi = 1 - 1 / (1 + rs)
        return rsi
    
    def _skewness(self, data):
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        n = len(data)
        if std == 0 or n < 3:
            return 0
        skew = (np.sum((data - mean) ** 3) / n) / (std ** 3)
        return skew
    
    def _kurtosis(self, data):
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        n = len(data)
        if std == 0 or n < 4:
            return 0
        kurt = (np.sum((data - mean) ** 4) / n) / (std ** 4) - 3
        return kurt
