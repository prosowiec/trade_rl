import pandas as pd
import numpy as np
import gym
from gym import spaces
import ta 

class PortfolioEnv(gym.Env):
    def __init__(self, close_data: pd.Series, window_size=96, initial_cash=10000.0,
                 transaction_cost=1, max_allocation=0.5):
        """
        close_data: Series z cenami jednego aktywa, indeks = daty
        """
        super(PortfolioEnv, self).__init__()
        self.close_data = close_data
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_allocation = max_allocation
        self.trader_action = 0
        self.total_porfolio = initial_cash
        self.cash = initial_cash
        self.action_space = spaces.Dict({
            'trader': spaces.Discrete(3),  # 0=hold, 1=buy, 2=sell
            'portfolio_manager': spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        })
        self.indicators = PortfolioIndicators(self.window_size)
        
        # Obserwacja = okno cen + pozycja + gotówka + poprzednia akcja + alokacja
        obs_len = self.window_size + 1 + 1 + 1 + 1  # close + position + cash + prev_trader + prev_alloc
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_len,), dtype=np.float32)
        self.position = 0
        
        self.states_buy = []
        self.states_sell = []
        self.states_allocation = []
        self.shares_sell = []
        self.shares_buy = []
        self.states_position = []
        
        
        self.reset()

    def _get_obs(self):
        window_start = max(0, self.current_step - self.window_size)
        window = self.close_data.iloc[window_start:self.current_step].values.flatten()

        min_val = np.min(window)
        max_val = np.max(window)
        market_data = ((window - min_val) / (max_val - min_val + 1e-8)).astype(np.float32)

        current_price = self.close_data.iloc[self.current_step]
        total_value = self.cash + self.position * current_price

        norm_total_value = total_value / self.initial_cash
        norm_position = (self.position * current_price) / self.initial_cash
        cash_alloc = self.cash / (total_value + 1e-8)
        asset_alloc = (self.position * current_price) / (total_value + 1e-8)

        portfolio_info = np.array([
            float(norm_position),
            float(norm_total_value),
            float(cash_alloc),
            float(asset_alloc),
            float(self.trader_action)
        ], dtype=np.float32)

        indicators = self.indicators.calculate_all_indicators(self.close_data,self.current_step, self.portfolio_value_history,
                                                              self.cash, self.position, current_price)

        return np.round(np.concatenate([
            #market_data,
            portfolio_info,
            indicators
        ]), 4)
    
    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.position = 0.0  # liczba akcji
        self.prev_trader_action = 0
        self.prev_allocation = 0.0
        self.trader_action = 0
        self.portfolio_value_history = []
        self.total_porfolio = self.initial_cash
        
        self.shares_sell = []
        self.shares_buy = []
        self.states_buy = []
        self.states_sell = []
        self.states_allocation = []
        self.states_position = []
        self.cash_state = []

        return self._get_obs()


    def step(self, action):
        executed = False
        self.trader_action = action['trader']
        allocation = action['portfolio_manager'][0] #* self.max_allocation
        curr_price = self.close_data.iloc[self.current_step].values[0]
        prev_price = self.close_data.iloc[self.current_step - 1].values[0]
        prev_value = self.cash + self.position * prev_price
        transaction_cost = 0

        

        if self.current_step >= len(self.close_data):
            return self._get_obs(), 0.0, True, {}

        if np.round(allocation,3) == 0.0 and self.trader_action != 0 and self.current_step + 1< len(self.close_data) - 1:
            # Return penalty and current state unchanged
            return self._get_obs(), -0.1, False, {'error': 'Invalid allocation'}
        
        if np.isnan(allocation) or allocation < 0 or allocation > 1 and self.current_step  + 1< len(self.close_data) - 1:
            # Return penalty and current state unchanged
            return self._get_obs(), -1, False, {'error': 'Invalid allocation'}

        
        if self.trader_action == 1:  # BUY
            max_cash_available = self.cash - self.transaction_cost  # uwzględnij koszt stały
            
            if max_cash_available > 0:
                invest_amount = max_cash_available * allocation
                total_spend = invest_amount + self.transaction_cost
                shares = invest_amount / curr_price
                if total_spend <= self.cash:
                    
                    #print(shares,invest_amount,allocation, invest_amount / curr_price )
                    real_invest = shares * curr_price
                    total_spend = real_invest + self.transaction_cost
                    
                    if total_spend <= self.cash:
                        self.position += shares
                        self.cash -= total_spend
                        transaction_cost += self.transaction_cost
                        executed = True
                        self.states_buy.append(self.current_step)
                        self.shares_buy.append(shares)             
                
                
        elif self.trader_action == 2:  # SELL
            shares = self.position * allocation
            #print(shares,allocation )
            if shares > 0:
                revenue = shares * curr_price
                net_revenue = revenue #- self.transaction_cost
                if net_revenue > 0:
                    self.position -= shares
                    self.cash += net_revenue
                    #transaction_cost += self.transaction_cost
                    executed = True
                    self.states_sell.append(self.current_step)
                    self.shares_sell.append(shares)        
                    
        if executed:
            self.states_allocation.append(allocation)
            self.states_position.append(self.position)
            
        self.cash_state.append(self.cash)
        curr_value = self.cash + self.position * curr_price
        
        self.total_porfolio = curr_value
        portfolio_return = (curr_value - prev_value) / (prev_value + 1e-8) 
        entropy_coeff = 0.01  # Współczynnik entropii, można dostosować
        #reward = portfolio_return   #np.log(1 + portfolio_return + 1e-8) - transaction_cost / (prev_value + 1e-8)
        entropy = - (allocation * np.log(allocation + 1e-8) + (1 - allocation) * np.log(1 - allocation + 1e-8))
        reward = allocation * portfolio_return + entropy_coeff * entropy    
        #reward = np.log(1 + max(0, portfolio_return)) * allocation - np.log(1 - min(0, portfolio_return)) * (1 - allocation)

        # allocation_penalty = -0.01 * (allocation)  # im mniej zainwestowane, tym gorzej
        # reward += allocation_penalty

        # 3. Kara za zbyt częste zmiany alokacji (np. skoki z 0 do 1)
        
        
        #allocation_change_penalty = -0.2 * abs(allocation - prev_action)
        #reward += allocation_change_penalty
        
        # if self.trader_action != 0 and not executed:
        #     reward -= 200 # kara za niewykonaną transakcję
        
        # if invest_amount > self.cash and self.trader_action == 1:
        #     reward -= 10

        # elif executed:
        #     reward += 50

        self.prev_trader_action = self.trader_action
        self.prev_allocation = allocation
        self.portfolio_value_history.append(curr_value)
        self.current_step += 1

        done = self.current_step >= len(self.close_data) - 1
        info = {
            'portfolio_value': curr_value,
            'cash': self.cash,
            'position': self.position,
            'transaction_cost': transaction_cost,
            'return': portfolio_return,
            'executed': executed
        }
        reward = np.clip(reward, -1.0, 1.0)  
        
        return self._get_obs(), reward, done, info
        
    def get_portfolio_allocation(self):
        price = self.close_data.iloc[self.current_step - 1] if self.current_step > 0 else self.close_data.iloc[0]
        total_value = self.cash + self.position * price
        return {
            'asset_allocation': (self.position * price) / (total_value + 1e-8),
            'cash_allocation': self.cash / (total_value + 1e-8),
            'total_value': total_value,
            'shares': self.position
        }

    def sample_action(self):
        return {
            'trader': self.action_space['trader'].sample(),
            'portfolio_manager': self.action_space['portfolio_manager'].sample()
        }

    def get_price_window(self):
        window_start = max(0, self.current_step - self.window_size)
        window = self.close_data.iloc[window_start:self.current_step].values.flatten()
        min_val = np.min(window)
        max_val = np.max(window)
        market_data = ((window - min_val) / (max_val - min_val + 1e-8)).astype(np.float32)
        return market_data


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
