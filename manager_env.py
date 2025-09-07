import pandas as pd
import numpy as np
import gym
from gym import spaces
import ta 


class DifferentialSharpeRatio:
    def __init__(self, eta=0.01):
        self.eta = eta
        self.A = 0.0  # running mean return
        self.B = 0.0  # running mean squared return
    
    def update(self, R_t):
        """
        Update Differential Sharpe Ratio with new return R_t.
        Returns the differential Sharpe ratio D_t.
        """
        # Increments
        delta_A = R_t - self.A
        delta_B = R_t**2 - self.B

        # Compute numerator and denominator
        numerator = self.B * delta_A - 0.5 * self.A * delta_B
        denominator = (self.B - self.A**2 + 1e-8) ** (1.5)

        D_t = numerator / (denominator + 1e-8)

        # Update A, B (exponentially smoothed estimates)
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return D_t

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
        self.total_portfolio = initial_cash
        self.dsr = DifferentialSharpeRatio(eta=0.01)

        self.action_space = spaces.Dict({
            'trader': spaces.MultiDiscrete([3] * self.n_assets),  # 0=hold, 1=buy, 2=sell per asset
            'portfolio_manager': spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        })

        # Przykład: okno cen (n_assets * window), pozycje, gotówka, akcje tradera, alokacja
        obs_len = self.n_assets * window_size + self.n_assets + 1 + self.n_assets + self.n_assets
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_len,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.total_portfolio = self.initial_cash
        self.position = np.zeros(self.n_assets)  # ilość akcji każdego aktywa
        self.prev_trader_action = np.zeros(self.n_assets)
        self.prev_allocation = np.zeros(self.n_assets)
        self.trader_action = np.zeros(self.n_assets)
        self.portfolio_value_history = []
        
        # Initialize visualization tracking arrays
        self.states_buy = [[] for _ in range(self.n_assets)]
        self.states_sell = [[] for _ in range(self.n_assets)]
        self.states_allocation = [[] for _ in range(self.n_assets)]
        self.shares_buy = [[] for _ in range(self.n_assets)]
        self.shares_sell = [[] for _ in range(self.n_assets)]
        self.asset_value_history = [[] for _ in range(self.n_assets)]
        
        self.asset_percentage_sell_history = [[] for _ in range(self.n_assets)]
        self.asset_percentage_buy_history = [[] for _ in range(self.n_assets)]
        
        return self._get_obs()

    def _get_obs(self):
        window_start = max(0, self.current_step - self.window_size)
        curr_prices = self.close_data.iloc[window_start:self.current_step].values  # shape: (window, n_assets)

        min_vals = self.close_data.min().values
        max_vals = self.close_data.max().values
        norm_prices = (curr_prices - min_vals) / (max_vals - min_vals + 1e-8)  # shape: (n_assets,)


        norm_actions = self.trader_action #/ 2.0
        for i in range(self.n_assets):
            if self.prev_trader_action[i] == 1:
                norm_actions[i] = 1.0 
            elif self.prev_trader_action[i] == 2:
                norm_actions[i] = -1.0
            else:
                norm_actions[i] = 0.0
        norm_actions = norm_actions[:, np.newaxis]
        
        last_prices = self.close_data.iloc[self.current_step - 1].values
        asset_values = self.position * last_prices
        total_value = self.cash + np.sum(asset_values)
        portfolio_shares = asset_values / (total_value + 1e-8)   # udział każdego aktywa
        cash_share = self.cash / (total_value + 1e-8)            # udział gotówki
        #portfolio_shares = np.concatenate([portfolio_shares, [cash_share]])
        #print(f"Portfolio shares: {portfolio_shares}")
        portfolio_shares = portfolio_shares[:, np.newaxis]
        cash_share = cash_share * np.ones((self.n_assets, 1))  # cash share as a column vector
        #print(cash_share.shape)
        obs = np.concatenate([norm_prices.T, norm_actions, portfolio_shares, cash_share], axis=1)  # shape: (n_assets, 2)
    
        return np.round(obs, 4)
    
    
    def step(self, action):
        executed = False
        self.trader_action = np.array(action['trader'])
        allocation = np.array(action['portfolio_manager'])
        curr_prices = self.close_data.iloc[self.current_step].values
        prev_prices = self.close_data.iloc[self.current_step - 1].values
        prev_value = self.cash + np.sum(self.position * prev_prices)
        transaction_cost_total = 0

        #print(allocation)
        # if np.any(np.isnan(allocation)) or np.any(allocation < 0) or np.any(allocation > 1):
        #     return self._get_obs(), -1.0, False, {'error': 'Invalid allocation'}

        MAX_ALLOCATION = 0.3  # 30% maksymalnie dla pojedynczego aktywa
        #allocation = np.clip(allocation, 0, MAX_ALLOCATION)
        
        current_allocations = []
        for i in range(self.n_assets):
            act = self.trader_action[i]
            alloc = allocation[i]
            #print(alloc)
            price = curr_prices[i]
            # Track allocation for each asset at each step
            self.states_allocation[i].append(alloc)
            
            if act == 1:  # BUY
                #alloc = max(0, min(alloc, MAX_ALLOCATION))  # limit allocation to max_allocation
                current_allocation = (self.position[i] * price) / prev_value
                allocation_left = max(0, MAX_ALLOCATION - current_allocation)

                #invest_amount = self.cash * alloc
                invest_amount = self.cash * allocation_left
                invest_amount = min(invest_amount + self.transaction_cost, self.cash)
                #invest_amount = min(allocation_left * prev_value, min(invest_amount, self.cash)) # nie można wydać więcej niż mamy gotówki
                current_allocations.append(invest_amount / prev_value)

                cost = invest_amount + self.transaction_cost
                shares = invest_amount / price
                cost = min(cost, self.cash)  
                
                
                self.position[i] += shares
                self.cash -= cost
                transaction_cost_total += self.transaction_cost
                executed = True
                
                # Track buy action
                self.states_buy[i].append(self.current_step)
                self.shares_buy[i].append(shares)
                self.asset_percentage_buy_history[i].append(shares * price / (prev_value + 1e-8))
                    
            elif act == 2:  # SELL
                shares_to_sell = self.position[i] * alloc
                shares_to_sell = min(shares_to_sell, self.position[i])  # nie można sprzedać więcej niż się posiada
                revenue = shares_to_sell * price
                self.position[i] -= shares_to_sell
                self.cash += revenue  # koszt transakcji pomijany przy sprzedaży
                executed = True
                    
                self.states_sell[i].append(self.current_step)
                self.shares_sell[i].append(shares_to_sell)
                self.asset_percentage_sell_history[i].append(shares_to_sell * price / (prev_value + 1e-8))
                
            else:
                # For hold actions, add 0 shares
                self.shares_buy[i].append(0)
                self.shares_sell[i].append(0)
                self.asset_percentage_sell_history[i].append(0)
                self.asset_percentage_buy_history[i].append(0)

            
            asset_value = self.position[i] * curr_prices[i]
            self.asset_value_history[i].append(asset_value)


        entropy_coeff = 0.2
        diversity_coeff = 0.2
        entropy = -np.sum(allocation * np.log(allocation + 1e-8))

        # kara za koncentrację
        concentration = np.sum(allocation**2)
        diversity_bonus = 1 - concentration  # 0 → całość w 1 aktywo, max=1 przy równym rozłożeniu

        #self.cash_vector = np.full(12, self.initial_cash / 12)
        curr_value = self.cash + np.sum(self.position * curr_prices)
        #curr_value = self.cash_vector + self.position * curr_prices
        self.total_portfolio = curr_value
        portfolio_return = (curr_value - prev_value) / (prev_value + 1e-8)

        entropy_coeff = 0.01
        entropy = -np.sum(allocation * np.log(allocation + 1e-8) + (1 - allocation) * np.log(1 - allocation + 1e-8))
        reward = np.dot(allocation, portfolio_return ) #+ entropy_coeff * entropy #+ diversity_coeff * diversity_bonus
        #reward = np.sum(reward)
        #reward = portfolio_return
        #reward = portfolio_return + self.cash_vector
        reward = np.clip(reward, -1.0, 1.0)
        #print(reward)

        asset_sharpes = []
        for i in range(self.n_assets):
            if len(self.asset_value_history[i]) > 1:
                values = np.array(self.asset_value_history[i])
                returns = np.diff(values) / (values[:-1] + 1e-8)
                if len(returns) > 1:
                    mean_r = np.mean(returns)
                    std_r = np.std(returns) + 1e-8
                    sharpe = mean_r / std_r
                    asset_sharpes.append(sharpe)
                else:
                    asset_sharpes.append(0.0)
            else:
                asset_sharpes.append(0.0)

        asset_sharpes = np.array(asset_sharpes)

        # Ważone Sharpe wg allocation (to zachęca do alokowania w aktywa z lepszym profilem ryzyko/zwrot)
        reward = np.dot(allocation, asset_sharpes)  + entropy_coeff * entropy #portfolio_return
        #reward = asset_sharpes
        #print(reward)
        # Klipowanie
        
        portfolio_values = np.array(self.portfolio_value_history)

        if len(portfolio_values) > 1:
            # log returns albo zwykłe zwroty procentowe
            portfolio_returns = np.diff(portfolio_values) / (portfolio_values[:-1] + 1e-8)

            if len(portfolio_returns) > 1:
                mean_r = np.mean(portfolio_returns)
                std_r = np.std(portfolio_returns) + 1e-8
                portfolio_sharpe = mean_r / std_r
            else:
                portfolio_sharpe = 0.0
        else:
            portfolio_sharpe = 0.0
        
        
   
        #print(f"Penalty: {penalty}, Allocation: {allocation}")
        reward = asset_sharpes# - penalty#* 100 #- penalty
        #reward = np.dot(allocation, portfolio_sharpe)
        reward = np.dot(allocation, asset_sharpes) #+ entropy_coeff * entropy #+ diversity_coeff * diversity_bonus
        reward = portfolio_sharpe #* 100
        reward = np.clip(reward, -1.0, 1.0) 
        print(f"Step {self.current_step} | Reward: {reward} | Portfolio Sharpe: {portfolio_sharpe:.4f} | Cash: {self.cash:.2f} | Total val: {curr_value:.2f}")
        
        #print(f"Current step {self.current_step} | Reward {reward} | Allocations {allocation}")
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

        window_start = max(0, self.current_step - self.window_size)
        curr_prices = self.close_data.iloc[window_start:self.current_step].values  # shape: (window, n_assets)
        min_vals = self.close_data.min().values
        max_vals = self.close_data.max().values

        norm_prices = (curr_prices - min_vals) / (max_vals - min_vals + 1e-8)  # shape: (n_assets,)

        return np.array(norm_prices).T
    
    def get_visualization_data(self, asset_index):
        """
        Get visualization data for a specific asset
        
        Args:
            asset_index: Index of the asset (0 to n_assets-1)
            
        Returns:
            dict: Dictionary containing visualization data
        """
        if asset_index >= self.n_assets:
            raise ValueError(f"Asset index {asset_index} out of range. Max index: {self.n_assets-1}")
            
        return {
            'prices': self.close_data[self.asset_names[asset_index]].values,
            'buy_points': self.states_buy[asset_index],
            'sell_points': self.states_sell[asset_index],
            'allocations': self.states_allocation[asset_index],
            'shares_buy': self.shares_buy[asset_index],
            'shares_sell': self.shares_sell[asset_index]
        }
    
    def get_all_visualization_data(self):
        """
        Get visualization data for all assets
        
        Returns:
            dict: Dictionary with asset names as keys and visualization data as values
        """
        all_data = {}
        for i, asset_name in enumerate(self.asset_names):
            all_data[asset_name] = self.get_visualization_data(i)
        return all_data

