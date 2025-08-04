import matplotlib.pyplot as plt
import torch
import numpy as np

def evaluate_steps(env, model, device="cuda:0", OHCL = False):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    action = 0
    while not done:
        # konwersja stanu na tensora
        #state_tensor = torch.tensor(state, dtype=torch.float32, device=device)#.unsqueeze(0)
        if not OHCL:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)


        with torch.no_grad():
            if OHCL:
                #state_tensor, position_ratio = state_tensor
                state = (state[0], np.full_like(state[1], 0.5))
                action = model.get_action_target(state )
            else:
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        #print(f"Krok: {steps}, Akcja: {action}, Nagroda: {reward:.2f}, Łączny zysk: {env.total_profit:.2f}")

    #print(state_tensor)
    return total_reward


def render_env(env, title_suffix="", OHCL = False):
    if OHCL:
        prices = env.ohlc_data[:,3]
    else:
        prices = env.data
        
    buy_points = env.states_buy
    sell_points = env.states_sell
    buy_points = [i for i in env.states_buy if i < len(prices)]
    sell_points = [i for i in env.states_sell if i < len(prices)]
    profit = env.total_profit #+ np.sum(test_env.data[-1] - test_env.inventory) 
    
    plt.figure(figsize=(14, 6))
    plt.plot(prices, label='Cena', linewidth=1.5)

    if buy_points:
        plt.scatter(buy_points, [prices[i] for i in buy_points],
                    color='green', marker='^', label='Kup', s=100)
    if sell_points:
        plt.scatter(sell_points, [prices[i] for i in sell_points],
                    color='red', marker='v', label='Sprzedaj', s=100)
    
    
    plt.title(f'Działania agenta {title_suffix} | Łączny zysk: {profit:.2f}')
    plt.axvline(x = 48, color = 'red', label = 'Początek okna czasowego')
    plt.xlabel('Krok')
    plt.ylabel('Cena')
        
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    
def render_env_ddpg(env, title_suffix="", OHCL=False, window_size=96):
    if OHCL:
        prices = env.ohlc_data[:, 3]
    else:
        prices = env.data

    profit = env.total_profit

    # Bezpieczne filtrowanie punktów
    buy_points = [i for i in env.states_buy if i < len(prices)]
    sell_points = [i for i in env.states_sell if i < len(prices)]
    paired_points = zip(buy_points[:len(sell_points)], sell_points)

    profits = []
    for buy_idx, sell_idx in paired_points:
        buy_price = prices[buy_idx]
        sell_price = prices[sell_idx]
        profit = sell_price - buy_price
        profits.append(profit)

    realized_profit = sum(profits)

    # Dopasowanie alokacji do dostępnych punktów kupna
    print(f"Alokacje: {env.allocations}")
    allocations = np.abs(env.allocations[:len(buy_points)])
    sizes = [150 * a for a in allocations]  # Skalowanie wielkości markerów

    plt.figure(figsize=(14, 6))
    plt.plot(prices, label='Cena', linewidth=1.5)

    if buy_points:
        plt.scatter(
            buy_points,
            [prices[i] for i in buy_points],
            color='green',
            marker='^',
            label='Kup',
            s=sizes
        )

    if sell_points:
        plt.scatter(
            sell_points,
            [prices[i] for i in sell_points],
            color='red',
            marker='v',
            label='Sprzedaj',
            s=100
        )

    plt.title(f'Działania agenta {title_suffix} | Łączny zysk: {env.total_profit:.2f}')
    plt.axvline(x=window_size, color='red', label='Początek okna czasowego')
    plt.xlabel('Krok')
    plt.ylabel('Cena')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

