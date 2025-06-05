import pandas as pd
import torch
import logging


from source.IB_connector import retrive_market_data
from torch.utils.data import Dataset, DataLoader
from source.LSTMdataset import WindowDataset, WindowGenerator
from LSTMmodel import train_LSTM, save_model
from enviroments import TimeSeriesEnvFuturePredict
from rl_agent import DQNAgent, train_episode, save_dqn_agent
from tqdm import tqdm
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)


def get_training_data(ticker):
    training_set = retrive_market_data([ticker], duration = "8 m", time_interval = "15 mins")
    training_set = training_set[ticker]
    training_set['Volume'] = training_set['Volume'].astype(float)
    training_set['Date'] = pd.to_datetime(training_set['Date'].str.replace(' US/Eastern',''),format="%Y%m%d %H:%M:%S")
    
    col_select = ['Open','High','Low','Close','Volume']
    training_set = training_set[col_select]
    training_set.dtypes
    CLOSE_INDEX = col_select.index('Close')
    
    n = len(training_set)
    train_lstm = training_set[0:int(n*0.4)]
    val_lstm = training_set[int(n*0.4):int(n*0.5)]
    
    rl_train = training_set[int(n*0.5):int(n*0.9)]
    rl_test = training_set[int(n*0.9):]

    dataset_mean = training_set.mean()
    dataset_std = training_set.std()
    
    return {"train_lstm": train_lstm,
            "val_lstm": val_lstm,
            "rl_train": rl_train,
            "rl_test": rl_test,
            "dataset_mean": dataset_mean,
            "dataset_std": dataset_std,
            "CLOSE_INDEX": CLOSE_INDEX}

def create_data_loader(norm_train,norm_val, WINDOW_SIZE=48, OUT_STEPS=16):
    w1 = WindowGenerator(input_width=WINDOW_SIZE, label_width=OUT_STEPS, shift=OUT_STEPS, 
                        train_df=norm_train, val_df=norm_val, test_df=norm_val, 
                        label_columns=['Close'])
    # Stworzenie datasetu:
    train_dataset = w1.make_dataset(norm_train.values)  # Zakładając, że train_df to np. pandas.DataFrame
    val_dataset = w1.make_dataset(norm_val.values)  # Zakładając, że train_df to np. pandas.DataFrame

    # DataLoader:
    BATCH_SIZE = 512 # 256 * 2 = 512
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_loader, val_loader

def train_agent(ticker= "AAPL"):
    OUT_STEPS = 16
    WINDOW_SIZE = 48
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #ticker = "AAPL"
    
    
    data = get_training_data(ticker)
    logging.info(f"Training data for {ticker} retrieved successfully.")
    
    
    train_norm = (data["train_lstm"] - data["dataset_mean"]) / data["dataset_std"]
    val_norm = (data["val_lstm"] - data["dataset_mean"]) / data["dataset_std"]
    lstm_train_loader, lstm_val_loader = create_data_loader(train_norm, val_norm, WINDOW_SIZE, OUT_STEPS)
    
    num_epochs = 1500
    learning_rate = 0.014
    num_features = 5
    input_size = num_features
    
    logging.info(f"Starting LSTM training with {num_epochs} epochs and learning rate {learning_rate}.")
    LSTM_model, history = train_LSTM(lstm_train_loader, input_size, lstm_units=32, out_steps=OUT_STEPS, num_features=num_features,
              num_epochs=num_epochs, learning_rate=learning_rate, device=device, show_loss_every=500)
    
    model_lstm_name_ = f"models/trained_lstm_{ticker}_epoch{num_epochs}.pth"
    save_model(LSTM_model, model_lstm_name_)
    logging.info(f"LSTM model trained and saved as {model_lstm_name_}.")
    
    rl_train = data["rl_train"]
    env = TimeSeriesEnvFuturePredict(data=rl_train['Close'].values, lstm=LSTM_model,lstm_data=rl_train.values,device=device,
                                 train_std = data["CLOSE_INDEX"],train_mean= data["CLOSE_INDEX"], window_size=48, future_size=OUT_STEPS)
    

    agent = DQNAgent(observation_space=env.observation_space.shape[0], action_space=env.action_space.n)
    
    epsilon = 0.95
    EPISODES = 400
    logging.info(f"Starting training for {EPISODES} episodes with initial epsilon {epsilon}.")
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        train_episode(agent,env, episode, epsilon)

    logging.info(f"Starting training for {EPISODES} episodes with initial epsilon {epsilon}.")
    model_rl_name_ = f"models/trained_agent_{ticker}_episode{EPISODES}.pth"
    save_dqn_agent(agent, model_rl_name_)
    logging.info(f"DQN agent trained and saved as {model_rl_name_}.")
    
    
    
    
if __name__ == "__main__":
    train_agent()