from sqlalchemy import create_engine,func
from sqlalchemy.dialects.sqlite import insert  
from sqlalchemy.orm import declarative_base, sessionmaker
from utils.db_models import Base, StockData, ModelData, TrainingLogs  # Dodaj plik models.py poniżej\
import pandas as pd
import os

Base = declarative_base()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "trade.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    
def upload_stock_data(training_set, db = SessionLocal()):
    n = len(training_set)

    ticker = training_set["ticker"].iloc[0]

    # Usuń istniejące dane dla tego tickera
    db.query(StockData).filter(StockData.ticker == ticker).delete()
    db.commit()
    
    
    train_df = training_set[0:int(n * 0.6)].copy()
    val_df = training_set[int(n * 0.6):int(n * 0.8)].copy()
    test_df = training_set[int(n * 0.8):].copy()

    train_df["train_split"] = "train"
    val_df["train_split"] = "validation"
    test_df["train_split"] = "test"

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Wstawiamy dane do bazy
    for _, row in full_df.iterrows():
        record = StockData(
            ticker=row["ticker"],
            date=str(row["Date"]),  # upewnij się, że jest to string lub datetime
            open=row["Open"],
            high=row["High"],
            low=row["Low"],
            close=row["Close"],
            volume=row["Volume"],
            train_split=row["train_split"]
        )
        db.add(record)

    db.commit()
    
def read_stock_data(ticker: str, db = SessionLocal()):
    # Pobierz dane z bazy dla danego tickera
    records = db.query(StockData).filter(StockData.ticker == ticker).all()

    if not records:
        return None, None, None, None

    # Index(['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker'], dtype='object')
    data = pd.DataFrame([{
        "ticker": r.ticker,
        "date": r.date,
        "open": r.open,
        "high": r.high,
        "low": r.low,
        "close": r.close,
        "volume": r.volume,
        "train_split": r.train_split
    } for r in records])

    # Podziel na 4 zestawy
    train_df = data[data["train_split"] == "train"].reset_index(drop=True)
    val_df = data[data["train_split"] == "validation"].reset_index(drop=True)
    test_df = data[data["train_split"] == "test"].reset_index(drop=True)

    return train_df, val_df, test_df

def upsert_training_logs(reward_all, evaluate_rewards, test_rewards,ticker, db=SessionLocal()):
    training_log_df = pd.DataFrame({'trainRewards' : reward_all,'evaluateRewards' : evaluate_rewards,'testRewards' : test_rewards })
    training_log_df['ticker'] = ticker
    training_log_df['episode'] = training_log_df.index
    
    db.query(TrainingLogs).filter(TrainingLogs.ticker == ticker).delete()
    db.commit()

    
    # dodanie nowych rekordów
    db.bulk_insert_mappings(TrainingLogs, training_log_df.to_dict(orient='records'))
    
    db.commit()    
    
    return training_log_df