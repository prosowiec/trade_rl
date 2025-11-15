from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class StockData(Base):
    __tablename__ = "stock_data"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)  # Ticker symbol of the stock
    date = Column(String)  # lub Date, jeśli masz datetime
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    train_split = Column(String)  # train, validation, test
    
class ModelData(Base):
    __tablename__ = "model_data"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, unique=True, index=True)
    model_path = Column(String)  # Path to the saved model file
    training_date = Column(String)  # Date when the model was trained
    accuracy = Column(Float)  # Model accuracy or other metrics
    additional_info = Column(String)  # Any additional information about the model
    
class TrainingLogs(Base):
    __tablename__ = "training_logs"
    id = Column(Integer, primary_key=True, index=True)
    trainRewards = Column(Float)
    evaluateRewards = Column(Float)
    testRewards = Column(Float)
    ticker = Column(String)
    episode = Column(Integer)

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String, nullable=False)           # BUY / SELL / HOLD
    allocation = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    position = Column(Float, nullable=False)
    asset_name = Column(String, nullable=False)
    executed = Column(Boolean, default=True)
    timestamp = Column(DateTime, default=datetime.timezone.utc)

class TickersList(Base):
    __tablename__ = "tickers_list"

    id = Column(Integer, primary_key=True, index=True)
    group_name = Column(String, index=True)
    ticker = Column(String, unique=False)
    active = Column(Boolean, default=True)