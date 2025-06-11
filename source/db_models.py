from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base

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
    
