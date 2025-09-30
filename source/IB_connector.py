import pandas as pd
import threading
import time
import random
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from source.start_IB import open_ib_con, check_if_app_exist

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}  # Lista do przechowywania danych
        self.reqId_to_ticker = {}
        self.appStarted = True if check_if_app_exist("Interactive Brokers") else open_ib_con()
        
        self.positions = []
        self.positions_ready = threading.Event()
        
        self.portfolio = []
        self.account_values = {}
        self.portfolio_ready = threading.Event()

    def historicalData(self, reqId, bar):
        ticker = self.reqId_to_ticker[reqId]  # Pobieramy ticker po reqId
        #print(f"Date: {bar.date}, Open: {bar.open}, High: {bar.high}, Low: {bar.low}, Close: {bar.close}")
        #self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])
        self.data[ticker].append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])
        
    def get_data_df(self) -> pd.DataFrame:
        dfs = {ticker: pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close", "Volume"]) for ticker, data in self.data.items()}
        return dfs
    
    def position(self, account, contract, position, avgCost):
        self.positions.append({
            "account": account,
            "symbol": contract.symbol,
            "secType": contract.secType,
            "currency": contract.currency,
            "position": position,
            "avgCost": avgCost
        })
        
    def updatePortfolio(self, contract, position, marketPrice, marketValue,
                        averageCost, unrealizedPNL, realizedPNL, accountName):
        self.portfolio.append({
            "account": accountName,
            "symbol": contract.symbol,
            "secType": contract.secType,
            "currency": contract.currency,
            "position": position,
            "avgCost": averageCost,
            "marketPrice": marketPrice,
            "marketValue": marketValue,
            "unrealizedPNL": unrealizedPNL,
            "realizedPNL": realizedPNL
        })

    def updateAccountValue(self, key, val, currency, accountName):
        # np. key = 'NetLiquidation', 'CashBalance', 'AvailableFunds', ...
        self.account_values[key] = val

    def accountDownloadEnd(self, account):
        self.portfolio_ready.set()
        
    def create_contract(self, symbol):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract
    
    def get_batch_market_data(self, tickers,duration = "3 M", time_interval = "30 mins", sleep_time=5):
        for i, ticker in enumerate(tickers):
            self.data[ticker] = []  # Tworzymy pustą listę na dane
            self.reqId_to_ticker[i] = ticker  # Mapujemy reqId -> ticker
            self.reqHistoricalData(
                reqId=i, 
                contract=self.create_contract(ticker),
                endDateTime="",  
                durationStr=duration,  
                barSizeSetting=time_interval,  
                whatToShow="ADJUSTED_LAST",  
                useRTH=1,  
                formatDate=1,  
                keepUpToDate=False,  
                chartOptions=[]
            )
            time.sleep(sleep_time)  # Mały odstęp, by uniknąć rate limit
    
    

def retrive_market_data(tickers= ["AAPL", "MSFT", "TSLA"], duration = "3 M", time_interval = "30 mins", sleep_time=5):
    app = IBapi()
    app.connect("127.0.0.1", 7497, clientId=random.randint(1, 9999))

    # Wątek obsługujący API
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    time.sleep(1)
    app.get_batch_market_data(tickers, duration = duration, time_interval =time_interval, sleep_time=sleep_time)

    dfs = app.get_data_df()

    app.disconnect()

    return dfs

def retrieve_positions():
    app = IBapi()
    app.connect("127.0.0.1", 7497, clientId=random.randint(1, 9999))

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    time.sleep(1)  # daj IB czas na inicjalizację
    app.reqPositions()

    # czekamy aż callback positionEnd() ustawi event
    app.positions_ready.wait(timeout=5)
    app.disconnect()
    

    return pd.DataFrame(app.positions)

def retrieve_account_and_portfolio(account=""):  
    app = IBapi()
    app.connect("127.0.0.1", 7497, clientId=random.randint(1, 9999))

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    time.sleep(1)
    app.reqAccountUpdates(True, account)

    app.portfolio_ready.wait(timeout=5)
    app.disconnect()

    return pd.DataFrame(app.portfolio), pd.Series(app.account_values)