import pandas as pd
import threading
import time
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

    def historicalData(self, reqId, bar):
        ticker = self.reqId_to_ticker[reqId]  # Pobieramy ticker po reqId
        #print(f"Date: {bar.date}, Open: {bar.open}, High: {bar.high}, Low: {bar.low}, Close: {bar.close}")
        #self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])
        self.data[ticker].append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])
        
    def get_data_df(self) -> pd.DataFrame:
        dfs = {ticker: pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close", "Volume"]) for ticker, data in self.data.items()}
        return dfs


    
    def create_contract(self, symbol):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract
    
    def get_batch_market_data(self, tickers,duration = "3 M", time_interval = "30 mins"):
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
            time.sleep(3)  # Mały odstęp, by uniknąć rate limit
    
    

def retrive_market_data(tickers= ["AAPL", "MSFT", "TSLA"], duration = "3 M", time_interval = "30 mins"):
    app = IBapi()
    app.connect("127.0.0.1", 7497, clientId=1)

    # Wątek obsługujący API
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    time.sleep(1)
    app.get_batch_market_data(tickers, duration = duration, time_interval =time_interval)

    dfs = app.get_data_df()

    app.disconnect()

    return dfs


