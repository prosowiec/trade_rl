import pandas as pd
import threading
import time
import random
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
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
        
        self.nextOrderId = None
        self.nextOrderId_event = threading.Event()
        self.order_status = {}
        self.exec_reports = {}
        self.open_orders = {}

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
    
    def nextValidId(self, orderId: int):
        self.nextOrderId = orderId
        self.nextOrderId_event.set()
        
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId,
                    parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        self.order_status[orderId] = {
            "status": status,
            "filled": filled,
            "remaining": remaining,
            "avgFillPrice": avgFillPrice,
            "lastFillPrice": lastFillPrice
        }

    # callback: informacje o otwartym zleceniu
    def openOrder(self, orderId, contract, order, orderState):
        self.open_orders[orderId] = {
            "contract": contract,
            "order": order,
            "orderState": orderState
        }

    # callback: wykonania
    def execDetails(self, reqId, contract, execution):
        oid = execution.orderId
        self.exec_reports.setdefault(oid, []).append({
            "execId": execution.execId,
            "time": execution.time,
            "shares": execution.shares,
            "price": execution.price,
            "permId": execution.permId
        })
        
    def _create_order(self, action: str, quantity: float, order_type: str = "MKT", limit_price: float = None, tif: str = "DAY") -> Order:
        """
        action: "BUY" or "SELL"
        order_type: "MKT" or "LMT"
        limit_price: required if order_type == "LMT"
        tif: time in force, e.g. "DAY", "GTC"
        """
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = order_type
        if order_type == "LMT":
            if limit_price is None:
                raise ValueError("limit_price required for LMT orders")
            order.lmtPrice = limit_price
        order.tif = tif
        return order

    def place_order(self, contract: Contract, order: Order, wait_for_next_id: bool = True) -> int:
        """
        Składa zlecenie. Zwraca orderId.
        Jeśli nextOrderId nie jest jeszcze ustawione, czeka (do ~5s) — możesz zwiększyć timeout wedle potrzeb.
        """
        if wait_for_next_id and self.nextOrderId is None:
            # poczekaj moment na nextValidId
            self.nextOrderId_event.wait(timeout=5)
        if self.nextOrderId is None:
            raise RuntimeError("nextOrderId not set (no nextValidId received) — upewnij się, że połączyłeś się i uruchomiłeś pętlę API")

        oid = self.nextOrderId
        # inkrementujemy nextOrderId aby kolejny order miał unikalne id
        self.nextOrderId += 1

        self.placeOrder(oid, contract, order)
        return oid

    # krótsze wrappery:
    def buy_market(self, symbol: str, qty: float):
        c = self.create_contract(symbol)
        order = self._create_order("BUY", qty, order_type="MKT")
        return self.place_order(c, order)

    def sell_market(self, symbol: str, qty: float):
        c = self.create_contract(symbol)
        order = self._create_order("SELL", qty, order_type="MKT")
        return self.place_order(c, order)

    def buy_limit(self, symbol: str, qty: float, price: float):
        c = self.create_contract(symbol)
        order = self._create_order("BUY", qty, order_type="LMT", limit_price=price)
        return self.place_order(c, order)

    def sell_limit(self, symbol: str, qty: float, price: float):
        c = self.create_contract(symbol)
        order = self._create_order("SELL", qty, order_type="LMT", limit_price=price)
        return self.place_order(c, order)

    
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

def retrieve_positions(app : IBapi):
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

def retrieve_account_and_portfolio(app : IBapi, account=""):  
    app = IBapi()
    app.connect("127.0.0.1", 7497, clientId=random.randint(1, 9999))

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    time.sleep(1)
    app.reqAccountUpdates(True, account)

    app.portfolio_ready.wait(timeout=5)
    app.disconnect()

    return pd.DataFrame(app.portfolio), pd.Series(app.account_values)