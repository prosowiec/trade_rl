# import .transform
# import .predictions
from IB_connector import retrive_market_data

import pandas as pd 
import numpy as np

training_set_aapl = retrive_market_data(["AAPL"])
training_set = training_set_aapl['AAPL']

training_set['Date'] = pd.to_datetime(training_set['Date'].str.replace(' US/Eastern',''),format="%Y%m%d %H:%M:%S")

print(training_set)