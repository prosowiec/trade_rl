
class Tickers:
    def __init__(self):
        self.TICKERS_penny = [
            "WVE",   # Wave Life Sciences Ltd :contentReference[oaicite:0]{index=0}
            "ATUS",  # Altice USA Inc :contentReference[oaicite:1]{index=1}
            "CIFR",  # Cipher Mining Inc :contentReference[oaicite:2]{index=2}
            "LAZR",  # Luminar Technologies Inc :contentReference[oaicite:3]{index=3}
            "AAOI",  # Applied Optoelectronics Inc :contentReference[oaicite:4]{index=4}
            "IREN",  # Iris Energy Ltd :contentReference[oaicite:5]{index=5}
            "EXK",   # Endeavour Silver Corp :contentReference[oaicite:6]{index=6}
            "LAC",   # Lithium Americas Corp Newco :contentReference[oaicite:7]{index=7}
            "CTMX",  # Cytomx Therapeutics Inc :contentReference[oaicite:8]{index=8}
            "NB"     # Niocorp Developments Ltd :contentReference[oaicite:9]{index=9}
        ]

        self.BIG_TICKERS =  ['NVDA', 'MSFT', 'AAPL', 'GOOG', 'AMZN',
                'META', 'AVGO', 'TSLA', 'JPM',
                'WMT', 'V', 'ORCL', 'LLY', 'NFLX',
                'MA', 'XOM', 'JNJ'  
        ]
        
        self.TRASH_TICKERS = ["CLFD","IRS","BRC","TBRG","CCNE","CVEO"]

        self.groups ={
            'PENNY': self.TICKERS_penny,
            'BIG': self.BIG_TICKERS,
            'TRASH': self.TRASH_TICKERS
        }
            
    def get_current_tickers(self):
        return self.TICKERS_penny
    
    def get_tickers(self, group_name = 'TICKERS_penny'):
        
        return self.groups[group_name]

if __name__ == "__main__":
    pass