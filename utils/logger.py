import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class NoSendingIBMessagesFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not (
            "SENDING placeOrder" in msg or
            "SENDING reqHistoricalData" in msg or
            "ANSWER updateAccountTime" in msg 
        )
        
def start_logger():
    logger = logging.getLogger()
    logger.addFilter(NoSendingIBMessagesFilter())

    ib_loggers = [
        logging.getLogger("ibapi"),
        logging.getLogger("ibapi.client"),
        logging.getLogger("ibapi.wrapper"),
        logging.getLogger("ibapi.connection"),
    ]

    for ib_logger in ib_loggers:
        ib_logger.addFilter(NoSendingIBMessagesFilter())



if __name__ == "__main__":
    logging.info("Logger is configured and running.")