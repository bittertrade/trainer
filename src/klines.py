"""
API-GETTER for binance
use for getting historical data from binance
"""
from binance.api import json
from binance.spot import Spot
from datetime import datetime
from shared import creds

api_key = creds.api_key
api_secret = creds.api_secret
spot = Spot(api_key=api_key, api_secret=api_secret)


# get data from binance
def grab_history(coin, interval="1m", limit=1000):

    if limit > 1000:
        tmp = grab_history(coin, interval, limit=limit-1000)
        tmp2 = spot.klines(symbol=coin, interval=interval, limit=1000, endTime=tmp[0][0])
        tmp2.pop(-1)  # this would be a double element so we need to rmeove it from one of the lists
        # we return tmp2 added onto the front of tmp, since tmp would show more recent history first
        return tmp2 + tmp
    else:
        return spot.klines(symbol=coin, interval=interval, limit=limit)


def write_klines_json(klines, filename):
    with open(f"{filename}.json", "w") as f:
        f.write("[")
        i=0
        for kline in klines:
            # convert epoch to datetime
            open_time = datetime.fromtimestamp(kline[0]/1000).strftime("%Y-%m-%d %H:%M:%S")
            line = {
                "open_time": open_time,
                "open_time_epoch": kline[0],
                "open": kline[1],
                "high": kline[2],
                "low":  kline[3],
                "close": kline[4],
                "volume":   kline[5],
                "close_time":   kline[6],
                "quote_asset_volume":   kline[7],
                "number_of_trades":     kline[8],
                "taker_buy_base_asset_volume":  kline[9],
                "taker_buy_quote_asset_volume":     kline[10],
                "ignore":   kline[11]
            }
            f.write(json.dumps(line, indent=4) + ",\n")
        f.write("]")

def write_klines_csv(klines, filename:str):
    if not filename.endswith(".csv"):
        filename += ".csv"
    with open(f"{filename}", "w") as f:
        f.write("open_time,open_time_epoch,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ignore\n")
        for kline in klines:
            #convert epoch to datetime
            open_time = datetime.fromtimestamp(kline[0]/1000).strftime("%Y-%m-%d %H:%M:%S")
            line = ",".join([str(k) for k in kline])
            f.write(f"{open_time},{line}\n")
    return filename
