from binance.api import json

coins = ["BTCUSDT", "LTCUSDT", "DOGEUSDT", "ETHUSDT"]
interval = "1h"
limit = 10000
training_interval = 120
epoch = 500
# column = ['close', 'close,volume', 'close,volume,number_of_trades']
column = 'close,volume,number_of_trades'
col_len = 3
target_interval = 1


gpu_mem = 6000
# batch_size = gpu_mem // (training_interval*col_len)
batch_size = 128

with open("sc.json", "w") as f:
    f.write("[\n")

    for coin in coins:
        tmp = {
            "coin_name": coin,
            "interval": interval,
            "limit": limit,
            "split_size": 0.8,
            "training_intervals": training_interval,
            "step": 1,
            "shuffle": False,
            "columns": column,
            "batch_size": batch_size,
            "epochs": epoch,
            "target_interval": target_interval,
            "auto_save": True
        }
        f.write(json.dumps(tmp, indent=4))
        f.write(",\n")
    f.write("\n]")
