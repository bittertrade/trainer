"""
File for running the app
"""
from genericpath import exists
import math
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from coinData import *
from trainer import *

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import time

def run():
    running = True
    #Print a welcomme screen using ascii art
    print("Welcome\n Please select an option from the menu below:")
    while running:
        valid_commands = ['1', '2', '3', '4']
        print("1. Create a new model")
        print("2. Load an existing model for testing")
        print("3. Exit")
        print("4. Run from a script")

        #Get user input
        user_input = input("Enter your choice: ")
        #Check if the user input is valid
        if user_input in valid_commands:
            if user_input == '1':
                #Create a new model
                create_new_model()
            elif user_input == '2':
                #Load an existing model
                load_and_test()
            elif user_input == '3':
                #Exit the program
                running = False
            elif user_input == '4':
                #Run from a script
                prompt_for_script()

        else: 
            print("\n"*5)
            print("Invalid command, only displayed options are valid")

def create_new_model():
    #Get the user input
    print("This is for Coin selection and creation")
    print("Please enter your coin in a valid string format")
    print("Example: BTCUSDT\n")
    coin_name =         input_or_default(input("Enter your coin: ('BTCUSDT'): "), "BTCUSDT")
    interval =          input_or_default(input("Enter your interval: ('1h'): "), "1h")
    limit =             int(input_or_default(input("Enter your limit: ('1000'): "), "1000"))
    split_size =        float(input_or_default(input("Enter train/test split size: ('0.8'): "), "0.8"))
    training_intervals =int(input_or_default(input("Enter length of desired interval length: ('60'): "), "60"))
    step =              int(input_or_default(input("Enter step size(skip intervals between training sets): ('1'): "), "1"))
    shuffle =           input_or_default(input("Enter shuffle size: ('y/N): "), "N")
    target_interval =   int(input_or_default(input("Enter target interval (e.g. 1 min in the future/ 5 min in the future): ('1'): "), "1"))
    columns = input_or_default(input("Enter your columns: ('close,volume,number_of_trades'): "), "close,volume,number_of_trades")
    if shuffle.lower() == 'y':
        shuffle = True
    else:
        shuffle = False
    print("Please enter the desired training parameters")
    batch_size = int(input_or_default(input("Enter your batch size: ('128'): "), "128"))
    epochs = int(input_or_default(input("Enter your epochs: ('1'): "), "1"))
    auto_save = input_or_default(input("Auto save model? (Y/n): "), "y")
    if auto_save.lower() == "y":
        auto_save = True
    else:
        auto_save = False

    params = {
        'coin_name': coin_name,
        'interval': interval,
        'limit': limit,
        'split_size': split_size,
        'training_intervals': training_intervals,
        'step': step,
        'shuffle': shuffle,
        'columns': columns,
        'batch_size': batch_size,
        'epochs': epochs,
        'target_interval': target_interval,
        'auto_save': auto_save
    }
    train_model_from_script(params)

def input_or_default(input, default):
    if input == "":
        return default
    else:
        return input

"""
script must be a dict with the following keys:
    {
        "coin_name": "BTCUSDT",
        "interval": "1h",
        "limit": 1000,
        "split_size": 0.8,
        "training_intervals": 60,
        "step": 1,
        "shuffle": False,
        "columns": ["close"],
        "batch_size": 1,
        "epochs": 1,
        'target_interval': 1,
        "auto_save": True
    }
"""

def prompt_for_script():
    #verify that the file exist
    print("Please enter the path to your script")
    print("Example: ./scripts/script1.json")
    script_path = input("Enter your script path: ")
    script = []
    try:
        with open(script_path, 'r') as f:
            script = json.load(f)
    except:
        print("Invalid script path")
        return

    for s in script:
        print('Training model for coin: ' + s['coin_name'])
        train_model_from_script(s)


def train_model_from_script(params):
    coin_name =         params['coin_name']
    interval =          params['interval']
    limit =             params['limit']
    split_size =        params['split_size']
    training_intervals= params['training_intervals']
    step =              params['step']
    shuffle =           params['shuffle']
    columns =           params['columns']
    batch_size =        params['batch_size']
    epochs =            params['epochs']
    target_interval =   params['target_interval']
    auto_save =         params['auto_save']
    photo_save = "+".join(["./plots/", coin_name, interval, str(training_intervals)])

    #Get the coin data
    #check to see if the current csv file already exist, and has at least the desired limit
    coin_save = fetch_write_coin(coin_name, interval, limit)
    #Prepare the coin data for model creation
    klines = load_klines_from_csv(coin_save)
    columns = columns.split(',')
    klines = klines.filter(columns)
    klines = klines.values

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    x_data = x_scaler.fit_transform(klines[:, :])
    y_data = y_scaler.fit_transform(klines[:, :1])#The first column should be the close column

    x_data, y_data = section_data(x_data, y_data, training_intervals, step, target_interval)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=split_size, shuffle=shuffle)

    del klines
    del x_data
    del y_data

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = y_scaler.inverse_transform(y_test)

    plots_location = f"./plots/{coin_name}/{interval}+{target_interval}/{training_intervals}/{'+'.join(columns)}/{time.time()}/"
    model_location = f"./saved_models/{coin_name}/{interval}+{target_interval}/{training_intervals}/{'+'.join(columns)}/"

    if not os.path.exists(plots_location):
        os.makedirs(plots_location)

    #check if location exist and if there is an existing model there, if the model exists, load it
    #since the folder path is all the relevant model dimensions (for now) we can load it from there
    if os.path.exists(model_location):
        #find a model at the model_location
        print("Existing model exists for these dimensions, if you think is is an error please quit and delete the model")
        print("Loading existing model")
        model = load_model(model_location)
    else:
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(200, return_sequences=True))
        model.add(LSTM(200, return_sequences=True))
        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        os.makedirs(model_location)
        model.save(model_location)

    start_time = time.time()
    with open(plots_location + "meta.json", "a") as f:
        f.write('{\n"header":')
        json.dump({
            "coin_name": coin_name,
            "interval": interval,
            "limit": limit,
            "split_size": split_size,
            "training_intervals": training_intervals,
            "step": step,
            "shuffle": shuffle,
            "columns": columns,
        }, f)
        f.write(',\n"prediction_data":[\n')
    print(f"Beginning training for {coin_name} {interval} {training_intervals} {columns}")   

    best_rsme = 999_999_999
    for i in range(epochs):
        # save_file_info = (coin_name, interval, training_intervals)
        # model, runtime = train_model(x_train, y_train, save_file_info, batch_size, 1, auto_save)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1)
        #save on the 10th epoch, then 9th, then 8th etc... until every epoch
        if i > 5:
        # if i == 0 or (i+1) % 5 == 0 or i == epochs - 1:
            predictions = model.predict(x_test)
            predictions = y_scaler.inverse_transform(predictions)

            rsme = np.sqrt(np.mean((predictions - y_test)**2))
            photo_save = f"{plots_location}{time.strftime('%Y%m%d-%H%M%S')}.png"
            csvloc = plots_location + f"{time.strftime('%Y%m%d-%H%M%S')}.csv"

            title = f"Prediction Plot for '{coin_name}' every '{interval}' looking at previous '{training_intervals}' intervals trained data from '{columns}'. Epochs: {i+1}, Batch Size: {batch_size}"

            # plt.style.use("dark_background")
            plt.figure(figsize=(16, 8))
            plt.title(title)
            plt.xlabel("Date", fontsize=18)
            plt.ylabel("Close Price", fontsize=18)
            plt.plot(y_test, label='Train')
            plt.plot(predictions, label='Predicted')
            plt.legend(['Actual', 'Predicted'], loc='lower right')
            plt.savefig(photo_save)
            plt.close()

            difference = y_test - predictions

            df = pd.DataFrame({"actual": y_test.flatten(), "predicted": predictions.flatten(), "difference": difference.flatten()})
            df.to_csv(csvloc)

            metadata = {
                "rsme": rsme,
                "epochs": i+1,
                "runtime": time.time() - start_time,
                "prediction_data": csvloc,
                "photo_save": photo_save
            }
            with open(plots_location + "meta.json", "a") as f:
                json.dump(metadata, f)
                f.write(",\n")

            if auto_save and rsme < best_rsme:
                print("New best model found, saving")
                model.save(model_location)
                best_rsme = rsme
                
    with open(plots_location + "meta.json", "a") as f:
        f.write("    ]\n}")

def load_and_test():
    coin_name =         input_or_default(input("Enter your coin: ('BTCUSDT'): "), "BTCUSDT")
    interval =          input_or_default(input("Enter your interval: ('1h'): "), "1h")
    limit =             int(input_or_default(input("Enter your limit *FOR TESTING*: ('1000'): "), "1000"))
    training_intervals =int(input_or_default(input("Enter length of desired interval length: ('60'): "), "60"))
    target_interval =   int(input_or_default(input("Enter target interval (e.g. 1 min in the future/ 5 min in the future): ('1'): "), "1"))
    columns = input_or_default(input("Enter your columns: ('close,volume,number_of_trades'): "), "close,volume,number_of_trades")
    columns = columns.split(",")

    print("your input model parameters are:")
    print(f"coin_name: {coin_name}")
    print(f"interval: {interval}")
    print(f"limit: {limit}")
    print(f"training_intervals: {training_intervals}")
    print(f"target_interval: {target_interval}")
    print(f"columns: {columns}\n\n")

    print("Checking for existing model")
    model_location = f"./saved_models/{coin_name}/{interval}+{target_interval}/{training_intervals}/{'+'.join(columns)}"

    print(f"Model location: {model_location}")
    if not os.path.isdir(model_location):
        print("Model does not exist, please train a model first")
        return
    
    print("Model exists, loading model...")
    model = load_model(model_location)
    print("Model loaded")

    fn = fetch_write_coin(coin_name, interval, limit+training_intervals+target_interval)
    klines = load_klines_from_csv(fn)
    
    klines = klines.filter(columns)
    klines = klines.values

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))


    x_data = x_scaler.fit_transform(klines[:, :])
    y_data = y_scaler.fit_transform(klines[:, :1])

    x_data, y_data = section_data(x_data, y_data, training_intervals, 1, target_interval)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    print(f"X data shape: {x_data.shape}")
    print(f"X data shape: {y_data.shape}")

    predictions = model.predict(x_data)
    predictions = y_scaler.inverse_transform(predictions)
    y_data = y_scaler.inverse_transform(y_data)


    plots_location = f"./plots/tested_models/{coin_name}/{interval}_{target_interval}_{training_intervals}_{'+'.join(columns)}_"

    if not os.path.exists(plots_location):
        os.makedirs(plots_location)

    photo_save = f"{plots_location}{time.strftime('%Y%m%d-%H%M%S')}.png"
    csvloc = plots_location + f"{time.strftime('%Y%m%d-%H%M%S')}.csv"

    title = f"Prediction Plot for '{coin_name}' every '{interval}' looking at previous '{training_intervals}' intervals trained data from '{columns}'. TESTED ON LOADED MODEL"

    # plt.style.use("ggplot")
    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price", fontsize=18)
    plt.plot(y_data[:-target_interval], label='Train')
    plt.plot(predictions[target_interval:], label='Predicted')
    plt.legend(['Actual', 'Predicted'], loc='lower right')
    plt.savefig(photo_save)
    plt.close()

    difference = y_data - predictions

    df = pd.DataFrame({"actual": y_data.flatten(), "predicted": predictions.flatten(), "difference": difference.flatten()})
    df.to_csv(csvloc)

    
  
if __name__ == "__main__":
    run()
