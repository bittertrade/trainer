from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, CuDNNLSTM
import time
import os

def train_model(x_train, y_train, sfi, batch_size=128, epochs=1, auto_save=False):
    coin_name = sfi[0]
    trading_interval = sfi[1]
    history_length = sfi[2]
    filename = "+".join(["./saved_models/", coin_name, trading_interval, str(history_length)])

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    start_time = time.time()
    if auto_save:
        for i in range(epochs):
            print("Saving, your next epoch will begin shortly")
            print(f"Epoch {i+1}/{epochs}")
            save_model(model, filename)
            model.fit(x_train, y_train, batch_size=batch_size, epochs=1)
    else:
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    return model, time.time() - start_time


def save_model(model, filename):

    model.save(filename, save_format="h5")
    return filename

def predict(x, model):
    return model.predict(x)
