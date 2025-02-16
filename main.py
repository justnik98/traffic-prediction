import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import math

# Задаем начальную и конечную дату
start_date = dt.datetime(2020, 4, 1)
end_date = dt.datetime(2023, 4, 1)

# Загружаем данные с Yahoo Finance
data = yf.download("GOOGL", start=start_date, end=end_date)

# Выводим информацию о загруженных данных
pd.set_option('display.max_rows', 4)
pd.set_option('display.max_columns', 5)
print(data)

# Настроим 80% данных для обучения
training_data_len = math.ceil(len(data) * 0.8)
print("Размер обучающего набора:", training_data_len)

# Разделяем данные на обучающую и тестовую выборки
train_data = data[:training_data_len]
test_data = data[training_data_len:]
print("Размер обучающей выборки:", train_data.shape)
print("Размер тестовой выборки:", test_data.shape)

# Проверим, как устроены многоуровневые колонки
print("Колонки данных:", data.columns)

# Доступ к значениям 'Open' для тикера 'GOOGL'
dataset_train = train_data[('Open', 'GOOGL')].values

# Преобразуем в 2D массив для использования в моделях машинного обучения
dataset_train = np.reshape(dataset_train, (-1, 1))
print("Размер обучающих данных:", dataset_train.shape)

# Доступ к тестовым данным для 'Open' для тикера 'GOOGL'
dataset_test = test_data[('Open', 'GOOGL')].values
dataset_test = np.reshape(dataset_test, (-1, 1))
print("Размер тестовых данных:", dataset_test.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
# scaling dataset
scaled_train = scaler.fit_transform(dataset_train)

print(scaled_train[:5])
# Selecting Open Price values
dataset_test = test_data.Open.values
# Reshaping 1D to 2D array
dataset_test = np.reshape(dataset_test, (-1, 1))
# Normalizing values between 0 and 1
scaled_test = scaler.fit_transform(dataset_test)
print(*scaled_test[:5])
X_train = []
y_train = []
for i in range(50, len(scaled_train)):
    X_train.append(scaled_train[i - 50:i, 0])
    y_train.append(scaled_train[i, 0])
    if i <= 51:
        print(X_train)
        print(y_train)
        print()
X_test = []
y_test = []
for i in range(50, len(scaled_test)):
    X_test.append(scaled_test[i - 50:i, 0])
    y_test.append(scaled_test[i, 0])
# The data is converted to Numpy array
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
print("X_train :", X_train.shape, "y_train :", y_train.shape)
# The data is converted to numpy array
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))
print("X_test :", X_test.shape, "y_test :", y_test.shape)

# importing libraries
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import GRU, Bidirectional
from keras.optimizers import SGD
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# initializing the RNN
regressor = Sequential()

# adding RNN layers and dropout regularization
regressor.add(SimpleRNN(units=50,
                        activation="tanh",
                        return_sequences=True,
                        input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units=50,
                        activation="tanh",
                        return_sequences=True))

regressor.add(SimpleRNN(units=50,
                        activation="tanh",
                        return_sequences=True))

regressor.add(SimpleRNN(units=50))

# adding the output layer
regressor.add(Dense(units=1, activation='sigmoid'))

# compiling RNN
regressor.compile(optimizer=SGD(learning_rate=0.01,
                                decay=1e-6,
                                momentum=0.9,
                                nesterov=True),
                  loss="mean_squared_error")

# fitting the model
regressor.fit(X_train, y_train, epochs=20, batch_size=2)
regressor.summary()
# Initialising the model
regressorLSTM = Sequential()

# Adding LSTM layers
regressorLSTM.add(LSTM(50,
                       return_sequences=True,
                       input_shape=(X_train.shape[1], 1)))
regressorLSTM.add(LSTM(50,
                       return_sequences=False))
regressorLSTM.add(Dense(25))

# Adding the output layer
regressorLSTM.add(Dense(1))

# Compiling the model
regressorLSTM.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=["accuracy"])

# Fitting the model
regressorLSTM.fit(X_train,
                  y_train,
                  batch_size=1,
                  epochs=12)
regressorLSTM.summary()
# Initialising the model
regressorGRU = Sequential()

# GRU layers with Dropout regularisation
regressorGRU.add(GRU(units=50,
                     return_sequences=True,
                     input_shape=(X_train.shape[1], 1),
                     activation='tanh'))
regressorGRU.add(Dropout(0.2))

regressorGRU.add(GRU(units=50,
                     return_sequences=True,
                     activation='tanh'))

regressorGRU.add(GRU(units=50,
                     return_sequences=True,
                     activation='tanh'))

regressorGRU.add(GRU(units=50,
                     activation='tanh'))

# The output layer
regressorGRU.add(Dense(units=1,
                       activation='relu'))
# Compiling the RNN
regressorGRU.compile(optimizer=SGD(learning_rate=0.01,
                                   decay=1e-7,
                                   momentum=0.9,
                                   nesterov=False),
                     loss='mean_squared_error')

# Fitting the data
regressorGRU.fit(X_train, y_train, epochs=20, batch_size=1)
regressorGRU.summary()
# predictions with X_test data
y_RNN = regressor.predict(X_test)
y_LSTM = regressorLSTM.predict(X_test)
y_GRU = regressorGRU.predict(X_test)
# scaling back from 0-1 to original
y_RNN_O = scaler.inverse_transform(y_RNN)
y_LSTM_O = scaler.inverse_transform(y_LSTM)
y_GRU_O = scaler.inverse_transform(y_GRU)
fig, axs = plt.subplots(3, figsize=(18, 12), sharex=True, sharey=True)
fig.suptitle('Model Predictions')

# Plot for RNN predictions
axs[0].plot(train_data.index[150:], train_data.Open[150:], label="train_data", color="b")
axs[0].plot(test_data.index, test_data.Open, label="test_data", color="g")
axs[0].plot(test_data.index[50:], y_RNN_O, label="y_RNN", color="brown")
axs[0].legend()
axs[0].title.set_text("Basic RNN")

# Plot for LSTM predictions
axs[1].plot(train_data.index[150:], train_data.Open[150:], label="train_data", color="b")
axs[1].plot(test_data.index, test_data.Open, label="test_data", color="g")
axs[1].plot(test_data.index[50:], y_LSTM_O, label="y_LSTM", color="orange")
axs[1].legend()
axs[1].title.set_text("LSTM")

# Plot for GRU predictions
axs[2].plot(train_data.index[150:], train_data.Open[150:], label="train_data", color="b")
axs[2].plot(test_data.index, test_data.Open, label="test_data", color="g")
axs[2].plot(test_data.index[50:], y_GRU_O, label="y_GRU", color="red")
axs[2].legend()
axs[2].title.set_text("GRU")

plt.xlabel("Days")
plt.ylabel("Open price")

plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Функция для вычисления MSE
def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


# Расчет MSE для каждой модели
mse_RNN = calculate_mse(scaled_test[50:], y_RNN)
mse_LSTM = calculate_mse(scaled_test[50:], y_LSTM)
mse_GRU = calculate_mse(scaled_test[50:], y_GRU)

# Выводим результаты
print(f'MSE для модели RNN: {mse_RNN}')
print(f'MSE для модели LSTM: {mse_LSTM}')
print(f'MSE для модели GRU: {mse_GRU}')

# Расчет MSE для каждой модели
R2_RNN = calculate_r2(scaled_test[50:], y_RNN)
R2_LSTM = calculate_r2(scaled_test[50:], y_LSTM)
R2_GRU = calculate_r2(scaled_test[50:], y_GRU)

# Выводим результаты
print(f'R2 score для модели RNN: {R2_RNN}')
print(f'R2 score для модели LSTM: {R2_LSTM}')
print(f'R2 score для модели GRU: {R2_GRU}')
