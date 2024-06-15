import pandas as pd
import numpy as np
import tensorflow as tf
import json
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from keras.models import Sequential
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

df = pd.read_csv('Bawang Putih Bonggol.csv')


def replace_comma(df, columns):
    for col in columns:
        df[col] = df[col].str.replace(',', '.')
    return df


def convert_to_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


columns_to_convert = ['Tn', 'Tx', 'Tavg', 'RR', 'ss']

df = replace_comma(df, columns_to_convert)
df = convert_to_numeric(df, columns_to_convert)


def handle_missing_values(df):
    return df.fillna(method='ffill').fillna(method='bfill')


df = handle_missing_values(df)

from sklearn.preprocessing import MinMaxScaler


def normalize_data(df, exclude_columns):
    scaler = MinMaxScaler()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    features = df.drop(columns=exclude_columns)
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    for col in exclude_columns:
        scaled_df[col] = df[col].values
    return scaled_df, scaler


exclude_columns = ['Bawang Putih Bonggol', 'Date', 'ddd_car']
df, scaler = normalize_data(df, exclude_columns)

unique_values = {'SE', 'E', 'N', 'W', 'NW', 'SW', 'NE', 'S', 'C'}


def one_hot_encode_with_unique_values(df, unique_values, column):
    for value in unique_values:
        df[column + '_' + value] = (df[column] == value).astype(int)
    df.drop(columns=[column], inplace=True)
    return df


df = one_hot_encode_with_unique_values(df, unique_values, 'ddd_car')


def split_data(df, train_ratio=0.8, val_ratio=0.1):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_data = df[:train_end]
    val_data = df[train_end:val_end]
    test_data = df[val_end:]
    return train_data, val_data, test_data


train, val, test = split_data(df)
fitur = df.drop(columns=['Bawang Putih Bonggol', 'Date']).columns.tolist()

X_train = train[fitur].values
y_train = train['Bawang Putih Bonggol'].values
X_val = val[fitur].values
y_val = val['Bawang Putih Bonggol'].values
X_test = test[fitur].values
y_test = test['Bawang Putih Bonggol'].values

timestep = 30
X_train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_train, y_train, length=timestep, batch_size=1)
X_val_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_val, y_val, length=timestep, batch_size=1)
X_test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_test, y_test, length=timestep, batch_size=1)

model = Sequential()
model.add(LSTM(units=50, input_shape=(timestep, len(fitur)), return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train_generator, epochs=50, validation_data=X_val_generator)

y_pred = model.predict(X_test_generator)

predictions_output = {"predictions": y_pred.tolist(), "actual_values": y_test.tolist()}

predictions_json = json.dumps(predictions_output, indent=2)

with open('predictions_output.json', 'w') as f:
    f.write(predictions_json)

with open('predictions_output.json', 'r') as f:
    predictions_data = json.load(f)

print(predictions_data)