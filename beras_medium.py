import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os

def replace_comma(df, columns):
    for col in columns:
        df[col] = df[col].str.replace(',', '.')
    return df


def convert_to_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def handle_missing_values(df):
    return df.fillna(method='ffill').fillna(method='bfill')


def normalize_data(df, exclude_columns):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    features = df.drop(columns=exclude_columns)

    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(df[['Beras Medium']])

    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df['Beras Medium'] = scaled_target

    for col in exclude_columns:
        scaled_df[col] = df[col].values

    return scaled_df, feature_scaler, target_scaler


def one_hot_encode_with_unique_values(df, unique_values, column):
    for value in unique_values:
        df[column + '_' + value] = (df[column] == value).astype(int)
    df.drop(columns=[column], inplace=True)
    return df


def split_data(df, train_ratio=0.8, val_ratio=0.1):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_data = df[:train_end]
    val_data = df[train_end:val_end]
    test_data = df[val_end:]
    return train_data, val_data, test_data


def main():
    df = pd.read_csv('data/Beras Medium.csv')

    columns_to_convert = ['Tn', 'Tx', 'Tavg', 'RR', 'ss']
    df = replace_comma(df, columns_to_convert)
    df = convert_to_numeric(df, columns_to_convert)
    df = handle_missing_values(df)

    exclude_columns = ['Beras Medium', 'Date', 'ddd_car']
    df, feature_scaler, target_scaler = normalize_data(df, exclude_columns)

    unique_values = {'SE', 'E', 'N', 'W', 'NW', 'SW', 'NE', 'S', 'C'}
    df = one_hot_encode_with_unique_values(df, unique_values, 'ddd_car')

    train, val, test = split_data(df)

    fitur = df.drop(columns=['Beras Medium', 'Date']).columns.tolist()

    X_train = train[fitur].values
    y_train = train['Beras Medium'].values
    X_val = val[fitur].values
    y_val = val['Beras Medium'].values
    X_test = test[fitur].values
    y_test = test['Beras Medium'].values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    timestep = 30
    batch_size = 16
    X_train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        X_train, y_train, length=timestep, batch_size=1)
    X_val_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        X_val, y_val, length=timestep, batch_size=1)
    X_test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
        X_test, y_test, length=timestep, batch_size=1)

    model1 = Sequential()
    model1.add(Bidirectional(LSTM(units=200, return_sequences=True), input_shape=(timestep, len(fitur))))
    model1.add(Dropout(0.4))
    model1.add(Bidirectional(GRU(units=150, return_sequences=True)))
    model1.add(Dropout(0.4))
    model1.add(LSTM(units=75, return_sequences=True))
    model1.add(Dropout(0.3))
    model1.add(GRU(units=50))
    model1.add(Dropout(0.4))
    model1.add(Dense(units=1))

    model1.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

    model1.fit(X_train_generator, epochs=100, validation_data=X_val_generator)

    test_loss = model1.evaluate(X_test_generator)

    print(f'Test Loss: {test_loss}')

    return model1


if __name__ == '__main__':
    model = main()

    model.save_weights("beras_medium.weights.h5")
