import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('data/Tepung Terigu Curah.csv')

df.head()
import pandas as pd

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

df.head()
# Mengisi missing values menggunakan forward fill dan backward fill
def handle_missing_values(df):
    return df.fillna(method='ffill').fillna(method='bfill')

# Mengisi missing values untuk semua dataset di dataframe_normal
df = handle_missing_values(df)

# Menampilkan contoh data setelah mengisi missing values
print("Dataframe Normal (Tepung Terigu) setelah mengatasi missing values:")
df.head()
def normalize_data(df, exclude_columns):
    scaler = MinMaxScaler()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    features = df.drop(columns=exclude_columns)
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    for col in exclude_columns:
        scaled_df[col] = df[col].values
    return scaled_df, scaler

exclude_columns = ['Tepung Terigu (Curah)', 'Date', 'ddd_car']
df, scaler = normalize_data(df, exclude_columns)
unique_values = {'SE', 'E', 'N', 'W', 'NW', 'SW', 'NE', 'S', 'C'}


# Fungsi untuk melakukan one-hot encoding berdasarkan unique values dari kolom 'ddd_car'
def one_hot_encode_with_unique_values(df, unique_values, column):
    # Membuat kolom-kolom baru berdasarkan unique values dari kolom 'ddd_car'
    for value in unique_values:
        df[column + '_' + value] = (df[column] == value).astype(int)
    # Menghapus kolom 'ddd_car' asli karena sudah tidak diperlukan lagi setelah one-hot encoding
    df.drop(columns=[column], inplace=True)
    return df

# Menampilkan contoh unique values dari kolom 'ddd_car'
print("Unique values of 'ddd_car':", unique_values)

# Melakukan one-hot encoding pada setiap dataset dalam dataframe_normal
df = one_hot_encode_with_unique_values(df, unique_values, 'ddd_car')

# Menampilkan contoh data setelah one-hot encoding untuk 'Beras Premium'
print("Dataframe Normal (Tepung Terigu) setelah one-hot encoding:")
df.head(10)
def split_data(df, train_ratio=0.8, val_ratio=0.1):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = df[:train_end]
    val_data = df[train_end:val_end]
    test_data = df[val_end:]

    return train_data, val_data, test_data

split_data_sets = {}
train, val, test = split_data(df)
split_data_sets = {'train': train, 'val': val, 'test': test}

# Assuming df and split_data_sets are already defined
fitur = df.drop(columns=['Tepung Terigu (Curah)', 'Date']).columns.tolist()

# Pemisahan fitur dan target
X_train = split_data_sets['train'][fitur].values
y_train = split_data_sets['train']['Tepung Terigu (Curah)'].values
X_val = split_data_sets['val'][fitur].values
y_val = split_data_sets['val']['Tepung Terigu (Curah)'].values
X_test = split_data_sets['test'][fitur].values
y_test = split_data_sets['test']['Tepung Terigu (Curah)'].values

# Normalisasi data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape data sesuai dengan timestep 30
timestep = 30
batch_size = 1
X_train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_train, y_train, length=timestep, batch_size=batch_size)
X_val_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_val, y_val, length=timestep, batch_size=batch_size)
X_test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_test, y_test, length=timestep, batch_size=batch_size)

# Membangun model LSTM dan GRU yang lebih kompleks
model = Sequential()
model.add(Bidirectional(LSTM(units=200, return_sequences=True), input_shape=(timestep, len(fitur))))
model.add(Dropout(0.4))
model.add(Bidirectional(GRU(units=150, return_sequences=True)))
model.add(Dropout(0.4))
model.add(LSTM(units=75, return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(units=50))
model.add(Dropout(0.4))
model.add(Dense(units=1))

# Kompilasi model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Melatih model
history = model.fit(X_train_generator, epochs=100, validation_data=X_val_generator)

# Evaluasi model pada data uji
test_loss = model.evaluate(X_test_generator)
print(f'Test Loss: {test_loss}')

# Prediksi pada data uji
y_pred = model.predict(X_test_generator).flatten()

model.save_weights('tepung_terigu.weights.h5')