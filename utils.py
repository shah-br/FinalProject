from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy
import pandas as pd
import math
import os

# setting a seed for reproducibility
numpy.random.seed(10)

def read_all_stock_files(folder_path):
    allFiles = []
    for (_, _, files) in os.walk(folder_path):
        allFiles.extend(files)
        break

    dataframe_dict = {}
    for stock_file in allFiles:
        df = pd.read_csv(folder_path + "/" +stock_file)
        dataframe_dict[(stock_file.split('_'))[0]] = df

    return dataframe_dict

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# create dataset from the dataframe
def create_preprocessed_Dataset(df):
    # Check which date column name exists and standardize it
    date_col = 'datetime' if 'datetime' in df.columns else 'date'
    
    # Keep only date and close columns, then select only close for processing
    df = df[[date_col, 'close']]
    df = df['close']
    dataset = df.values
    dataset = dataset.reshape(-1, 1)
    dataset = dataset.astype('float32')

    # split into train and test sets
    train_size = len(dataset) - 2
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)  # Changed this line
    testX, testY = create_dataset(test, look_back)     # And this line

    return trainX, trainY, testX, testY

# extract input dates and closing price value of stocks
def getData(df):
    # Check which date column name exists
    date_col = 'datetime' if 'datetime' in df.columns else 'date'
    
    # Get the last 30 rows for testing
    test_data = df.tail(10)
    # Get all data except last 30 rows for training
    train_data = df.head(len(df) - 10)
    
    # Extract dates and prices for actual values
    test_dates = test_data[date_col].tolist()
    actual_prices = test_data['close'].tolist()
    
    return train_data, test_data, actual_prices

def prepare_data(data):
    """
    Prepare data using only the closing prices
    """
    X = data['close'].values.reshape(-1, 1)
    y = data['close'].values
    return X, y

def linear_regression(train_data, test_data):
    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    score = r2_score(y_test, predictions)
    return predictions.tolist(), score

def random_forests(train_data, test_data):
    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    score = r2_score(y_test, predictions)
    return predictions.tolist(), score

def KNN(train_data, test_data):
    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)
    
    # Scale the features - important for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    
    score = r2_score(y_test, predictions)
    return predictions.tolist(), score

def DT(train_data, test_data):
    X_train, y_train = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)
    
    model = tree.DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    score = r2_score(y_test, predictions)
    return predictions.tolist(), score

def LSTM_model(train_data, test_data):
    # Scale all data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data['close'].values.reshape(-1, 1))
    test_scaled = scaler.transform(test_data['close'].values.reshape(-1, 1))
    
    # Reshape for LSTM [samples, timesteps, features]
    X_train = train_scaled.reshape((train_scaled.shape[0], 1, 1))
    X_test = test_scaled.reshape((test_scaled.shape[0], 1, 1))
    
    model = Sequential([
        LSTM(50, input_shape=(1, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, train_scaled, epochs=100, batch_size=32, verbose=0)
    
    predictions = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(test_scaled)
    
    score = r2_score(y_test, predictions)
    return predictions.flatten().tolist(), score
