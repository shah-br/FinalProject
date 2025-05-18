from flask import Flask, render_template, request, jsonify
from utils import *
import pandas as pd
import os

app = Flask(__name__)

# Load all stock data at startup
dataset_path = "dataset"
stock_data = read_all_stock_files(dataset_path)

@app.route('/')
def index():
    # Get list of all stocks
    stock_list = list(stock_data.keys())
    return render_template('index.html', stocks=stock_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock_name = request.form['stock']
        
        if stock_name not in stock_data:
            return jsonify({'error': 'Stock not found'})
        
        df = stock_data[stock_name]
        train_data, test_data, actual_prices = getData(df)
        
        # Get predictions from all models
        lr_predictions, lr_score = linear_regression(train_data, test_data)
        rf_predictions, rf_score = random_forests(train_data, test_data)
        knn_predictions, knn_score = KNN(train_data, test_data)
        dt_predictions, dt_score = DT(train_data, test_data)
        lstm_predictions, lstm_score = LSTM_model(train_data, test_data)
        
        return jsonify({
            'success': True,
            'dates': {
                'test_dates': test_data['datetime' if 'datetime' in df.columns else 'date'].tolist(),
            },
            'actual_prices': actual_prices,
            'prediction': {
                'linear_regression': lr_predictions,
                'random_forest': rf_predictions,
                'knn': knn_predictions,
                'decision_tree': dt_predictions,
                'lstm': lstm_predictions
            },
            'scores': {
                'linear_regression': float(lr_score),
                'random_forest': float(rf_score),
                'knn': float(knn_score),
                'decision_tree': float(dt_score),
                'lstm': float(lstm_score)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.debug = True
    app.run()
