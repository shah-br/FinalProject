# Stock Price Analysis and Prediction

## Overview
This project implements multiple machine learning models to predict stock prices for the last 10 data points. It provides a web interface to visualize predictions and compare model performance using R² scores.
![image](https://github.com/user-attachments/assets/4600c569-9f57-4bdc-8f92-7bf5492f0ac8)
![image](https://github.com/user-attachments/assets/faf8b332-5622-43ea-980c-ce7c05c98ed0)


## Models Implemented
- **Linear Regression**
  - Uses StandardScaler for feature normalization
  - Basic model for price prediction
  - Implements simple linear relationships

- **Random Forest**
  - Uses 100 estimators for robust predictions
  - Random state 42 for reproducibility
  - No scaling needed due to tree-based nature

- **K-Nearest Neighbors (KNN)**
  - Uses 3 neighbors for prediction
  - Implements StandardScaler for feature normalization
  - Pattern-based prediction approach

- **Decision Tree**
  - Single decision tree regressor
  - Random state 42 for reproducibility
  - Direct price value predictions

- **LSTM (Long Short-Term Memory)**
  - 50 units in LSTM layer
  - MinMaxScaler for data normalization
  - 100 epochs with batch size 32

## Key Functions
- `getData()`: Splits data into train (all but last 10 points) and test (last 10 points)
- `prepare_data()`: Prepares feature vectors using closing prices
- `read_all_stock_files()`: Loads stock data from dataset directory

## Model Evaluation
- Uses R² score (coefficient of determination)
- Scores displayed with 2 decimal places (e.g., 0.85)
- Models automatically sorted by performance
- Real-time comparison of actual vs predicted values

## Setup and Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Required Dependencies
- Flask
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Chart.js (version included via CDN)
- StandardScaler and MinMaxScaler

## Project Structure
```
FinalProject-main/
├── app.py              # Flask application with routes
├── utils.py            # Model implementations and utilities
├── dataset/            # Stock data files
└── templates/          # HTML templates
    ├── base.html      # Base template
    └── index.html     # Main interface
```

## Running the Project
```bash
# Activate virtual environment
venv\Scripts\activate

# Start Flask server
python app.py

# Access the application
# Open browser and visit: http://localhost:5000
```

## Features
- Multiple model comparison in single view
- Real-time predictions
- Interactive line chart
- R² score based model evaluation
- Automatic model ranking
- Month-Year formatted x-axis
- Responsive web interface
- Loading state indicator

## Usage
1. Select a stock from the dropdown
2. Click "Predict" button
3. View R² scores for all models
4. Compare predictions on the chart
5. Models sorted by R² score

## Contributing
Feel free to submit issues and enhancement requests.
