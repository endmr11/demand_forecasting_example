# 📊 Sales Forecasting Model

This project uses time series modeling techniques such as ARIMA, SARIMA, and LSTM (Long Short-Term Memory) to forecast retail sales. The file explains the project steps and the modeling methods used. The project is written in Python and utilizes various libraries for data analysis, modeling, and forecasting.

---

## 📁 Project Structure

1. **Data Preparation and Preprocessing**  
2. **Monthly Sales Analysis**  
3. **Time Series Modeling**  
4. **Model Comparison and Evaluation**  
5. **Model Performance Comparison**

---

## 📚 Libraries Used

- **pandas**: Data manipulation and analysis  
- **numpy**: Numerical computations  
- **matplotlib**: Data visualization  
- **seaborn**: Advanced data visualization  
- **statsmodels**: For ARIMA and SARIMA modeling  
- **keras**: For building the LSTM model  
- **scikeras**: To integrate Keras with scikit-learn  
- **scikit-learn**: For data splitting, hyperparameter tuning, and evaluation  

---

## ⚙️ Steps

### 1. Data Preparation and Preprocessing

- Loaded raw data from `train.csv`
- Extracted features from the "date" column such as year, month, day, weekday, and weekend flag  
- Target column: `sales`

### 2. Monthly Sales Analysis

- Grouped data by year and month to compute total monthly sales  
- Visualized monthly sales trends and identified peak values  
- Performed trend analysis to detect increase or decrease patterns  

### 3. Time Series Modeling

#### 3.1 ARIMA Model

- Used ARIMA (AutoRegressive Integrated Moving Average)
- Parameters:
  - `p`: Autoregression (AR) order  
  - `d`: Degree of differencing (to ensure stationarity)  
  - `q`: Moving Average (MA) order  
- Trained ARIMA and forecasted sales for the next 12 months

#### 3.2 SARIMA Model

- Seasonal ARIMA model (SARIMA) to capture seasonality  
- Includes a seasonal period `S = 12` (monthly seasonality)
- Trained SARIMA and analyzed forecast and residual errors

#### 3.3 LSTM Model

- Implemented a Long Short-Term Memory (LSTM) neural network  
- Normalized the data  
- Trained and evaluated the model  
- Forecasted sales for the next 12 months

---

### 4. Model Comparison and Evaluation

- Visualized predictions from ARIMA, SARIMA, and LSTM  
- Compared forecasts against actual sales values  
- Assessed accuracy and reliability of each model

---

### 5. Model Performance Comparison

- Measured model performance using **Root Mean Squared Error (RMSE)**  
- Selected the best-performing model based on lowest RMSE

---

## 🧩 Custom Functions

- `analyze_monthly_sales()`: Groups and analyzes monthly sales  
- `create_dataset()`: Converts data into time series format  
- `time_series_cv()`: Performs cross-validation for time series  
- `arima_grid_search()`: Tunes hyperparameters for ARIMA  
- `create_lstm_model()`: Builds the LSTM neural network model  

---

## 💻 Setup

Install required libraries:

📦 Dataset:  
https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data

```bash
pip install pandas numpy matplotlib seaborn statsmodels keras scikit-learn scikeras
