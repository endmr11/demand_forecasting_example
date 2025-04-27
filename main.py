import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.src.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA  # ARIMA modelini statsmodels kütüphanesiyle kuruyoruz
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import math
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from scikeras.wrappers import KerasRegressor

train_data = pd.read_csv("train.csv", parse_dates=["date"])


train_data["year"] = train_data["date"].dt.year #.dt.year: Tarihten yıl bilgisini alır.
train_data["month"] = train_data["date"].dt.month #.dt.month: Ay bilgisini alır.
train_data["day"] = train_data["date"].dt.day #.dt.day: Gün bilgisini alır.
train_data["dayofweek"] = train_data["date"].dt.dayofweek #.dt.dayofweek
train_data["is_weekend"] = train_data["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)#apply(lambda x: 1 if x >= 5 else 0): Cumartesi (5) ve Pazar (6) ise 1, diğer günler 0 olacak
print(train_data.head())


#Aylık Satış Analizi - Her yılın her ayında toplam kaç satış yapılmış bunu bulmak.
def analyze_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Yıl ve Ay'a göre gruplama yap, Sales toplamını al
    monthly_sales = df.groupby(["year", "month"])["sales"].sum().reset_index() #Her yılın her ayında toplam satış sayısını topluyoruz ve sonucu düzgün bir DataFrame çevirelim
    # 2. 'year' ve 'month' kolonlarından bir tarih objesi (YYYY-MM-01 gibi) oluştur
    monthly_sales["date"] = pd.to_datetime(monthly_sales.assign(day=1)[["year", "month", "day"]])#Ay ve yılı birleştirip tam bir tarih formatı yaratıyoruz
    return monthly_sales

#Genel Satış Eğilimi (Trend)
#En Yüksek Satış Noktası
#Ortalama Satış
#Mevsimsel Dalgalanmalar
train_monthly_sales = analyze_monthly_sales(train_data)
print(train_monthly_sales.head())

# Ortalama satış değeri
train_mean_sales = train_monthly_sales["sales"].mean()

# En yüksek satış yapılan ayın verisi
train_max_sales_row = train_monthly_sales.loc[train_monthly_sales["sales"].idxmax()]

plt.figure(figsize=(14, 6))
sns.set_style("whitegrid")

sns.lineplot(data=train_monthly_sales, x="date", y="sales", marker="o", label="Aylık Satışlar")

# Ortalama satış çizgisi
plt.axhline(train_mean_sales, color="red", linestyle="--", label=f"Ortalama Satış: {train_mean_sales:.0f}")

# En yüksek satış noktasını belirleme
plt.scatter(train_max_sales_row["date"], train_max_sales_row["sales"], color="orange", s=100, zorder=5, label="En Yüksek Satış")
plt.text(train_max_sales_row["date"], train_max_sales_row["sales"] + 500, f"{train_max_sales_row['sales']}",
         color="black", ha="center", va="bottom", fontsize=10)
plt.title("Train Detaylı Aylık Satış Trend Analizi", fontsize=16)
plt.xlabel("Tarih", fontsize=12)
plt.ylabel("Toplam Satış", fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.show()

# 'date' kolonu
train_monthly_sales.set_index('date', inplace=True)

# ARIMA Modeli
# Veriyi durağan hale getirme (trend ve mevsimsel etkilerden arındırma)
# Aylık satışların farkını alarak diff() ile durağanlık sağlıyoruz
train_monthly_sales['sales_diff'] = train_monthly_sales['sales'].diff()

plt.figure(figsize=(12, 6))
plt.plot(train_monthly_sales['sales_diff'])
plt.title("Durağanlaştırılmış Satış Verisi")
plt.show()


# ARIMA modelini kuruyoruz
#p: AR kısmı için lag sayısı.
#d: Fark alma derecesi (durağan hale getirmek için).
#q: MA kısmı için hata terimi sayısı.
#5 geçmiş ayın etkisini kullanıyoruz
#1 fark alıyoruz
#Hata terimi eklemiyoruz

model = ARIMA(train_monthly_sales['sales'], order=(5, 1, 0))  # (p,d,q) şeklinde parametreler
model_fit = model.fit()
print(model_fit.summary())

# 12 ay sonrası tahmin
forecast = model_fit.forecast(steps=12)
forecast_dates = pd.date_range(train_monthly_sales.index[-1], periods=13, freq='M')[1:]

plt.figure(figsize=(12, 6))
plt.plot(train_monthly_sales.index, train_monthly_sales['sales'], label="Gerçek Satışlar")
plt.plot(forecast_dates, forecast, label="Tahmin Edilen Satışlar", color='red', linestyle='--')
plt.title("ARIMA Modeli ile Satış Tahmini")
plt.xlabel("Tarih")
plt.ylabel("Satış")
plt.legend()
plt.show()

# SARIMA
sarima_model = SARIMAX(train_monthly_sales['sales'],
                       order=(5, 1, 0),  # ARIMA
                       seasonal_order=(1, 1, 1, 12),  # Mevsimsel parametreler (S=12 çünkü aylık verimiz var)
                       enforce_stationarity=False,
                       enforce_invertibility=False)
# Model egitme
sarima_model_fit = sarima_model.fit(disp=False)
# Modelin özeti
print(sarima_model_fit.summary())

# 12 ay sonrasına kadar tahmin yapıyoruz
sarima_forecast = sarima_model_fit.forecast(steps=12)
sarima_forecast_dates = pd.date_range(train_monthly_sales.index[-1], periods=13, freq='M')[1:]

# Tahmin
plt.figure(figsize=(12, 6))
plt.plot(train_monthly_sales.index, train_monthly_sales['sales'], label="Gerçek Satışlar")
plt.plot(sarima_forecast_dates, sarima_forecast, label="SARIMA Tahmin Edilen Satışlar", color='red', linestyle='--')
plt.title("SARIMA Modeli ile Satış Tahmini")
plt.xlabel("Tarih")
plt.ylabel("Satış")
plt.legend()
plt.show()

# Modelin hata payları
residuals = sarima_model_fit.resid

# Residuals
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title("SARIMA Modeli Residuals (Hata Payları)")
plt.xlabel("Zaman")
plt.ylabel("Hata")
plt.show()

# Residualsun histogramı
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=30)
plt.title("SARIMA Modeli Residuals Histogramı")
plt.xlabel("Hata")
plt.ylabel("Frekans")
plt.show()

# Residual ortalama ve varyans
print(f"Residuals Ortalama: {residuals.mean()}")
print(f"Residuals Varyansı: {residuals.var()}")


# Veriyi normalize etme
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_sales = scaler.fit_transform(train_data['sales'].values.reshape(-1, 1))

# Veriyi X, y formatına dönüştürme (zaman serisi için)
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 12  # 12 ay
X, y = create_dataset(scaled_sales, time_step)

# X
X = X.reshape(X.shape[0], X.shape[1], 1)

# Eğitim ve test verisini ayırma
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))  # Çıktı katmanı

# Modeli derleme
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Test seti tahmin
predicted_sales = model.predict(X_test)

# Tahminleri eski ölçeğe geri dönüştürme
predicted_sales = scaler.inverse_transform(predicted_sales.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Tüm modellerin tahminlerini karşılaştırma
plt.figure(figsize=(15, 7))

# Gerçek veriler
plt.plot(train_monthly_sales.index, train_monthly_sales['sales'], label='Gerçek Satışlar', color='black', linewidth=2)

# ARIMA tahminleri
arima_predictions = model_fit.predict(start=0, end=len(train_monthly_sales)-1)
plt.plot(train_monthly_sales.index, arima_predictions, label='ARIMA Tahminleri', linestyle='--', alpha=0.7)

# SARIMA tahminleri
sarima_predictions = sarima_model_fit.predict(start=0, end=len(train_monthly_sales)-1)
plt.plot(train_monthly_sales.index, sarima_predictions, label='SARIMA Tahminleri', linestyle='--', alpha=0.7)

plt.title('Model Tahminlerinin Karşılaştırması')
plt.xlabel('Tarih')
plt.ylabel('Satış')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Mevsimsellik Analizi
decomposition = seasonal_decompose(train_monthly_sales['sales'], model='multiplicative', period=12)

# Mevsimsel ayrıştırma
plt.figure(figsize=(15, 10))

# Orijinal veri
plt.subplot(411)
plt.plot(train_monthly_sales.index, train_monthly_sales['sales'])
plt.title('Orijinal Satış Verisi')
plt.xticks(rotation=45)

# Trend
plt.subplot(412)
plt.plot(decomposition.trend)
plt.title('Trend')
plt.xticks(rotation=45)

# Mevsimsellik
plt.subplot(413)
plt.plot(decomposition.seasonal)
plt.title('Mevsimsellik')
plt.xticks(rotation=45)

# Artık (Residual)
plt.subplot(414)
plt.plot(decomposition.resid)
plt.title('Artık (Residual)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Mevsimsel güç analizi
seasonal_strength = 1 - (decomposition.resid.var() / (decomposition.seasonal + decomposition.resid).var())
print(f"\nMevsimsel Güç: {seasonal_strength:.4f}")

# Time Series Cross Validation için fonksiyon
def time_series_cv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = mean_squared_error(y_test, y_pred)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

# ARIMA için hyperparameter tuning
def arima_grid_search(data, p_range, d_range, q_range):
    best_aic = float('inf')
    best_params = None
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(data, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_params = (p, d, q)
                except:
                    continue
    
    return best_params, best_aic

# ARIMA için en iyi parametreler
p_range = range(0, 6)
d_range = range(0, 2)
q_range = range(0, 6)

best_params, best_aic = arima_grid_search(train_monthly_sales['sales'], p_range, d_range, q_range)
print(f"\nARIMA En İyi Parametreler: {best_params}")
print(f"En İyi AIC: {best_aic}")

def create_lstm_model(units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# LSTM modelini KerasRegressora dönüştürme
lstm_model = KerasRegressor(build_fn=create_lstm_model, verbose=0)

# Grid search
param_grid = {
    'units': [30, 50, 70],
    'dropout': [0.1, 0.2, 0.3]
}

# Grid search
grid_search = GridSearchCV(estimator=lstm_model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

print("\nLSTM En İyi Parametreler:")
print(grid_search.best_params_)
print(f"En İyi Skor: {grid_search.best_score_}")

# En iyi modeli kullanarak tahmin yap
best_lstm_model = create_lstm_model(**grid_search.best_params_)
best_lstm_model.fit(X_train, y_train, epochs=5, batch_size=32)
best_lstm_predictions = best_lstm_model.predict(X_test)
best_lstm_predictions = scaler.inverse_transform(best_lstm_predictions)

# En iyi LSTM modelinin performansı
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='Gerçek Satışlar')
plt.plot(best_lstm_predictions, label='En İyi LSTM Tahminleri', color='red')
plt.title("En İyi LSTM Modeli ile Satış Tahmini")
plt.legend()
plt.show()

# Model Performans Karşılaştırması (En İyi Modeller)
print("\nEn İyi Modellerin Performans Karşılaştırması:")
print(f"ARIMA (p={best_params[0]}, d={best_params[1]}, q={best_params[2]}) RMSE: {math.sqrt(mean_squared_error(train_monthly_sales['sales'], arima_predictions)):.2f}")
print(f"LSTM (units={grid_search.best_params_['units']}, dropout={grid_search.best_params_['dropout']}) RMSE: {math.sqrt(mean_squared_error(y_test_actual, best_lstm_predictions)):.2f}")



