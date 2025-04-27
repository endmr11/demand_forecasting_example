# Satış Tahmini Modeli

Bu proje, perakende satışlarını tahmin etmek için ARIMA, SARIMA ve LSTM (Long Short-Term Memory) gibi zaman serisi modelleme yöntemlerini kullanmaktadır. Bu dosya, proje adımlarını ve kullanılan yöntemleri açıklar. Proje, Python'da yazılmış olup, veri analizi, modelleme ve tahmin için çeşitli kütüphaneler kullanmaktadır.

## Proje İçeriği

1. **Veri Hazırlığı ve Ön İşleme**
2. **Aylık Satış Analizi**
3. **Zaman Serisi Modelleme**
4. **Model Karşılaştırması ve Değerlendirmesi**
5. **Model Performans Karşılaştırması**

## Kullanılan Kütüphaneler

- **pandas**: Veri analizi ve manipülasyonu için.
- **numpy**: Sayısal hesaplamalar için.
- **matplotlib**: Veri görselleştirme için.
- **seaborn**: İleri düzey görselleştirme için.
- **statsmodels**: ARIMA ve SARIMA modelleri için.
- **keras**: LSTM modelinin oluşturulması için.
- **scikeras**: Keras'ı sklearn uyumlu hale getirmek için.
- **scikit-learn**: Veri bölme, hiperparametre ayarlama ve model değerlendirme için.

## Adımlar

### 1. Veri Hazırlığı ve Ön İşleme

Projeye başlarken, ham veri dosyasını `train.csv` dosyasından okuduk ve aşağıdaki işlemeleri gerçekleştirdik:

- **Tarih Kolonları**: "date" kolonundan yıl, ay, gün, haftanın günü ve hafta sonu olup olmadığı gibi özellikler türettik.
- **Satış Verisi**: "sales" kolonunun analizi ve tahmin yapılması hedeflendi.

### 2. Aylık Satış Analizi

Satışların zaman içindeki eğilimini analiz etmek için:
- Her yılın her ayında toplam satışları gruplayıp, ay bazında satışların toplamını hesapladık.
- Aylık satışları görselleştirerek ortalama satış değerini ve en yüksek satış noktasını belirledik.
- Trend analizi yaparak satışlarda görülen artış/azalış durumunu inceledik.

### 3. Zaman Serisi Modelleme

Zaman serisi tahminlemesi yapmak için üç farklı model kullandık:

#### 3.1 ARIMA Modeli

ARIMA (AutoRegressive Integrated Moving Average) modelini uyguladık. Modelin parametrelerini belirlemek için:
- **p**: Geçmiş değerlerin etkisi (AR kısmı).
- **d**: Fark alma (Durağanlık sağlamak için).
- **q**: Hata terimi sayısı (MA kısmı).

ARIMA modelini eğittikten sonra, gelecek 12 ayın satışlarını tahmin ettik.

#### 3.2 SARIMA Modeli

ARIMA'nın mevsimsel versiyonu olan SARIMA'yı kullandık. SARIMA, aynı zamanda sezonluk bileşenleri modelleyebilmek için:
- **S**: Mevsimsel periyot (12 ay).

Model eğitildikten sonra, SARIMA'nın tahminlerini ve model hata paylarını analiz ettik.

#### 3.3 LSTM Modeli

LSTM (Long Short-Term Memory), zaman serileri verisiyle çalışan derin öğrenme modelidir. Bu modelde:
- Veriyi normalize ettik.
- LSTM'nin eğitim ve test aşamalarını gerçekleştirdik.
- 12 aylık satış tahminini modelledik.

### 4. Model Karşılaştırması ve Değerlendirmesi

Elde ettiğimiz tahminleri görselleştirerek, **ARIMA**, **SARIMA** ve **LSTM** modellerini karşılaştırdık:
- Gerçek satış verilerini, her üç modelin tahminleriyle karşılaştırarak hangi modelin daha doğru tahminler yaptığını değerlendirdik.

### 5. Model Performans Karşılaştırması

Her modelin doğruluğunu karşılaştırmak için:
- **Root Mean Squared Error (RMSE)** metriği kullanarak her modelin performansını ölçtük.
- En iyi performansı gösteren model, tahmin sonuçlarına göre seçildi.

## Kullanılan Fonksiyonlar

- `analyze_monthly_sales()`: Aylık satışları gruplar ve analiz eder.
- `create_dataset()`: Veriyi zaman serisi formatına dönüştürür.
- `time_series_cv()`: Zaman serisi çapraz doğrulama işlemi yapar.
- `arima_grid_search()`: ARIMA modelinin hiperparametre ayarlarını yapar.
- `create_lstm_model()`: LSTM modelini oluşturur.


## Kurulum

Proje için gerekli kütüphanelerin kurulumu:

Veri Seti: https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data

```bash
pip install pandas numpy matplotlib seaborn statsmodels keras scikit-learn scikeras
