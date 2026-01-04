# Essay Puanlama Sistemi - SBERT ile Otomatik Puanlama

Bu proje, SBERT (Sentence-BERT) kullanarak essay'leri otomatik olarak puanlayan bir makine öğrenmesi sistemidir.

## Özellikler

- **7 Farklı Kriter için Puanlama**: TITLE, THESIS, ORGANISATION, SUPPORT, ANALYSIS, SENTENCE, GRAMMAR
- **SBERT Embeddings**: Çok dilli (Türkçe destekli) semantik analiz
- **Gelişmiş Özellik Mühendisliği**: Tutarlılık analizi, Altın Vektör benzerliği, metin istatistikleri
- **Model Kaydetme/Yükleme**: Eğitilmiş modelleri kaydedip sonradan kullanma
- **Web Arayüzü**: Streamlit ile kullanıcı dostu web arayüzü

## Kurulum

1. **Gereksinimleri yükleyin:**
```bash
pip install -r requirements.txt
```

2. **Veri hazırlığı:**
   - `data/` dizinine Word dosyalarınızı (.docx) ekleyin
   - `data/ALL_SCORES.xlsx` dosyasını hazırlayın (skorlar içermeli)

## Kullanım

### 1. Model Eğitimi

Modelleri eğitmek için `sbert.py` dosyasını çalıştırın:

```bash
python sbert.py
```

Bu işlem:
- Word dosyalarından essay'leri okur
- Excel dosyasından skorları alır
- Her kriter için model eğitir
- Modelleri `saved_models/` dizinine kaydeder
- Sonuçları `model_results.txt` dosyasına yazar

### 2. Web Arayüzü ile Puanlama

Eğitilmiş modelleri kullanarak essay puanlaması yapmak için:

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak açılacak arayüzde:
1. Sol taraftaki sidebar'dan "Modelleri Yükle" butonuna tıklayın
2. Essay metninizi ana alana yazın veya yapıştırın
3. "Puanla" butonuna tıklayın
4. Tüm kriterler için puanlarınızı görün

## Dosya Yapısı

```
SBERT/
├── data/                    # Veri dosyaları
│   ├── A1.docx
│   ├── A2.docx
│   └── ALL_SCORES.xlsx
├── saved_models/           # Kaydedilmiş modeller (otomatik oluşturulur)
│   ├── model_TITLE.pkl
│   ├── model_THESIS.pkl
│   └── ...
├── sbert.py                # Model eğitimi ve tahmin fonksiyonları
├── app.py                  # Streamlit web arayüzü
├── requirements.txt       # Python paketleri
├── model_results.txt       # Eğitim sonuçları (otomatik oluşturulur)
└── README.md              # Bu dosya
```

## Model Detayları

### Kullanılan Algoritmalar
- Random Forest
- Gradient Boosting
- Ridge Regression
- Lasso Regression
- ElasticNet
- AdaBoost
- SVR (Support Vector Regression)
- XGBoost (opsiyonel)
- LightGBM (opsiyonel)

### Özellikler
- **SBERT Embeddings**: 768 boyutlu semantik vektörler
- **Metin İstatistikleri**: Kelime sayısı, cümle sayısı, ortalama uzunluklar
- **Tutarlılık Analizi**: Ardışık cümleler arası semantik benzerlik
- **Altın Vektör Benzerliği**: Yüksek puanlı essay'lere olan benzerlik
- **PCA**: Boyut azaltma (overfitting önleme)

### Model Seçimi
- K-Fold Cross Validation (5-fold)
- Stratified K-Fold (hedef değişken dağılımını korur)
- Sample Weighting (dengesiz veri için)
- GridSearchCV (hiperparametre optimizasyonu)

## Notlar

- Modeller eğitilmeden önce web arayüzü kullanılamaz
- İlk çalıştırmada SBERT modeli indirilecektir (yaklaşık 471MB)
- Küçük veri setleri için model performansı sınırlı olabilir
- Windows'ta `n_jobs=1` kullanılır (multiprocessing sorunlarını önlemek için)

## 🌐 Web'de Yayınlama (Deployment)

Uygulamanızı web'de yayınlamak için detaylı rehber için **`DEPLOYMENT.md`** dosyasına bakın.

### Hızlı Başlangıç (Streamlit Cloud):

1. **GitHub Repository Oluşturun:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/kullaniciadi/repo-adi.git
   git push -u origin main
   ```

2. **Streamlit Cloud'a Deploy Edin:**
   - https://streamlit.io/cloud adresine gidin
   - GitHub hesabınızla giriş yapın
   - "New app" > Repository seçin > "Deploy!"

3. **Model Dosyalarını Yükleyin:**
   - Model dosyaları büyükse GitHub LFS kullanın:
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git add .gitattributes saved_models/*.pkl
   git commit -m "Add models with LFS"
   git push
   ```

### Diğer Deployment Seçenekleri:
- **Railway**: https://railway.app
- **Render**: https://render.com
- **VPS**: DigitalOcean, AWS EC2, vb.

Detaylı talimatlar için `DEPLOYMENT.md` dosyasını inceleyin.

## Sorun Giderme

### Modeller yüklenmiyor
- `saved_models/` dizininin var olduğundan emin olun
- `sbert.py` dosyasını çalıştırarak modelleri eğitin

### SBERT modeli indirilemiyor
- İnternet bağlantınızı kontrol edin
- Proxy ayarlarınızı kontrol edin
- Manuel indirme için: https://huggingface.co/sentence-transformers

### NLTK hataları
- NLTK verileri otomatik indirilmeye çalışılır
- Hata durumunda kod basit regex'e geri döner

### Deployment sorunları
- `DEPLOYMENT.md` dosyasındaki "Yaygın Sorunlar" bölümüne bakın
- Deployment platformunun log'larını kontrol edin
- Model dosyalarının repository'de olduğundan emin olun

## Lisans

Bu proje eğitim amaçlıdır.

