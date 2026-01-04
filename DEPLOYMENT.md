# 🌐 Streamlit Uygulamasını Web'de Yayınlama Rehberi

Bu rehber, Essay Puanlama Sistemi'ni web'de yayınlamak için adım adım talimatlar içerir.

## 📋 Ön Hazırlık

### 1. Model Dosyalarını Hazırlama

Önce modellerinizi eğitip kaydettiğinizden emin olun:

```bash
python sbert.py
```

Bu işlem `saved_models/` dizininde model dosyalarını oluşturur. **ÖNEMLİ:** Model dosyaları büyük olabilir (her biri birkaç MB). Deployment için bu dosyaları da yüklemek gerekecek.

### 2. Git Repository Hazırlama

Projenizi Git ile yönetiyorsanız:

```bash
git init
git add .
git commit -m "Initial commit"
```

**Not:** `saved_models/` klasörünü `.gitignore`'a eklemeyin (veya GitHub LFS kullanın - aşağıda açıklanacak).

---

## 🚀 Deployment Seçenekleri

### Seçenek 1: Streamlit Cloud (ÖNERİLEN - En Kolay) ⭐

Streamlit Cloud, Streamlit uygulamaları için özel olarak tasarlanmış ücretsiz bir platformdur.

#### Adımlar:

1. **GitHub Repository Oluşturun:**
   - GitHub'da yeni bir repository oluşturun
   - Projenizi push edin:
   ```bash
   git remote add origin https://github.com/kullaniciadi/repo-adi.git
   git push -u origin main
   ```

2. **Streamlit Cloud'a Giriş Yapın:**
   - https://streamlit.io/cloud adresine gidin
   - "Sign up" ile GitHub hesabınızla giriş yapın

3. **Uygulamayı Deploy Edin:**
   - "New app" butonuna tıklayın
   - Repository'nizi seçin
   - Branch: `main` (veya `master`)
   - Main file: `app.py`
   - "Deploy!" butonuna tıklayın

4. **Model Dosyalarını Yükleme:**
   - Model dosyaları büyükse (>100MB), GitHub LFS kullanın:
   ```bash
   # GitHub LFS kurulumu (ilk kez)
   git lfs install
   
   # .pkl dosyalarını LFS ile takip et
   git lfs track "*.pkl"
   git add .gitattributes
   git add saved_models/*.pkl
   git commit -m "Add model files with LFS"
   git push
   ```

5. **Ortam Değişkenleri (Gerekirse):**
   - Streamlit Cloud'da "Settings" > "Secrets" bölümünden ortam değişkenleri ekleyebilirsiniz

#### Avantajları:
- ✅ Tamamen ücretsiz
- ✅ Otomatik HTTPS
- ✅ GitHub ile entegre (otomatik güncelleme)
- ✅ Kolay kullanım
- ✅ Özel domain desteği (ücretli plan)

#### Dezavantajları:
- ⚠️ Model dosyaları büyükse GitHub LFS gerekir (ücretsiz plan: 1GB)
- ⚠️ CPU ve RAM limitleri var

---

### Seçenek 2: Railway 🚂

Railway, modern bir deployment platformudur ve ücretsiz plan sunar.

#### Adımlar:

1. **Railway Hesabı Oluşturun:**
   - https://railway.app adresine gidin
   - GitHub hesabınızla giriş yapın

2. **Yeni Proje Oluşturun:**
   - "New Project" > "Deploy from GitHub repo"
   - Repository'nizi seçin

3. **Deployment Ayarları:**
   - Railway otomatik olarak `requirements.txt` dosyanızı algılar
   - **Start Command:** `streamlit run app.py --server.port $PORT`
   - Port otomatik olarak `$PORT` environment variable'ından alınır

4. **Model Dosyalarını Yükleme:**
   - Model dosyalarını repository'ye ekleyin
   - Veya Railway'ın "Volumes" özelliğini kullanarak dosyaları yükleyin

5. **Ortam Değişkenleri:**
   - Railway dashboard'da "Variables" sekmesinden ekleyebilirsiniz

#### `Procfile` Oluşturun (Opsiyonel):

```bash
# Procfile (proje kök dizininde)
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

#### Avantajları:
- ✅ Ücretsiz plan (aylık $5 kredi)
- ✅ Otomatik HTTPS
- ✅ Kolay deployment
- ✅ Log görüntüleme

#### Dezavantajları:
- ⚠️ Ücretsiz plan sınırlı kaynaklara sahip
- ⚠️ Uyku modu (inaktiflik sonrası)

---

### Seçenek 3: Render 🎨

Render, modern bir cloud platformudur.

#### Adımlar:

1. **Render Hesabı Oluşturun:**
   - https://render.com adresine gidin
   - GitHub hesabınızla giriş yapın

2. **Yeni Web Service Oluşturun:**
   - "New" > "Web Service"
   - Repository'nizi seçin

3. **Ayarları Yapılandırın:**
   - **Name:** İstediğiniz isim
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

4. **Model Dosyalarını Yükleme:**
   - Model dosyalarını repository'ye ekleyin
   - Veya Render Disk kullanın (ücretli)

#### Avantajları:
- ✅ Ücretsiz plan mevcut
- ✅ Otomatik HTTPS
- ✅ Kolay kullanım

#### Dezavantajları:
- ⚠️ Ücretsiz plan uyku moduna girer (15 dakika inaktiflik)
- ⚠️ Disk alanı sınırlı

---

### Seçenek 4: VPS (DigitalOcean, AWS, vb.) 🖥️

Kendi sunucunuzu yönetmek istiyorsanız:

#### Adımlar:

1. **VPS Satın Alın:**
   - DigitalOcean, AWS EC2, Linode, vb. bir VPS satın alın
   - Ubuntu 20.04 veya üzeri önerilir

2. **Sunucuya Bağlanın:**
   ```bash
   ssh root@sunucu-ip-adresi
   ```

3. **Gerekli Yazılımları Kurun:**
   ```bash
   # Python ve pip
   sudo apt update
   sudo apt install python3 python3-pip git -y
   
   # Nginx (reverse proxy için)
   sudo apt install nginx -y
   ```

4. **Projeyi Klonlayın:**
   ```bash
   cd /var/www
   git clone https://github.com/kullaniciadi/repo-adi.git
   cd repo-adi
   ```

5. **Bağımlılıkları Yükleyin:**
   ```bash
   pip3 install -r requirements.txt
   ```

6. **Systemd Service Oluşturun:**
   ```bash
   sudo nano /etc/systemd/system/streamlit-app.service
   ```
   
   İçeriği:
   ```ini
   [Unit]
   Description=Streamlit Essay Scoring App
   After=network.target
   
   [Service]
   Type=simple
   User=www-data
   WorkingDirectory=/var/www/repo-adi
   Environment="PATH=/usr/bin:/usr/local/bin"
   ExecStart=/usr/local/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

7. **Service'i Başlatın:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable streamlit-app
   sudo systemctl start streamlit-app
   ```

8. **Nginx Reverse Proxy Kurun:**
   ```bash
   sudo nano /etc/nginx/sites-available/streamlit-app
   ```
   
   İçeriği:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_read_timeout 86400;
       }
   }
   ```
   
   ```bash
   sudo ln -s /etc/nginx/sites-available/streamlit-app /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

9. **SSL Sertifikası (Let's Encrypt):**
   ```bash
   sudo apt install certbot python3-certbot-nginx -y
   sudo certbot --nginx -d your-domain.com
   ```

#### Avantajları:
- ✅ Tam kontrol
- ✅ Sınırsız kaynak (planınıza göre)
- ✅ Özel domain
- ✅ Uyku modu yok

#### Dezavantajları:
- ⚠️ Ücretli (aylık $5-20)
- ⚠️ Teknik bilgi gerektirir
- ⚠️ Bakım sizin sorumluluğunuzda

---

## 📦 Model Dosyalarını Yönetme

Model dosyaları büyük olabilir. İşte birkaç seçenek:

### Seçenek A: GitHub LFS (Önerilen)

```bash
# GitHub LFS kurulumu
git lfs install

# .pkl dosyalarını takip et
git lfs track "*.pkl"
git add .gitattributes
git add saved_models/*.pkl
git commit -m "Add models with LFS"
git push
```

### Seçenek B: Cloud Storage (S3, Google Cloud Storage)

Model dosyalarını cloud storage'a yükleyin ve uygulama başlangıcında indirin:

```python
# app.py'ye ekleyin
import boto3
import os

def download_models_from_s3():
    s3 = boto3.client('s3')
    bucket_name = 'your-bucket-name'
    
    for criterion in CRITERIA:
        s3.download_file(
            bucket_name, 
            f'models/model_{criterion}.pkl',
            f'saved_models/model_{criterion}.pkl'
        )
```

### Seçenek C: Model Dosyalarını Repository'ye Eklemek

Küçük model dosyaları için doğrudan repository'ye ekleyebilirsiniz:

```bash
git add saved_models/
git commit -m "Add model files"
git push
```

---

## 🔧 Deployment Öncesi Kontrol Listesi

- [ ] `requirements.txt` dosyası güncel ve tüm bağımlılıkları içeriyor
- [ ] `saved_models/` dizininde tüm model dosyaları mevcut
- [ ] `app.py` dosyası çalışıyor (yerel test)
- [ ] Model dosyaları yükleniyor (yerel test)
- [ ] SBERT modeli indirilebiliyor
- [ ] Git repository hazır ve push edilmiş
- [ ] `.gitignore` dosyası uygun şekilde yapılandırılmış

---

## 🐛 Yaygın Sorunlar ve Çözümleri

### 1. Model Dosyaları Bulunamıyor

**Sorun:** `FileNotFoundError: saved_models/model_TITLE.pkl`

**Çözüm:**
- Model dosyalarının repository'de olduğundan emin olun
- Deployment platformunda dosya yollarını kontrol edin
- Mutlak yol yerine göreli yol kullanın: `os.path.join(os.getcwd(), 'saved_models', ...)`

### 2. SBERT Modeli İndirilemiyor

**Sorun:** İnternet bağlantısı veya proxy sorunları

**Çözüm:**
- Deployment platformunda internet erişimi olduğundan emin olun
- İlk yüklemede model otomatik indirilir (cache'lenir)

### 3. Memory Hatası

**Sorun:** `MemoryError` veya uygulama çöküyor

**Çözüm:**
- Model dosyalarını optimize edin (PCA kullanın)
- Daha küçük SBERT modeli kullanın
- Deployment platformunda daha fazla RAM seçin

### 4. Port Hatası

**Sorun:** `Address already in use`

**Çözüm:**
- `$PORT` environment variable'ını kullanın
- Start command'da `--server.port $PORT` ekleyin

### 5. Yavaş Yükleme

**Sorun:** İlk yükleme çok yavaş

**Çözüm:**
- Model dosyalarını cache'leyin
- Lazy loading kullanın (sadece gerektiğinde yükle)

---

## 📊 Performans Optimizasyonu

1. **Model Caching:**
   ```python
   @st.cache_resource
   def load_models():
       # Model yükleme kodu
       return models
   ```

2. **SBERT Model Caching:**
   ```python
   @st.cache_resource
   def load_sbert_model():
       return SentenceTransformer('all-mpnet-base-v2')
   ```

3. **Lazy Loading:**
   - Modelleri sadece gerektiğinde yükleyin
   - Kullanıcı "Puanla" butonuna tıkladığında yükle

---

## 🔒 Güvenlik Notları

1. **API Keys:** Ortam değişkenlerinde saklayın, kodda hardcode etmeyin
2. **Model Dosyaları:** Hassas veri içeriyorsa şifreleyin
3. **Rate Limiting:** Çok fazla istek gelmesini önleyin
4. **Input Validation:** Kullanıcı girdilerini doğrulayın

---

## 📞 Destek

Sorun yaşarsanız:
1. Deployment platformunun log'larını kontrol edin
2. Yerel olarak test edin: `streamlit run app.py`
3. GitHub Issues'da sorun açın

---

## ✅ Başarılı Deployment Sonrası

Deployment başarılı olduktan sonra:
- ✅ URL'nizi test edin
- ✅ Tüm kriterler için puanlama yapın
- ✅ Performansı izleyin
- ✅ Kullanıcı geri bildirimlerini toplayın

**İyi şanslar! 🚀**

