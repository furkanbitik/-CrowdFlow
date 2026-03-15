# 🎯 CrowdFlow — Ajanistik Kalabalık Anomali Tespit Sistemi

"Person of Interest" dizisindeki gözetleme makinesinden ilham alınarak geliştirilen,
üretim kalitesinde çoklu ajan kalabalık anomali tespit sistemi.

## 📋 Genel Bakış

CrowdFlow, bilgisayar görüşü, derin öğrenme ve LangChain tabanlı ajanistik yapay zeka
kullanarak gerçek zamanlı tehlikeli kalabalık davranışlarını tespit eden bir sistemdir.

### Tespit Edilen Anomali Türleri

| Anomali | Açıklama |
|---------|----------|
| **Panik Kaçış** | Ani yüksek hız, merkezden uzaklaşma hareketi |
| **Kavga Kümesi** | Lokalize yoğun hareket + poz çarpışmaları |
| **Darboğaz** | Yüksek yoğunluk + sıfıra yakın hız |
| **Kişi Düşmesi** | Takip edilen kişinin dikey düşüşü veya kaybolması |
| **Ani Dağılma** | Merkezden dışa doğru patlama örüntüsü |

## 🏗️ Mimari

Sistem 5 uzmanlaşmış ajandan oluşur:

1. **VisionAgent** — YOLOv8 + DeepSORT + MediaPipe ile insan tespiti ve takibi
2. **PatternAgent** — Optik akış, yoğunluk haritası ve yörünge analizi
3. **AnomalyAgent** — Konvolüsyonel otoenkodör + kural tabanlı anomali tespiti
4. **ReasoningAgent** — LangChain ReAct + ChromaDB RAG ile akıl yürütme
5. **VisualizationAgent** — Gerçek zamanlı görselleştirme ve Streamlit dashboard

## 🛠️ Teknoloji Yığını

- **Python 3.10+**
- **YOLOv8** (ultralytics) — İnsan tespiti
- **DeepSORT** (deep_sort_realtime) — Çoklu nesne takibi
- **MediaPipe** — Poz tahmini
- **OpenCV** — Video işleme + optik akış
- **PyTorch** — Otoenkodör anomali modeli
- **LangChain + LangGraph** — Çoklu ajan orkestasyonu
- **ChromaDB** — RAG olay hafızası için vektör deposu
- **Streamlit** — Dashboard arayüzü
- **Plotly** — Etkileşimli ısı haritaları ve grafikler

## 📁 Proje Yapısı

```
crowdflow/
├── agents/
│   ├── vision_agent.py        # YOLOv8 + DeepSORT + MediaPipe
│   ├── pattern_agent.py       # Optik akış + yoğunluk haritası
│   ├── anomaly_agent.py       # Otoenkodör anomali tespiti
│   ├── reasoning_agent.py     # LangChain ReAct akıl yürütme
│   └── visualization_agent.py # Gerçek zamanlı görselleştirme
├── models/
│   ├── autoencoder.py         # Konvolüsyonel otoenkodör modeli
│   └── train_autoencoder.py   # Eğitim betiği
├── core/
│   ├── orchestrator.py        # LangGraph durum makinesi
│   ├── config.py              # Merkezi yapılandırma
│   └── utils.py               # Veri sınıfları ve yardımcılar
├── memory/
│   └── chroma_store.py        # ChromaDB vektör deposu
├── dashboard/
│   └── app.py                 # Streamlit arayüzü
├── data/
│   └── videos/                # Video dosyaları dizini
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Kurulum

### 1. Depoyu Klonlayın

```bash
git clone https://github.com/kullaniciadi/CrowdFlow.git
cd CrowdFlow
```

### 2. Sanal Ortam Oluşturun

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 4. Ortam Değişkenlerini Yapılandırın

```bash
cp .env.example .env
# .env dosyasını düzenleyerek API anahtarlarınızı ekleyin
```

**Not:** ReasoningAgent'ın LLM özelliğini kullanmak için `OPENAI_API_KEY` gereklidir.
API anahtarı olmadan sistem kural tabanlı analiz modunda çalışır.

### 5. (Opsiyonel) Otoenkodörü Eğitin

Normal kalabalık videoları `data/videos/` dizinine koyun ve eğitimi başlatın:

```bash
python -m crowdflow.models.train_autoencoder
```

## ▶️ Çalıştırma

### Streamlit Dashboard (Önerilen)

```bash
streamlit run crowdflow/dashboard/app.py
```

Tarayıcınızda `http://localhost:8501` adresinde dashboard açılacaktır.

### Komut Satırı Modu

```python
from crowdflow.core.orchestrator import Orkestrator

ork = Orkestrator()
ork.baslat()

# Webcam ile çalıştır
ork.calistir()

# Veya video dosyası ile
ork.calistir("data/videos/ornek.mp4")
```

## 📊 Desteklenen Veri Kaynakları

| Mod | Açıklama |
|-----|----------|
| **WEBCAM** | Gerçek zamanlı webcam akışı |
| **DATASET** | Video dosyaları (.mp4, .avi, .mkv, .mov) |

Desteklenen veri setleri:
- UCSD Anomaly Dataset
- UCF-Crime Dataset
- Herhangi bir standart video dosyası

## ⚠️ Anomali Rapor Formatı

```
---
⚠️ ANOMALİ TESPİT EDİLDİ
Zaman     : 15.03.2026 14:30:45
Konum     : (12, 8)
Tip       : Panik Kaçış
Güven     : %87.5
Kişi Sayısı: 15
Analiz    : Bölgede 15 kişilik bir grup ani ve hızlı bir şekilde
            merkezden uzaklaşıyor. Bu durum panik kaçışına işaret ediyor.
Geçmiş    : 3 benzer olay bulundu...
Öneri     : Güvenlik ekibini alarma geçirin. Kaçış yollarını açık tutun.
Risk      : 🔴 HIGH
---
```

## 🔧 Yapılandırma

Tüm ayarlar `crowdflow/core/config.py` dosyasında merkezi olarak yönetilir.
Ortam değişkenleri `.env` dosyasından okunur.

Önemli yapılandırma parametreleri:

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `YOLO model` | yolov8n.pt | YOLO model ağırlıkları |
| `Güven eşiği` | 0.5 | Minimum tespit güveni |
| `Yörünge geçmişi` | 30 kare | Kişi başına takip edilen kare sayısı |
| `Panik hız eşiği` | 5.0 | Panik kaçış hız eşiği |
| `Otoenkodör boyutu` | 64x64 | Model giriş boyutu |

## 📜 Lisans

Bu proje MIT lisansı altında dağıtılmaktadır.
