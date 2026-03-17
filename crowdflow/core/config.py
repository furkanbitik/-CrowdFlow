"""
CrowdFlow Yapılandırma Modülü

Tüm sistem ayarları, eşik değerleri ve model parametreleri bu modülde
merkezi olarak yönetilir. Ortam değişkenleri .env dosyasından okunur.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Proje kök dizini (crowdflow/ dizini)
PROJE_KOK = Path(__file__).resolve().parent.parent
CROWDFLOW_KOK = PROJE_KOK


@dataclass
class YOLOAyarlari:
    """YOLOv8 insan tespiti yapılandırması."""

    model_yolu: str = "yolov8n.pt"
    guven_esigi: float = 0.5
    sinif_filtresi: list = field(default_factory=lambda: [0])  # 0 = insan sınıfı
    goruntu_boyutu: int = 640
    cihaz: str = "cpu"


@dataclass
class DeepSORTAyarlari:
    """DeepSORT takip algoritması yapılandırması."""

    maks_yas: int = 30
    n_baslangic: int = 3
    maks_iou_mesafesi: float = 0.7
    gomme_modeli: Optional[str] = "mobilenet"


@dataclass
class MediaPipeAyarlari:
    """MediaPipe poz tahmini yapılandırması."""

    statik_goruntu_modu: bool = False
    model_karmasikligi: int = 1
    duz_isaretler: bool = True
    min_tespit_guveni: float = 0.5
    min_takip_guveni: float = 0.5


@dataclass
class OptikAkisAyarlari:
    """Farneback optik akış parametreleri."""

    piramit_olcegi: float = 0.5
    piramit_katmanlari: int = 3
    pencere_boyutu: int = 15
    iterasyon_sayisi: int = 3
    poligon_n: int = 5
    poligon_sigma: float = 1.2
    bayraklar: int = 0


@dataclass
class YogunlukHaritasiAyarlari:
    """Kalabalık yoğunluk haritası parametreleri."""

    izgara_boyutu: tuple = (20, 20)
    gauss_sigma: float = 10.0
    normalize_et: bool = True


@dataclass
class OtoenkodorAyarlari:
    """Konvolüsyonel otoenkodör model yapılandırması."""

    giris_kanallari: int = 3
    gizli_boyut: int = 64
    ogrenme_orani: float = 1e-3
    batch_boyutu: int = 32
    epoch_sayisi: int = 50
    erken_durdurma_sabri: int = 10
    model_kayit_yolu: str = str(CROWDFLOW_KOK / "models" / "autoencoder_agirliklar.pth")
    kayan_pencere_boyutu: int = 16
    goruntu_boyutu: tuple = (64, 64)


@dataclass
class AnomaliEsikleri:
    """Anomali tespit eşik değerleri."""

    yeniden_yapilandirma_esigi: float = 0.02
    panik_hiz_esigi: float = 5.0
    kavga_yogunluk_esigi: float = 0.8
    darboğaz_hiz_esigi: float = 0.5
    darboğaz_yogunluk_esigi: float = 0.7
    dusme_dikey_esigi: float = 50.0
    dagilma_merkezden_uzaklik_esigi: float = 3.0
    guven_minimum: float = 0.3


@dataclass
class ChromaDBAyarlari:
    """ChromaDB vektör deposu yapılandırması."""

    koleksiyon_adi: str = "crowdflow_olaylar"
    kalici_dizin: str = str(CROWDFLOW_KOK / "memory" / "chroma_db")
    gomme_modeli: str = "all-MiniLM-L6-v2"
    maks_sonuc: int = 5


@dataclass
class LLMAyarlari:
    """LangChain LLM yapılandırması."""

    model_adi: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_anahtari: str = os.getenv("OPENAI_API_KEY", "")
    sicaklik: float = 0.3
    maks_token: int = 1024
    bellek_pencere_boyutu: int = 10


@dataclass
class DashboardAyarlari:
    """Streamlit dashboard yapılandırması."""

    sayfa_basligi: str = "CrowdFlow - Kalabalık Anomali Tespit Sistemi"
    sayfa_ikonu: str = "🎯"
    yerlesim: str = "wide"
    fps_gosterge: bool = True
    maks_log_satiri: int = 100
    isitma_haritasi_renk_skalasi: str = "YlOrRd"


@dataclass
class VideoAyarlari:
    """Video giriş yapılandırması."""

    varsayilan_mod: str = "WEBCAM"  # WEBCAM veya DATASET
    webcam_indeksi: int = 0
    video_dizini: str = str(CROWDFLOW_KOK / "data" / "videos")
    kare_genisligi: int = 640
    kare_yuksekligi: int = 480
    maks_fps: int = 30


@dataclass
class TakipAyarlari:
    """Genel takip ve yörünge parametreleri."""

    yorunge_gecmisi: int = 30  # son 30 kare
    maks_takip_id: int = 1000
    id_sifirlama_suresi: int = 300  # kare sayısı


@dataclass
class CrowdFlowYapilandirma:
    """
    Ana yapılandırma sınıfı.

    Tüm alt yapılandırmaları tek bir noktadan yönetir.
    """

    yolo: YOLOAyarlari = field(default_factory=YOLOAyarlari)
    deepsort: DeepSORTAyarlari = field(default_factory=DeepSORTAyarlari)
    mediapipe: MediaPipeAyarlari = field(default_factory=MediaPipeAyarlari)
    optik_akis: OptikAkisAyarlari = field(default_factory=OptikAkisAyarlari)
    yogunluk: YogunlukHaritasiAyarlari = field(default_factory=YogunlukHaritasiAyarlari)
    otoenkodor: OtoenkodorAyarlari = field(default_factory=OtoenkodorAyarlari)
    anomali: AnomaliEsikleri = field(default_factory=AnomaliEsikleri)
    chroma: ChromaDBAyarlari = field(default_factory=ChromaDBAyarlari)
    llm: LLMAyarlari = field(default_factory=LLMAyarlari)
    dashboard: DashboardAyarlari = field(default_factory=DashboardAyarlari)
    video: VideoAyarlari = field(default_factory=VideoAyarlari)
    takip: TakipAyarlari = field(default_factory=TakipAyarlari)
    log_seviyesi: str = os.getenv("LOG_SEVIYESI", "INFO")
    debug_modu: bool = os.getenv("DEBUG_MODU", "false").lower() == "true"


# Varsayılan global yapılandırma örneği
yapilandirma = CrowdFlowYapilandirma()
