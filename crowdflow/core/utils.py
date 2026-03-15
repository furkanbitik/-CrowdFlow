"""
CrowdFlow Yardımcı Araçlar Modülü

Ajanlar arası mesaj geçişi için veri sınıfları, loglama yapılandırması
ve ortak yardımcı fonksiyonlar bu modülde tanımlanır.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from crowdflow.core.config import yapilandirma


# ── Enum Tanımları ──────────────────────────────────────────────────────────


class AnomaliTipi(Enum):
    """Tespit edilebilecek anomali türleri."""

    PANIK_KACIS = "PANIC_FLIGHT"
    KAVGA_KUMESI = "FIGHT_CLUSTER"
    DARBOGAZ = "BOTTLENECK"
    KISI_DUSMESI = "PERSON_FALL"
    ANI_DAGILMA = "SUDDEN_DISPERSAL"


class RiskSeviyesi(Enum):
    """Olay risk seviyeleri."""

    DUSUK = "LOW"
    ORTA = "MEDIUM"
    YUKSEK = "HIGH"
    KRITIK = "CRITICAL"


class VideoModu(Enum):
    """Video giriş modları."""

    WEBCAM = "WEBCAM"
    DATASET = "DATASET"


# ── Anomali Tipi Türkçe Karşılıkları ────────────────────────────────────────

ANOMALI_TURKCE = {
    AnomaliTipi.PANIK_KACIS: "Panik Kaçış",
    AnomaliTipi.KAVGA_KUMESI: "Kavga Kümesi",
    AnomaliTipi.DARBOGAZ: "Darboğaz",
    AnomaliTipi.KISI_DUSMESI: "Kişi Düşmesi",
    AnomaliTipi.ANI_DAGILMA: "Ani Dağılma",
}

RISK_EMOJILERI = {
    RiskSeviyesi.DUSUK: "🟢 LOW",
    RiskSeviyesi.ORTA: "🟡 MEDIUM",
    RiskSeviyesi.YUKSEK: "🔴 HIGH",
    RiskSeviyesi.KRITIK: "🚨 CRITICAL",
}


# ── Veri Sınıfları (Ajanlar Arası Mesajlaşma) ──────────────────────────────


@dataclass
class TespitSonucu:
    """VisionAgent çıktısı: tek bir tespit edilen kişi."""

    id: int
    bbox: tuple  # (x1, y1, x2, y2)
    poz_noktalar: Optional[np.ndarray] = None  # 33x3 mediapipe keypoints
    hiz_vektoru: tuple = (0.0, 0.0)


@dataclass
class KareSonucu:
    """VisionAgent tam kare çıktısı."""

    kare_no: int
    zaman_damgasi: float
    tespitler: list = field(default_factory=list)  # List[TespitSonucu]
    kare: Optional[np.ndarray] = None


@dataclass
class OruntSonucu:
    """PatternAgent çıktısı."""

    kare_no: int
    akis_buyuklugu: Optional[np.ndarray] = None
    akis_yonu: Optional[np.ndarray] = None
    yogunluk_izgarasi: Optional[np.ndarray] = None
    yorungeler: dict = field(default_factory=dict)  # {id: [(x,y), ...]}


@dataclass
class AnomaliSonucu:
    """AnomalyAgent çıktısı."""

    anomali_tipi: AnomaliTipi
    guven_skoru: float
    izgara_konumu: tuple = (0, 0)
    zaman_damgasi: float = 0.0
    kisi_sayisi: int = 0
    kare_no: int = 0


@dataclass
class AkillAnaliz:
    """ReasoningAgent çıktısı."""

    anomali: AnomaliSonucu
    risk_seviyesi: RiskSeviyesi
    analiz_metni: str = ""
    gecmis_karsilastirma: str = ""
    oneri: str = ""
    tam_rapor: str = ""


@dataclass
class SistemDurumu:
    """Orchestrator sistem durumu."""

    aktif: bool = False
    video_modu: VideoModu = VideoModu.WEBCAM
    toplam_kare: int = 0
    toplam_anomali: int = 0
    son_anomali: Optional[AnomaliSonucu] = None
    fps: float = 0.0


# ── Loglama Yapılandırması ──────────────────────────────────────────────────


def logger_olustur(ad: str) -> logging.Logger:
    """
    Türkçe formatlı logger oluşturur.

    Args:
        ad: Logger adı (genellikle modül adı).

    Returns:
        Yapılandırılmış Logger nesnesi.
    """
    logger = logging.getLogger(ad)

    if not logger.handlers:
        seviye = getattr(logging, yapilandirma.log_seviyesi.upper(), logging.INFO)
        logger.setLevel(seviye)

        handler = logging.StreamHandler()
        handler.setLevel(seviye)

        format_str = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


# ── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────


def bbox_merkez(bbox: tuple) -> tuple:
    """
    Sınırlayıcı kutunun merkez noktasını hesaplar.

    Args:
        bbox: (x1, y1, x2, y2) formatında kutu.

    Returns:
        (cx, cy) merkez koordinatları.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_alan(bbox: tuple) -> float:
    """
    Sınırlayıcı kutunun alanını hesaplar.

    Args:
        bbox: (x1, y1, x2, y2) formatında kutu.

    Returns:
        Kutunun piksel alanı.
    """
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def oklid_mesafesi(p1: tuple, p2: tuple) -> float:
    """
    İki nokta arasındaki Öklid mesafesini hesaplar.

    Args:
        p1: (x, y) ilk nokta.
        p2: (x, y) ikinci nokta.

    Returns:
        İki nokta arasındaki mesafe.
    """
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def hiz_hesapla(onceki_konum: tuple, simdiki_konum: tuple, dt: float = 1.0) -> tuple:
    """
    İki konum arasındaki hız vektörünü hesaplar.

    Args:
        onceki_konum: (x, y) önceki kare konumu.
        simdiki_konum: (x, y) mevcut kare konumu.
        dt: Zaman aralığı (kare sayısı cinsinden).

    Returns:
        (vx, vy) hız vektörü.
    """
    if dt == 0:
        return (0.0, 0.0)
    vx = (simdiki_konum[0] - onceki_konum[0]) / dt
    vy = (simdiki_konum[1] - onceki_konum[1]) / dt
    return (vx, vy)


def kare_yeniden_boyutlandir(kare: np.ndarray, genislik: int, yukseklik: int) -> np.ndarray:
    """
    Kareyi belirtilen boyutlara yeniden boyutlandırır.

    Args:
        kare: Giriş görüntüsü (numpy dizisi).
        genislik: Hedef genişlik.
        yukseklik: Hedef yükseklik.

    Returns:
        Yeniden boyutlandırılmış görüntü.
    """
    import cv2

    return cv2.resize(kare, (genislik, yukseklik))


def zaman_damgasi_formatla(ts: float) -> str:
    """
    Unix zaman damgasını okunabilir Türkçe formata çevirir.

    Args:
        ts: Unix zaman damgası.

    Returns:
        'GG.AA.YYYY SS:DD:SN' formatında string.
    """
    return time.strftime("%d.%m.%Y %H:%M:%S", time.localtime(ts))


def normalize_et(dizi: np.ndarray) -> np.ndarray:
    """
    Numpy dizisini 0-1 aralığına normalize eder.

    Args:
        dizi: Giriş numpy dizisi.

    Returns:
        Normalize edilmiş dizi.
    """
    min_val = dizi.min()
    max_val = dizi.max()
    if max_val - min_val == 0:
        return np.zeros_like(dizi, dtype=np.float32)
    return ((dizi - min_val) / (max_val - min_val)).astype(np.float32)


def anomali_raporu_formatla(analiz: AkillAnaliz) -> str:
    """
    AkıllıAnaliz nesnesini belirtilen Türkçe rapor formatına çevirir.

    Args:
        analiz: ReasoningAgent çıktısı.

    Returns:
        Formatlanmış Türkçe rapor metni.
    """
    anomali = analiz.anomali
    tip_turkce = ANOMALI_TURKCE.get(anomali.anomali_tipi, str(anomali.anomali_tipi))
    risk_gosterge = RISK_EMOJILERI.get(analiz.risk_seviyesi, str(analiz.risk_seviyesi))
    zaman_str = zaman_damgasi_formatla(anomali.zaman_damgasi)

    rapor = f"""---
⚠️ ANOMALİ TESPİT EDİLDİ
Zaman     : {zaman_str}
Konum     : {anomali.izgara_konumu}
Tip       : {tip_turkce}
Güven     : %{anomali.guven_skoru * 100:.1f}
Kişi Sayısı: {anomali.kisi_sayisi}
Analiz    : {analiz.analiz_metni}
Geçmiş    : {analiz.gecmis_karsilastirma}
Öneri     : {analiz.oneri}
Risk      : {risk_gosterge}
---"""
    return rapor
