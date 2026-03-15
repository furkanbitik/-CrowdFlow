"""
CrowdFlow Orkestratör Modülü

LangGraph durum makinesi ile 5 ajanı koordine eden merkezi
orkestrasyon katmanı. Video karelerini sırayla tüm ajanlardan
geçirerek uçtan uca anomali tespit hattı oluşturur.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import cv2
import numpy as np

from crowdflow.agents.anomaly_agent import AnomalyAgent
from crowdflow.agents.pattern_agent import PatternAgent
from crowdflow.agents.reasoning_agent import ReasoningAgent
from crowdflow.agents.vision_agent import VisionAgent
from crowdflow.agents.visualization_agent import VisualizationAgent
from crowdflow.core.config import yapilandirma
from crowdflow.core.utils import (
    AkillAnaliz,
    AnomaliSonucu,
    KareSonucu,
    OruntSonucu,
    RiskSeviyesi,
    SistemDurumu,
    VideoModu,
    logger_olustur,
)
from crowdflow.memory.chroma_store import ChromaDepo

logger = logger_olustur("Orkestrator")


class AjanDurumu(Enum):
    """Ajan işlem durumları."""

    BEKLEMEDE = "beklemede"
    CALISIYOR = "calisiyor"
    TAMAMLANDI = "tamamlandi"
    HATA = "hata"


@dataclass
class HatDurumu:
    """LangGraph durum makinesi durumu."""

    kare: Optional[np.ndarray] = None
    kare_sonucu: Optional[KareSonucu] = None
    orunt_sonucu: Optional[OruntSonucu] = None
    anomaliler: list = field(default_factory=list)
    analizler: list = field(default_factory=list)
    cizilmis_kare: Optional[np.ndarray] = None
    ajan_durumlari: dict = field(default_factory=dict)
    hata_mesaji: str = ""


class Orkestrator:
    """
    Merkezi orkestrasyon sistemi.

    LangGraph tabanlı durum makinesi ile tüm ajanları koordine eder
    ve video işleme hattını yönetir.
    """

    def __init__(self):
        """Orkestratörü başlatır."""
        # Paylaşılan bileşenler
        self._chroma_depo = ChromaDepo()

        # Ajanlar
        self._vision = VisionAgent()
        self._pattern = PatternAgent()
        self._anomaly = AnomalyAgent()
        self._reasoning = ReasoningAgent(chroma_depo=self._chroma_depo)
        self._visualization = VisualizationAgent()

        # Video yakalama
        self._video_yakalama: Optional[cv2.VideoCapture] = None

        # Sistem durumu
        self._durum = SistemDurumu()
        self._hat: Any = None
        self._baslatildi: bool = False

    def baslat(self) -> None:
        """
        Tüm ajanları ve LangGraph hattını başlatır.
        """
        if self._baslatildi:
            logger.warning("Orkestratör zaten başlatılmış.")
            return

        logger.info("CrowdFlow Orkestratör başlatılıyor...")

        # Ajanları başlat
        self._vision.baslat()
        self._pattern.baslat()
        self._anomaly.baslat()
        self._reasoning.baslat()
        self._visualization.baslat()

        # LangGraph hattını oluştur
        self._hat_olustur()

        self._baslatildi = True
        self._durum.aktif = True
        logger.info("CrowdFlow Orkestratör başarıyla başlatıldı.")

    def _hat_olustur(self) -> None:
        """LangGraph durum makinesi hattını oluşturur."""
        try:
            from langgraph.graph import END, StateGraph

            # Durum grafiğini tanımla
            grafik = StateGraph(dict)

            # Düğümleri ekle
            grafik.add_node("vision", self._vision_dugumu)
            grafik.add_node("pattern", self._pattern_dugumu)
            grafik.add_node("anomaly", self._anomaly_dugumu)
            grafik.add_node("reasoning", self._reasoning_dugumu)
            grafik.add_node("visualization", self._visualization_dugumu)

            # Kenarları tanımla (sıralı akış)
            grafik.set_entry_point("vision")
            grafik.add_edge("vision", "pattern")
            grafik.add_edge("pattern", "anomaly")
            grafik.add_conditional_edges(
                "anomaly",
                self._anomali_var_mi,
                {
                    "evet": "reasoning",
                    "hayir": "visualization",
                },
            )
            grafik.add_edge("reasoning", "visualization")
            grafik.add_edge("visualization", END)

            self._hat = grafik.compile()
            logger.info("LangGraph durum makinesi oluşturuldu.")

        except ImportError:
            logger.warning(
                "LangGraph yüklenemedi. Sıralı işleme modu kullanılacak."
            )
            self._hat = None

    def _vision_dugumu(self, durum: dict) -> dict:
        """Vision ajanı düğümü."""
        kare = durum.get("kare")
        if kare is None:
            return durum

        kare_sonucu = self._vision.kareyi_isle(kare)
        durum["kare_sonucu"] = kare_sonucu
        return durum

    def _pattern_dugumu(self, durum: dict) -> dict:
        """Pattern ajanı düğümü."""
        kare_sonucu = durum.get("kare_sonucu")
        if kare_sonucu is None:
            return durum

        orunt_sonucu = self._pattern.kareyi_isle(kare_sonucu)
        durum["orunt_sonucu"] = orunt_sonucu
        return durum

    def _anomaly_dugumu(self, durum: dict) -> dict:
        """Anomaly ajanı düğümü."""
        kare_sonucu = durum.get("kare_sonucu")
        orunt_sonucu = durum.get("orunt_sonucu")
        if kare_sonucu is None or orunt_sonucu is None:
            durum["anomaliler"] = []
            return durum

        anomaliler = self._anomaly.analiz_et(kare_sonucu, orunt_sonucu)
        durum["anomaliler"] = anomaliler
        return durum

    def _anomali_var_mi(self, durum: dict) -> str:
        """Koşullu kenar: Anomali var mı?"""
        anomaliler = durum.get("anomaliler", [])
        return "evet" if anomaliler else "hayir"

    def _reasoning_dugumu(self, durum: dict) -> dict:
        """Reasoning ajanı düğümü."""
        anomaliler = durum.get("anomaliler", [])
        analizler = []

        for anomali in anomaliler:
            analiz = self._reasoning.analiz_et(anomali)
            analizler.append(analiz)

        durum["analizler"] = analizler
        return durum

    def _visualization_dugumu(self, durum: dict) -> dict:
        """Visualization ajanı düğümü."""
        kare_sonucu = durum.get("kare_sonucu")
        if kare_sonucu is None:
            return durum

        cizilmis = self._visualization.kareyi_ciz(
            kare_sonucu=kare_sonucu,
            orunt_sonucu=durum.get("orunt_sonucu"),
            anomaliler=durum.get("anomaliler"),
            analizler=durum.get("analizler"),
        )
        durum["cizilmis_kare"] = cizilmis

        # Risk zaman serisini güncelle
        anomaliler = durum.get("anomaliler", [])
        if anomaliler:
            maks_guven = max(a.guven_skoru for a in anomaliler)
            self._visualization.risk_kaydet(time.time(), maks_guven)
        else:
            self._visualization.risk_kaydet(time.time(), 0.0)

        return durum

    def kare_isle(self, kare: np.ndarray) -> dict:
        """
        Tek bir kareyi tam ajan hattından geçirir.

        Args:
            kare: BGR formatında video karesi.

        Returns:
            İşlenmiş sonuç sözlüğü.
        """
        if not self._baslatildi:
            raise RuntimeError("Orkestratör başlatılmamış. Önce baslat() çağrın.")

        baslangic = time.time()

        giris_durumu = {
            "kare": kare,
            "kare_sonucu": None,
            "orunt_sonucu": None,
            "anomaliler": [],
            "analizler": [],
            "cizilmis_kare": None,
        }

        # LangGraph hattı veya sıralı işleme
        if self._hat is not None:
            sonuc = self._hat.invoke(giris_durumu)
        else:
            sonuc = self._sirasal_isle(giris_durumu)

        # Sistem durumunu güncelle
        self._durum.toplam_kare += 1
        if sonuc.get("anomaliler"):
            self._durum.toplam_anomali += len(sonuc["anomaliler"])
            self._durum.son_anomali = sonuc["anomaliler"][-1]

        gecen = time.time() - baslangic
        self._durum.fps = 1.0 / gecen if gecen > 0 else 0

        return sonuc

    def _sirasal_isle(self, durum: dict) -> dict:
        """
        LangGraph kullanılamadığında sıralı ajan işleme.

        Args:
            durum: Giriş durumu sözlüğü.

        Returns:
            İşlenmiş durum sözlüğü.
        """
        durum = self._vision_dugumu(durum)
        durum = self._pattern_dugumu(durum)
        durum = self._anomaly_dugumu(durum)

        if durum.get("anomaliler"):
            durum = self._reasoning_dugumu(durum)

        durum = self._visualization_dugumu(durum)
        return durum

    def video_baslat(self, kaynak=None) -> None:
        """
        Video yakalamayı başlatır.

        Args:
            kaynak: Video kaynağı (int=webcam, str=dosya yolu).
                    None ise yapılandırmadan alınır.
        """
        if kaynak is None:
            if self._durum.video_modu == VideoModu.WEBCAM:
                kaynak = yapilandirma.video.webcam_indeksi
            else:
                logger.error("Dataset modunda video dosya yolu belirtilmelidir.")
                return

        self._video_yakalama = cv2.VideoCapture(kaynak)

        if not self._video_yakalama.isOpened():
            logger.error(f"Video kaynağı açılamadı: {kaynak}")
            self._video_yakalama = None
            return

        logger.info(f"Video kaynağı açıldı: {kaynak}")

    def sonraki_kare(self) -> Optional[dict]:
        """
        Bir sonraki kareyi okur ve işler.

        Returns:
            İşlenmiş sonuç sözlüğü veya None (video bittiyse).
        """
        if self._video_yakalama is None or not self._video_yakalama.isOpened():
            return None

        basarili, kare = self._video_yakalama.read()
        if not basarili:
            return None

        # Kareyi yeniden boyutlandır
        kare = cv2.resize(
            kare,
            (yapilandirma.video.kare_genisligi, yapilandirma.video.kare_yuksekligi),
        )

        return self.kare_isle(kare)

    def calistir(self, kaynak=None) -> None:
        """
        Video işleme döngüsünü başlatır (OpenCV penceresiyle).

        Args:
            kaynak: Video kaynağı.
        """
        self.video_baslat(kaynak)

        logger.info("Video işleme döngüsü başlatılıyor...")

        while self._durum.aktif:
            sonuc = self.sonraki_kare()
            if sonuc is None:
                break

            cizilmis = sonuc.get("cizilmis_kare")
            if cizilmis is not None:
                cv2.imshow("CrowdFlow - Canli Izleme", cizilmis)

            # Anomali raporlarını yazdır
            for analiz in sonuc.get("analizler", []):
                if isinstance(analiz, AkillAnaliz) and analiz.tam_rapor:
                    logger.info(f"\n{analiz.tam_rapor}")

            tus = cv2.waitKey(1) & 0xFF
            if tus == ord("q"):
                logger.info("Kullanıcı çıkış yaptı.")
                break

        self.durdur()

    def durum_al(self) -> SistemDurumu:
        """Mevcut sistem durumunu döndürür."""
        return self._durum

    def video_modu_ayarla(self, mod: VideoModu) -> None:
        """Video modunu değiştirir."""
        self._durum.video_modu = mod
        logger.info(f"Video modu değiştirildi: {mod.value}")

    def durdur(self) -> None:
        """Video işlemeyi durdurur."""
        self._durum.aktif = False

        if self._video_yakalama:
            self._video_yakalama.release()
            self._video_yakalama = None

        cv2.destroyAllWindows()
        logger.info("Video işleme durduruldu.")

    def sifirla(self) -> None:
        """Tüm ajanları sıfırlar."""
        self._vision.sifirla()
        self._pattern.sifirla()
        self._anomaly.sifirla()
        self._reasoning.sifirla()
        self._visualization.sifirla()
        self._durum = SistemDurumu()
        logger.info("Tüm ajanlar sıfırlandı.")

    def kapat(self) -> None:
        """Tüm bileşenleri kapatır ve kaynakları serbest bırakır."""
        self.durdur()
        self._vision.kapat()
        self._pattern.kapat()
        self._anomaly.kapat()
        self._reasoning.kapat()
        self._visualization.kapat()
        self._chroma_depo.kapat()
        self._baslatildi = False
        logger.info("CrowdFlow Orkestratör kapatıldı.")

    # ── Dashboard için veri erişim metodları ──────────────────────────────

    def anomali_logunu_al(self) -> list:
        """Dashboard için anomali log listesini döndürür."""
        return self._visualization.anomali_logunu_al()

    def son_analizleri_al(self) -> list:
        """Dashboard için son analizleri döndürür."""
        return self._visualization.son_analizleri_al()

    def risk_zaman_serisini_al(self) -> list:
        """Dashboard için risk zaman serisi verisini döndürür."""
        return self._visualization.risk_zaman_serisini_al()

    def olay_sayisi(self) -> int:
        """ChromaDB'deki toplam olay sayısını döndürür."""
        return self._chroma_depo.olay_sayisi()
