"""
CrowdFlow VisualizationAgent Modülü

Gerçek zamanlı video üzerinde sınırlayıcı kutular, takip kimlikleri,
yoğunluk ısı haritası ve anomali uyarı bannerları çizer. Streamlit
dashboard için veri hazırlar.
"""

from collections import deque
from typing import Optional

import cv2
import numpy as np

from crowdflow.core.config import yapilandirma
from crowdflow.core.utils import (
    ANOMALI_TURKCE,
    RISK_EMOJILERI,
    AkillAnaliz,
    AnomaliSonucu,
    KareSonucu,
    OruntSonucu,
    RiskSeviyesi,
    bbox_merkez,
    logger_olustur,
    zaman_damgasi_formatla,
)

logger = logger_olustur("VisualizationAgent")

# Renk sabitleri (BGR formatında)
RENKLER = {
    "yesil": (0, 255, 0),
    "kirmizi": (0, 0, 255),
    "mavi": (255, 0, 0),
    "sari": (0, 255, 255),
    "turuncu": (0, 165, 255),
    "beyaz": (255, 255, 255),
    "siyah": (0, 0, 0),
}

RISK_RENKLERI = {
    RiskSeviyesi.DUSUK: RENKLER["yesil"],
    RiskSeviyesi.ORTA: RENKLER["sari"],
    RiskSeviyesi.YUKSEK: RENKLER["turuncu"],
    RiskSeviyesi.KRITIK: RENKLER["kirmizi"],
}


class VisualizationAgent:
    """
    Görselleştirme Ajanı: Video üzerinde gerçek zamanlı çizim ve dashboard verisi.

    Bounding box, takip ID, yoğunluk haritası ve anomali uyarılarını
    video akışı üzerinde görselleştirir.
    """

    def __init__(self):
        """VisualizationAgent'ı başlatır."""
        self._anomali_logu: deque = deque(
            maxlen=yapilandirma.dashboard.maks_log_satiri
        )
        self._risk_zaman_serisi: deque = deque(maxlen=300)
        self._son_analizler: deque = deque(maxlen=10)
        self._baslatildi: bool = False

    def baslat(self) -> None:
        """VisualizationAgent'ı başlatır."""
        if self._baslatildi:
            logger.warning("VisualizationAgent zaten başlatılmış.")
            return
        self._baslatildi = True
        logger.info("VisualizationAgent başarıyla başlatıldı.")

    def kareyi_ciz(
        self,
        kare_sonucu: KareSonucu,
        orunt_sonucu: Optional[OruntSonucu] = None,
        anomaliler: Optional[list] = None,
        analizler: Optional[list] = None,
    ) -> np.ndarray:
        """
        Video karesinin üzerine tüm görselleştirmeleri çizer.

        Args:
            kare_sonucu: VisionAgent çıktısı.
            orunt_sonucu: PatternAgent çıktısı (opsiyonel).
            anomaliler: Tespit edilen anomali listesi (opsiyonel).
            analizler: ReasoningAgent analiz listesi (opsiyonel).

        Returns:
            Çizimler eklenmiş video karesi.
        """
        if kare_sonucu.kare is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        kare = kare_sonucu.kare.copy()

        # 1. Yoğunluk ısı haritası katmanı
        if orunt_sonucu and orunt_sonucu.yogunluk_izgarasi is not None:
            kare = self._isitma_haritasi_ciz(kare, orunt_sonucu.yogunluk_izgarasi)

        # 2. Sınırlayıcı kutular ve takip kimlikleri
        kare = self._bbox_ciz(kare, kare_sonucu)

        # 3. Yörünge çizgileri
        if orunt_sonucu and orunt_sonucu.yorungeler:
            kare = self._yorunge_ciz(kare, orunt_sonucu.yorungeler)

        # 4. Anomali uyarı bannerleri
        if anomaliler:
            kare = self._anomali_banner_ciz(kare, anomaliler)

        # 5. Analizleri logla
        if analizler:
            for analiz in analizler:
                self._anomali_logu.append(analiz)
                self._son_analizler.append(analiz)

        # 6. Bilgi paneli
        kare = self._bilgi_paneli_ciz(kare, kare_sonucu)

        return kare

    def _bbox_ciz(self, kare: np.ndarray, kare_sonucu: KareSonucu) -> np.ndarray:
        """
        Sınırlayıcı kutular ve takip kimliklerini çizer.

        Args:
            kare: Video karesi.
            kare_sonucu: Tespit sonuçları.

        Returns:
            Çizilmiş kare.
        """
        for tespit in kare_sonucu.tespitler:
            x1, y1, x2, y2 = tespit.bbox
            renk = RENKLER["yesil"]

            # Yüksek hızlı kişiler farklı renk
            hiz = np.sqrt(
                tespit.hiz_vektoru[0] ** 2 + tespit.hiz_vektoru[1] ** 2
            )
            if hiz > yapilandirma.anomali.panik_hiz_esigi:
                renk = RENKLER["kirmizi"]
            elif hiz > yapilandirma.anomali.panik_hiz_esigi * 0.5:
                renk = RENKLER["sari"]

            # Kutu çiz
            cv2.rectangle(kare, (x1, y1), (x2, y2), renk, 2)

            # ID etiketi
            etiket = f"ID:{tespit.id}"
            etiket_boyutu = cv2.getTextSize(
                etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )[0]
            cv2.rectangle(
                kare,
                (x1, y1 - etiket_boyutu[1] - 8),
                (x1 + etiket_boyutu[0] + 4, y1),
                renk,
                -1,
            )
            cv2.putText(
                kare,
                etiket,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                RENKLER["siyah"],
                1,
            )

        return kare

    def _isitma_haritasi_ciz(
        self, kare: np.ndarray, yogunluk: np.ndarray
    ) -> np.ndarray:
        """
        Yoğunluk ısı haritasını video üzerine bindirme olarak çizer.

        Args:
            kare: Video karesi.
            yogunluk: Yoğunluk ızgarası.

        Returns:
            Isı haritası bindirilmiş kare.
        """
        h, w = kare.shape[:2]

        # Yoğunluğu kare boyutuna büyüt
        harita = cv2.resize(yogunluk, (w, h), interpolation=cv2.INTER_LINEAR)

        # 0-255 aralığına normalize et
        harita_norm = (harita * 255).astype(np.uint8)

        # Renk haritası uygula
        renkli = cv2.applyColorMap(harita_norm, cv2.COLORMAP_JET)

        # Yarı saydam bindirme
        alfa = 0.3
        kare = cv2.addWeighted(kare, 1.0 - alfa, renkli, alfa, 0)

        return kare

    def _yorunge_ciz(
        self, kare: np.ndarray, yorungeler: dict
    ) -> np.ndarray:
        """
        Kişi yörüngelerini çizgiler olarak çizer.

        Args:
            kare: Video karesi.
            yorungeler: {id: [(x,y), ...]} yörünge sözlüğü.

        Returns:
            Yörüngeler çizilmiş kare.
        """
        for kisi_id, noktalar in yorungeler.items():
            if len(noktalar) < 2:
                continue

            for i in range(1, len(noktalar)):
                p1 = (int(noktalar[i - 1][0]), int(noktalar[i - 1][1]))
                p2 = (int(noktalar[i][0]), int(noktalar[i][1]))

                # Soluklaşan çizgi (yeni noktalar daha parlak)
                alfa = i / len(noktalar)
                renk = (
                    int(255 * alfa),
                    int(255 * (1 - alfa)),
                    0,
                )
                cv2.line(kare, p1, p2, renk, 1)

        return kare

    def _anomali_banner_ciz(
        self, kare: np.ndarray, anomaliler: list
    ) -> np.ndarray:
        """
        Anomali uyarı bannerlerini kare üzerine çizer.

        Args:
            kare: Video karesi.
            anomaliler: AnomaliSonucu listesi.

        Returns:
            Banner eklenmiş kare.
        """
        h, w = kare.shape[:2]

        for i, anomali in enumerate(anomaliler[:3]):
            tip_turkce = ANOMALI_TURKCE.get(
                anomali.anomali_tipi, str(anomali.anomali_tipi)
            )
            metin = f"UYARI: {tip_turkce} (%{anomali.guven_skoru * 100:.0f})"

            y_offset = 40 + i * 35

            # Yarı saydam kırmızı arka plan
            katman = kare.copy()
            cv2.rectangle(
                katman, (10, y_offset - 25), (w - 10, y_offset + 5),
                RENKLER["kirmizi"], -1
            )
            kare = cv2.addWeighted(kare, 0.7, katman, 0.3, 0)

            cv2.putText(
                kare,
                metin,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                RENKLER["beyaz"],
                2,
            )

        return kare

    def _bilgi_paneli_ciz(
        self, kare: np.ndarray, kare_sonucu: KareSonucu
    ) -> np.ndarray:
        """
        Kare numarası ve kişi sayısı bilgi panelini çizer.

        Args:
            kare: Video karesi.
            kare_sonucu: Kare sonucu.

        Returns:
            Bilgi paneli eklenmiş kare.
        """
        h, w = kare.shape[:2]

        bilgi = (
            f"Kare: {kare_sonucu.kare_no} | "
            f"Kisi: {len(kare_sonucu.tespitler)}"
        )

        # Alt bilgi çubuğu
        cv2.rectangle(kare, (0, h - 30), (w, h), RENKLER["siyah"], -1)
        cv2.putText(
            kare,
            bilgi,
            (10, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            RENKLER["beyaz"],
            1,
        )

        return kare

    def anomali_logunu_al(self) -> list:
        """Anomali log geçmişini döndürür."""
        return list(self._anomali_logu)

    def son_analizleri_al(self) -> list:
        """Son analizleri döndürür."""
        return list(self._son_analizler)

    def risk_zaman_serisini_al(self) -> list:
        """Risk zaman serisi verisini döndürür."""
        return list(self._risk_zaman_serisi)

    def risk_kaydet(self, zaman: float, risk_degeri: float) -> None:
        """
        Risk zaman serisi verisine yeni nokta ekler.

        Args:
            zaman: Zaman damgası.
            risk_degeri: Risk değeri (0-1).
        """
        self._risk_zaman_serisi.append({"zaman": zaman, "risk": risk_degeri})

    def sifirla(self) -> None:
        """Tüm durumu sıfırlar."""
        self._anomali_logu.clear()
        self._risk_zaman_serisi.clear()
        self._son_analizler.clear()
        logger.info("VisualizationAgent sıfırlandı.")

    def kapat(self) -> None:
        """Kaynakları serbest bırakır."""
        self.sifirla()
        self._baslatildi = False
        logger.info("VisualizationAgent kapatıldı.")
