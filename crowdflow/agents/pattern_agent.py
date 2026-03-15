"""
CrowdFlow PatternAgent Modülü

Yoğun optik akış hesaplama, kalabalık yoğunluk haritası oluşturma
ve kişi yörüngelerini takip etme işlemlerini gerçekleştirir.
"""

from collections import defaultdict, deque
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from crowdflow.core.config import yapilandirma
from crowdflow.core.utils import (
    KareSonucu,
    OruntSonucu,
    bbox_merkez,
    logger_olustur,
    normalize_et,
)

logger = logger_olustur("PatternAgent")


class PatternAgent:
    """
    Örüntü Ajanı: Optik akış, yoğunluk haritası ve yörünge analizi.

    Dense Optical Flow (Farneback) ile hareket analizi yapar,
    Gauss çekirdek tahmini ile yoğunluk haritası oluşturur ve
    her takip edilen kişinin yörünge geçmişini tutar.
    """

    def __init__(self):
        """PatternAgent'ı yapılandırma değerleriyle başlatır."""
        self._onceki_gri: Optional[np.ndarray] = None
        self._yorungeler: dict = defaultdict(
            lambda: deque(maxlen=yapilandirma.takip.yorunge_gecmisi)
        )
        self._kare_boyutu: Optional[tuple] = None
        self._baslatildi: bool = False

    def baslat(self) -> None:
        """PatternAgent'ı başlatır."""
        if self._baslatildi:
            logger.warning("PatternAgent zaten başlatılmış.")
            return

        self._baslatildi = True
        logger.info("PatternAgent başarıyla başlatıldı.")

    def kareyi_isle(self, kare_sonucu: KareSonucu) -> OruntSonucu:
        """
        Bir kare sonucunu alıp örüntü analizini gerçekleştirir.

        Args:
            kare_sonucu: VisionAgent'tan gelen KareSonucu nesnesi.

        Returns:
            OruntSonucu: Optik akış, yoğunluk ve yörünge bilgileri.
        """
        if not self._baslatildi:
            raise RuntimeError("PatternAgent başlatılmamış. Önce baslat() çağrın.")

        kare = kare_sonucu.kare
        if kare is None:
            return OruntSonucu(kare_no=kare_sonucu.kare_no)

        gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
        self._kare_boyutu = (kare.shape[1], kare.shape[0])  # (genişlik, yükseklik)

        # 1. Optik akış hesapla
        akis_buyuklugu, akis_yonu = self._optik_akis_hesapla(gri)

        # 2. Yoğunluk haritası oluştur
        yogunluk_izgarasi = self._yogunluk_haritasi_olustur(
            kare_sonucu.tespitler
        )

        # 3. Yörüngeleri güncelle
        self._yorungeleri_guncelle(kare_sonucu.tespitler)

        # Önceki kareyi güncelle
        self._onceki_gri = gri.copy()

        sonuc = OruntSonucu(
            kare_no=kare_sonucu.kare_no,
            akis_buyuklugu=akis_buyuklugu,
            akis_yonu=akis_yonu,
            yogunluk_izgarasi=yogunluk_izgarasi,
            yorungeler={
                k: list(v) for k, v in self._yorungeler.items()
            },
        )

        logger.debug(
            f"Kare {kare_sonucu.kare_no}: Akış hesaplandı, "
            f"{len(self._yorungeler)} kişi yörüngesi takip ediliyor."
        )

        return sonuc

    def _optik_akis_hesapla(
        self, gri: np.ndarray
    ) -> tuple:
        """
        Farneback yöntemiyle yoğun optik akış hesaplar.

        Args:
            gri: Gri tonlamalı mevcut kare.

        Returns:
            (akış_büyüklüğü, akış_yönü) numpy dizileri veya (None, None).
        """
        if self._onceki_gri is None:
            return None, None

        ayar = yapilandirma.optik_akis

        akis = cv2.calcOpticalFlowFarneback(
            self._onceki_gri,
            gri,
            None,
            pyr_scale=ayar.piramit_olcegi,
            levels=ayar.piramit_katmanlari,
            winsize=ayar.pencere_boyutu,
            iterations=ayar.iterasyon_sayisi,
            poly_n=ayar.poligon_n,
            poly_sigma=ayar.poligon_sigma,
            flags=ayar.bayraklar,
        )

        buyukluk, yon = cv2.cartToPolar(akis[..., 0], akis[..., 1])

        return buyukluk, yon

    def _yogunluk_haritasi_olustur(
        self, tespitler: list
    ) -> np.ndarray:
        """
        Gauss çekirdek tahmini ile kalabalık yoğunluk haritası oluşturur.

        Args:
            tespitler: TespitSonucu listesi.

        Returns:
            Yoğunluk ızgarası (numpy dizisi).
        """
        ayar = yapilandirma.yogunluk

        if self._kare_boyutu is None:
            return np.zeros(ayar.izgara_boyutu, dtype=np.float32)

        gen, yuk = self._kare_boyutu
        yogunluk = np.zeros((yuk, gen), dtype=np.float32)

        for tespit in tespitler:
            cx, cy = bbox_merkez(tespit.bbox)
            cx, cy = int(cx), int(cy)
            if 0 <= cx < gen and 0 <= cy < yuk:
                yogunluk[cy, cx] += 1.0

        # Gauss bulanıklaştırma uygula
        yogunluk = gaussian_filter(yogunluk, sigma=ayar.gauss_sigma)

        # İzgara boyutuna küçült
        izgara_y, izgara_x = ayar.izgara_boyutu
        yogunluk_izgara = cv2.resize(
            yogunluk, (izgara_x, izgara_y), interpolation=cv2.INTER_AREA
        )

        if ayar.normalize_et:
            yogunluk_izgara = normalize_et(yogunluk_izgara)

        return yogunluk_izgara

    def _yorungeleri_guncelle(self, tespitler: list) -> None:
        """
        Her takip edilen kişinin yörünge geçmişini günceller.

        Args:
            tespitler: TespitSonucu listesi.
        """
        aktif_idler = set()

        for tespit in tespitler:
            merkez = bbox_merkez(tespit.bbox)
            self._yorungeler[tespit.id].append(merkez)
            aktif_idler.add(tespit.id)

        # Artık görünmeyen kişilerin yörüngelerini temizle
        silinecek = [
            k for k in self._yorungeler
            if k not in aktif_idler and len(self._yorungeler[k]) == 0
        ]
        for k in silinecek:
            del self._yorungeler[k]

    def yorunge_al(self, kisi_id: int) -> list:
        """
        Belirli bir kişinin yörünge geçmişini döndürür.

        Args:
            kisi_id: Takip kimliği.

        Returns:
            [(x, y), ...] koordinat listesi.
        """
        return list(self._yorungeler.get(kisi_id, []))

    def ortalama_hiz_al(self, kisi_id: int) -> float:
        """
        Belirli bir kişinin son yörüngesindeki ortalama hızını hesaplar.

        Args:
            kisi_id: Takip kimliği.

        Returns:
            Ortalama hız değeri (piksel/kare).
        """
        yorunge = self.yorunge_al(kisi_id)
        if len(yorunge) < 2:
            return 0.0

        toplam = 0.0
        for i in range(1, len(yorunge)):
            dx = yorunge[i][0] - yorunge[i - 1][0]
            dy = yorunge[i][1] - yorunge[i - 1][1]
            toplam += np.sqrt(dx ** 2 + dy ** 2)

        return toplam / (len(yorunge) - 1)

    def sifirla(self) -> None:
        """Tüm durumu sıfırlar."""
        self._onceki_gri = None
        self._yorungeler.clear()
        self._kare_boyutu = None
        logger.info("PatternAgent sıfırlandı.")

    def kapat(self) -> None:
        """Kaynakları serbest bırakır."""
        self.sifirla()
        self._baslatildi = False
        logger.info("PatternAgent kapatıldı.")
