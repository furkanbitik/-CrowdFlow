"""
CrowdFlow AnomalyAgent Modülü (ÇEKIRDEK AJAN)

Konvolüsyonel otoenkodör tabanlı anomali tespiti yapar. Kayan pencere
yeniden yapılandırma hatasını anomali skoru olarak kullanır ve tespit
edilen anomalileri 5 türe sınıflandırır:

1. PANIK_KACIS: Ani yüksek hız, merkezden uzaklaşma
2. KAVGA_KUMESI: Lokalize yoğun hareket + poz çarpışmaları
3. DARBOGAZ: Yüksek yoğunluk + sıfıra yakın hız
4. KISI_DUSMESI: Tek kişinin dikey düşüşü veya kaybolması
5. ANI_DAGILMA: Merkezden dışa doğru patlama örüntüsü
"""

import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch

from crowdflow.core.config import yapilandirma
from crowdflow.core.utils import (
    AnomaliSonucu,
    AnomaliTipi,
    KareSonucu,
    OruntSonucu,
    bbox_merkez,
    logger_olustur,
    oklid_mesafesi,
)
from crowdflow.models.autoencoder import KonvolusyonelOtoenkodor

logger = logger_olustur("AnomalyAgent")


class AnomalyAgent:
    """
    Anomali Tespit Ajanı: Otoenkodör ve kural tabanlı hibrit anomali tespiti.

    Konvolüsyonel otoenkodörün yeniden yapılandırma hatası ile birlikte
    hareket ve yoğunluk tabanlı kuralları kullanarak anomalileri tespit
    ve sınıflandırır.
    """

    def __init__(self):
        """AnomalyAgent'ı yapılandırma değerleriyle başlatır."""
        self._model: Optional[KonvolusyonelOtoenkodor] = None
        self._cihaz = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._esikler = yapilandirma.anomali
        self._kayan_pencere: deque = deque(
            maxlen=yapilandirma.otoenkodor.kayan_pencere_boyutu
        )
        self._onceki_idler: set = set()
        self._onceki_bbox: dict = {}  # {id: bbox}
        self._baslatildi: bool = False

    def baslat(self) -> None:
        """
        Otoenkodör modelini yükler ve ajanı başlatır.

        Eğitilmiş model varsa yükler, yoksa yeni model oluşturur.
        """
        if self._baslatildi:
            logger.warning("AnomalyAgent zaten başlatılmış.")
            return

        logger.info("AnomalyAgent başlatılıyor...")

        self._model = KonvolusyonelOtoenkodor().to(self._cihaz)

        model_yolu = yapilandirma.otoenkodor.model_kayit_yolu
        if os.path.exists(model_yolu):
            self._model.load_state_dict(
                torch.load(model_yolu, map_location=self._cihaz, weights_only=True)
            )
            logger.info(f"Eğitilmiş model yüklendi: {model_yolu}")
        else:
            logger.warning(
                "Eğitilmiş model bulunamadı. "
                "Kural tabanlı tespit aktif, otoenkodör skoru devre dışı."
            )

        self._model.eval()
        self._baslatildi = True
        logger.info("AnomalyAgent başarıyla başlatıldı.")

    def analiz_et(
        self,
        kare_sonucu: KareSonucu,
        orunt_sonucu: OruntSonucu,
    ) -> list:
        """
        Kare ve örüntü verilerini analiz ederek anomali tespiti yapar.

        Args:
            kare_sonucu: VisionAgent çıktısı.
            orunt_sonucu: PatternAgent çıktısı.

        Returns:
            Tespit edilen anomalilerin listesi (AnomaliSonucu).
        """
        if not self._baslatildi:
            raise RuntimeError("AnomalyAgent başlatılmamış. Önce baslat() çağrın.")

        anomaliler = []
        zaman = time.time()

        # Otoenkodör tabanlı anomali skoru
        ae_skoru = self._otoenkodor_skoru_hesapla(kare_sonucu.kare)

        # Kural tabanlı anomali tespitleri
        panik = self._panik_kacis_kontrol(
            kare_sonucu, orunt_sonucu, zaman
        )
        if panik:
            anomaliler.append(panik)

        kavga = self._kavga_kumesi_kontrol(
            kare_sonucu, orunt_sonucu, zaman
        )
        if kavga:
            anomaliler.append(kavga)

        darbogaz = self._darbogaz_kontrol(
            kare_sonucu, orunt_sonucu, zaman
        )
        if darbogaz:
            anomaliler.append(darbogaz)

        dusme = self._kisi_dusmesi_kontrol(
            kare_sonucu, zaman
        )
        if dusme:
            anomaliler.append(dusme)

        dagilma = self._ani_dagilma_kontrol(
            kare_sonucu, orunt_sonucu, zaman
        )
        if dagilma:
            anomaliler.append(dagilma)

        # Otoenkodör skoru ile güven değerlerini güçlendir
        if ae_skoru is not None and ae_skoru > self._esikler.yeniden_yapilandirma_esigi:
            for anomali in anomaliler:
                artis = min(ae_skoru * 5, 0.3)
                anomali.guven_skoru = min(1.0, anomali.guven_skoru + artis)

        # Mevcut durumu güncelle
        self._durumu_guncelle(kare_sonucu)

        if anomaliler:
            logger.info(
                f"Kare {kare_sonucu.kare_no}: {len(anomaliler)} anomali tespit edildi."
            )

        return anomaliler

    def _otoenkodor_skoru_hesapla(
        self, kare: Optional[np.ndarray]
    ) -> Optional[float]:
        """
        Otoenkodör ile yeniden yapılandırma hatası hesaplar.

        Args:
            kare: BGR görüntü karesi.

        Returns:
            Ortalama yeniden yapılandırma hatası veya None.
        """
        if kare is None or self._model is None:
            return None

        boyut = yapilandirma.otoenkodor.goruntu_boyutu
        kucuk = cv2.resize(kare, boyut)
        tensor = (
            torch.from_numpy(kucuk)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        ).to(self._cihaz)

        self._kayan_pencere.append(tensor)

        with torch.no_grad():
            skor = self._model.yeniden_yapilandirma_hatasi(tensor)

        return float(skor.mean())

    def _panik_kacis_kontrol(
        self,
        kare_sonucu: KareSonucu,
        orunt_sonucu: OruntSonucu,
        zaman: float,
    ) -> Optional[AnomaliSonucu]:
        """
        Panik kaçış anomalisi kontrolü.

        Kriterleri:
        - Ortalama hız eşik değerini aşıyor
        - Hareket yönü merkezden dışa doğru (merkezcil)
        """
        if not kare_sonucu.tespitler:
            return None

        hizlar = []
        merkezden_uzaklasma = 0

        kare = kare_sonucu.kare
        if kare is not None:
            kare_merkez = (kare.shape[1] / 2, kare.shape[0] / 2)
        else:
            kare_merkez = (320, 240)

        for tespit in kare_sonucu.tespitler:
            vx, vy = tespit.hiz_vektoru
            hiz = np.sqrt(vx ** 2 + vy ** 2)
            hizlar.append(hiz)

            # Merkezden uzaklaşma kontrolü
            merkez = bbox_merkez(tespit.bbox)
            yon_x = merkez[0] - kare_merkez[0]
            yon_y = merkez[1] - kare_merkez[1]
            hareket_yonu = vx * yon_x + vy * yon_y
            if hareket_yonu > 0:
                merkezden_uzaklasma += 1

        ort_hiz = np.mean(hizlar) if hizlar else 0
        uzaklasma_orani = (
            merkezden_uzaklasma / len(kare_sonucu.tespitler)
            if kare_sonucu.tespitler
            else 0
        )

        if (
            ort_hiz > self._esikler.panik_hiz_esigi
            and uzaklasma_orani > 0.6
        ):
            guven = min(1.0, (ort_hiz / self._esikler.panik_hiz_esigi) * 0.5 + uzaklasma_orani * 0.5)

            return AnomaliSonucu(
                anomali_tipi=AnomaliTipi.PANIK_KACIS,
                guven_skoru=guven,
                izgara_konumu=self._yogun_bolge_bul(orunt_sonucu),
                zaman_damgasi=zaman,
                kisi_sayisi=len(kare_sonucu.tespitler),
                kare_no=kare_sonucu.kare_no,
            )

        return None

    def _kavga_kumesi_kontrol(
        self,
        kare_sonucu: KareSonucu,
        orunt_sonucu: OruntSonucu,
        zaman: float,
    ) -> Optional[AnomaliSonucu]:
        """
        Kavga kümesi anomalisi kontrolü.

        Kriterleri:
        - Lokalize bölgede yüksek yoğunluk
        - Yoğun hareket (yüksek optik akış büyüklüğü)
        - Yakın mesafedeki kişiler arasında çarpışma benzeri hareketler
        """
        if len(kare_sonucu.tespitler) < 2:
            return None

        # Yakın kişi çiftlerini bul
        yakin_ciftler = 0
        yogun_hareket_cifti = 0

        tespitler = kare_sonucu.tespitler
        for i in range(len(tespitler)):
            for j in range(i + 1, len(tespitler)):
                m1 = bbox_merkez(tespitler[i].bbox)
                m2 = bbox_merkez(tespitler[j].bbox)
                mesafe = oklid_mesafesi(m1, m2)

                if mesafe < 100:  # Yakın mesafe
                    yakin_ciftler += 1

                    # Her iki kişi de hareket ediyorsa
                    h1 = np.sqrt(
                        tespitler[i].hiz_vektoru[0] ** 2
                        + tespitler[i].hiz_vektoru[1] ** 2
                    )
                    h2 = np.sqrt(
                        tespitler[j].hiz_vektoru[0] ** 2
                        + tespitler[j].hiz_vektoru[1] ** 2
                    )
                    if h1 > 2.0 and h2 > 2.0:
                        yogun_hareket_cifti += 1

        # Yoğunluk kontrolü
        yogunluk_maks = 0.0
        if orunt_sonucu.yogunluk_izgarasi is not None:
            yogunluk_maks = float(orunt_sonucu.yogunluk_izgarasi.max())

        if (
            yogun_hareket_cifti >= 1
            and yogunluk_maks > self._esikler.kavga_yogunluk_esigi * 0.5
        ):
            guven = min(
                1.0,
                yogun_hareket_cifti * 0.3 + yogunluk_maks * 0.4 + 0.2,
            )

            return AnomaliSonucu(
                anomali_tipi=AnomaliTipi.KAVGA_KUMESI,
                guven_skoru=guven,
                izgara_konumu=self._yogun_bolge_bul(orunt_sonucu),
                zaman_damgasi=zaman,
                kisi_sayisi=yakin_ciftler * 2,
                kare_no=kare_sonucu.kare_no,
            )

        return None

    def _darbogaz_kontrol(
        self,
        kare_sonucu: KareSonucu,
        orunt_sonucu: OruntSonucu,
        zaman: float,
    ) -> Optional[AnomaliSonucu]:
        """
        Darboğaz anomalisi kontrolü.

        Kriterleri:
        - Yüksek yoğunluk (kalabalık birikimi)
        - Sıfıra yakın hız (hareket etmeme)
        """
        if not kare_sonucu.tespitler:
            return None

        # Ortalama hız
        hizlar = [
            np.sqrt(t.hiz_vektoru[0] ** 2 + t.hiz_vektoru[1] ** 2)
            for t in kare_sonucu.tespitler
        ]
        ort_hiz = np.mean(hizlar)

        # Yoğunluk
        yogunluk_maks = 0.0
        if orunt_sonucu.yogunluk_izgarasi is not None:
            yogunluk_maks = float(orunt_sonucu.yogunluk_izgarasi.max())

        if (
            ort_hiz < self._esikler.darbogaz_hiz_esigi
            and yogunluk_maks > self._esikler.darbogaz_yogunluk_esigi
        ):
            guven = min(
                1.0,
                (1.0 - ort_hiz / self._esikler.darbogaz_hiz_esigi) * 0.5
                + yogunluk_maks * 0.5,
            )

            return AnomaliSonucu(
                anomali_tipi=AnomaliTipi.DARBOGAZ,
                guven_skoru=guven,
                izgara_konumu=self._yogun_bolge_bul(orunt_sonucu),
                zaman_damgasi=zaman,
                kisi_sayisi=len(kare_sonucu.tespitler),
                kare_no=kare_sonucu.kare_no,
            )

        return None

    def _kisi_dusmesi_kontrol(
        self,
        kare_sonucu: KareSonucu,
        zaman: float,
    ) -> Optional[AnomaliSonucu]:
        """
        Kişi düşmesi anomalisi kontrolü.

        Kriterleri:
        - Takip edilen kişinin aniden kaybolması
        - Sınırlayıcı kutunun dikey yönde dramatik değişmesi
        """
        mevcut_idler = {t.id for t in kare_sonucu.tespitler}

        # Kaybolma kontrolü
        kaybolan_idler = self._onceki_idler - mevcut_idler

        # Dikey düşüş kontrolü
        for tespit in kare_sonucu.tespitler:
            if tespit.id in self._onceki_bbox:
                onceki = self._onceki_bbox[tespit.id]
                _, onceki_y1, _, onceki_y2 = onceki
                _, mevcut_y1, _, mevcut_y2 = tespit.bbox

                onceki_yukseklik = onceki_y2 - onceki_y1
                mevcut_yukseklik = mevcut_y2 - mevcut_y1

                # Yükseklik dramatik olarak azaldıysa (kişi düştü)
                if (
                    onceki_yukseklik > 0
                    and mevcut_yukseklik > 0
                    and (onceki_yukseklik - mevcut_yukseklik)
                    > self._esikler.dusme_dikey_esigi
                ):
                    guven = min(
                        1.0,
                        (onceki_yukseklik - mevcut_yukseklik)
                        / self._esikler.dusme_dikey_esigi
                        * 0.6,
                    )

                    return AnomaliSonucu(
                        anomali_tipi=AnomaliTipi.KISI_DUSMESI,
                        guven_skoru=guven,
                        izgara_konumu=(
                            int(bbox_merkez(tespit.bbox)[0]),
                            int(bbox_merkez(tespit.bbox)[1]),
                        ),
                        zaman_damgasi=zaman,
                        kisi_sayisi=1,
                        kare_no=kare_sonucu.kare_no,
                    )

        # Aniden kaybolan kişiler (potansiyel düşme)
        if kaybolan_idler and len(kaybolan_idler) <= 2:
            for kid in kaybolan_idler:
                if kid in self._onceki_bbox:
                    return AnomaliSonucu(
                        anomali_tipi=AnomaliTipi.KISI_DUSMESI,
                        guven_skoru=0.4,
                        izgara_konumu=(
                            int(bbox_merkez(self._onceki_bbox[kid])[0]),
                            int(bbox_merkez(self._onceki_bbox[kid])[1]),
                        ),
                        zaman_damgasi=zaman,
                        kisi_sayisi=1,
                        kare_no=kare_sonucu.kare_no,
                    )

        return None

    def _ani_dagilma_kontrol(
        self,
        kare_sonucu: KareSonucu,
        orunt_sonucu: OruntSonucu,
        zaman: float,
    ) -> Optional[AnomaliSonucu]:
        """
        Ani dağılma anomalisi kontrolü.

        Kriterleri:
        - Merkezden dışa doğru patlama örüntüsü
        - Tüm kişilerin merkezden uzaklaşması
        - Yüksek optik akış büyüklüğü
        """
        if len(kare_sonucu.tespitler) < 3:
            return None

        # Kalabalığın ağırlık merkezini hesapla
        merkezler = [bbox_merkez(t.bbox) for t in kare_sonucu.tespitler]
        agirlik_merkezi = (
            np.mean([m[0] for m in merkezler]),
            np.mean([m[1] for m in merkezler]),
        )

        # Her kişinin merkezden uzaklaşma yönünde hareket edip etmediğini kontrol et
        uzaklasma_sayisi = 0
        toplam_uzaklasma_hizi = 0.0

        for tespit in kare_sonucu.tespitler:
            merkez = bbox_merkez(tespit.bbox)
            vx, vy = tespit.hiz_vektoru

            # Merkezden yön vektörü
            dx = merkez[0] - agirlik_merkezi[0]
            dy = merkez[1] - agirlik_merkezi[1]
            norm = np.sqrt(dx ** 2 + dy ** 2)

            if norm > 0:
                dx /= norm
                dy /= norm

                # Hız vektörünün merkezden uzaklaşma bileşeni
                uzaklasma = vx * dx + vy * dy
                if uzaklasma > self._esikler.dagilma_merkezden_uzaklik_esigi:
                    uzaklasma_sayisi += 1
                    toplam_uzaklasma_hizi += uzaklasma

        uzaklasma_orani = uzaklasma_sayisi / len(kare_sonucu.tespitler)

        if uzaklasma_orani > 0.7:
            ort_uzaklasma = (
                toplam_uzaklasma_hizi / uzaklasma_sayisi
                if uzaklasma_sayisi > 0
                else 0
            )
            guven = min(
                1.0,
                uzaklasma_orani * 0.5
                + min(ort_uzaklasma / 10.0, 0.5),
            )

            return AnomaliSonucu(
                anomali_tipi=AnomaliTipi.ANI_DAGILMA,
                guven_skoru=guven,
                izgara_konumu=(
                    int(agirlik_merkezi[0]),
                    int(agirlik_merkezi[1]),
                ),
                zaman_damgasi=zaman,
                kisi_sayisi=len(kare_sonucu.tespitler),
                kare_no=kare_sonucu.kare_no,
            )

        return None

    def _yogun_bolge_bul(self, orunt_sonucu: OruntSonucu) -> tuple:
        """
        Yoğunluk haritasındaki en yoğun bölgenin konumunu bulur.

        Args:
            orunt_sonucu: PatternAgent çıktısı.

        Returns:
            (x, y) en yoğun bölge koordinatları.
        """
        if orunt_sonucu.yogunluk_izgarasi is None:
            return (0, 0)

        idx = np.unravel_index(
            orunt_sonucu.yogunluk_izgarasi.argmax(),
            orunt_sonucu.yogunluk_izgarasi.shape,
        )
        return (int(idx[1]), int(idx[0]))

    def _durumu_guncelle(self, kare_sonucu: KareSonucu) -> None:
        """Önceki kare durumunu günceller."""
        self._onceki_idler = {t.id for t in kare_sonucu.tespitler}
        self._onceki_bbox = {
            t.id: t.bbox for t in kare_sonucu.tespitler
        }

    def sifirla(self) -> None:
        """Tüm durumu sıfırlar."""
        self._kayan_pencere.clear()
        self._onceki_idler.clear()
        self._onceki_bbox.clear()
        logger.info("AnomalyAgent sıfırlandı.")

    def kapat(self) -> None:
        """Kaynakları serbest bırakır."""
        self.sifirla()
        self._baslatildi = False
        logger.info("AnomalyAgent kapatıldı.")
