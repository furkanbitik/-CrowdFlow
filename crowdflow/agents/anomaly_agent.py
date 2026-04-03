"""
CrowdFlow AnomalyAgent Modülü (SUÇ ODAKLI)

Kural tabanlı ve otoenkodör destekli suç tespiti yapar.
Tespit edilen anomaliler 4 suç türüne sınıflandırılır:

1. KAVGA: Yakın mesafedeki kişiler arasında yoğun karşılıklı hareket
2. SALDIRI: Bir kişinin diğerine agresif yaklaşması/kovalama
3. SUPHE_DAVRANIS: Pusuya yatma, ani yön değişimi, takip etme
4. KISI_DUSMESI: Saldırı sonucu yere düşme/kaybolma
"""

import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch

from core.config import yapilandirma
from core.utils import (
    AnomaliSonucu,
    AnomaliTipi,
    KareSonucu,
    OruntSonucu,
    bbox_merkez,
    logger_olustur,
    oklid_mesafesi,
)
from models.autoencoder import KonvolusyonelOtoenkodor

logger = logger_olustur("AnomalyAgent")


class AnomalyAgent:
    """
    Suç Odaklı Anomali Tespit Ajanı.

    Hareket analizi ve mesafe tabanlı kurallarla kavga, saldırı,
    şüpheli davranış ve kişi düşmesi tespiti yapar.
    """

    def __init__(self):
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
        self._onceki_hizlar: dict = {}  # {id: (vx, vy)} — şüpheli davranış için
        self._bekleme_sayaci: dict = {}  # {id: kare_sayisi} — pusu tespiti
        self._baslatildi: bool = False

    def baslat(self) -> None:
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
        if not self._baslatildi:
            raise RuntimeError("AnomalyAgent başlatılmamış. Önce baslat() çağrın.")

        anomaliler = []
        zaman = time.time()

        # Otoenkodör tabanlı anomali skoru
        ae_skoru = self._otoenkodor_skoru_hesapla(kare_sonucu.kare)

        # Suç odaklı anomali tespitleri
        kavga = self._kavga_kontrol(kare_sonucu, orunt_sonucu, zaman)
        if kavga:
            anomaliler.append(kavga)

        saldiri = self._saldiri_kontrol(kare_sonucu, orunt_sonucu, zaman)
        if saldiri:
            anomaliler.append(saldiri)

        suphe = self._suphe_davranis_kontrol(kare_sonucu, orunt_sonucu, zaman)
        if suphe:
            anomaliler.append(suphe)

        dusme = self._kisi_dusmesi_kontrol(kare_sonucu, zaman)
        if dusme:
            anomaliler.append(dusme)

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

    # ── KAVGA TESPİTİ ──────────────────────────────────────────────────────

    def _kavga_kontrol(
        self,
        kare_sonucu: KareSonucu,
        orunt_sonucu: OruntSonucu,
        zaman: float,
    ) -> Optional[AnomaliSonucu]:
        """
        Kavga tespiti: Yakın mesafedeki kişilerin yoğun, düzensiz hareketi.

        Kriterler:
        - En az 2 kişi yakın mesafede (kavga_mesafe_esigi)
        - Her ikisi de yüksek hızda hareket ediyor (kavga_hiz_esigi)
        - Karşılıklı hareket VEYA yoğun salınım (kavga hareketleri)
        """
        if len(kare_sonucu.tespitler) < 2:
            return None

        mesafe_esigi = self._esikler.kavga_mesafe_esigi
        hiz_esigi = self._esikler.kavga_hiz_esigi
        kavga_ciftleri = 0

        tespitler = kare_sonucu.tespitler
        for i in range(len(tespitler)):
            for j in range(i + 1, len(tespitler)):
                m1 = bbox_merkez(tespitler[i].bbox)
                m2 = bbox_merkez(tespitler[j].bbox)
                mesafe = oklid_mesafesi(m1, m2)

                if mesafe >= mesafe_esigi:
                    continue

                v1x, v1y = tespitler[i].hiz_vektoru
                v2x, v2y = tespitler[j].hiz_vektoru
                h1 = np.sqrt(v1x ** 2 + v1y ** 2)
                h2 = np.sqrt(v2x ** 2 + v2y ** 2)

                # En az biri hızlı hareket etmeli
                if h1 < hiz_esigi and h2 < hiz_esigi:
                    continue

                # Karşılıklı hareket: birbirine doğru yaklaşma
                dx = m2[0] - m1[0]
                dy = m2[1] - m1[1]
                norm = np.sqrt(dx ** 2 + dy ** 2)
                if norm > 0:
                    dx /= norm
                    dy /= norm

                yaklasma1 = v1x * dx + v1y * dy
                yaklasma2 = -(v2x * dx + v2y * dy)

                # Senaryo 1: Birbirine doğru hareket
                if yaklasma1 > 0 and yaklasma2 > 0:
                    kavga_ciftleri += 1
                # Senaryo 2: Aynı yerde yoğun salınım (boğuşma)
                elif h1 > hiz_esigi and h2 > hiz_esigi and mesafe < mesafe_esigi * 0.7:
                    kavga_ciftleri += 1
                # Senaryo 3: Biri hızlı, diğeri de yakında ve hareketli
                elif max(h1, h2) > hiz_esigi * 1.5 and min(h1, h2) > hiz_esigi * 0.5:
                    kavga_ciftleri += 1

        if kavga_ciftleri >= self._esikler.kavga_min_cift:
            guven = min(1.0, 0.4 + kavga_ciftleri * 0.3)

            return AnomaliSonucu(
                anomali_tipi=AnomaliTipi.KAVGA,
                guven_skoru=guven,
                izgara_konumu=self._yogun_bolge_bul(orunt_sonucu),
                zaman_damgasi=zaman,
                kisi_sayisi=min(kavga_ciftleri * 2, len(tespitler)),
                kare_no=kare_sonucu.kare_no,
            )

        return None

    # ── SALDIRI TESPİTİ ────────────────────────────────────────────────────

    def _saldiri_kontrol(
        self,
        kare_sonucu: KareSonucu,
        orunt_sonucu: OruntSonucu,
        zaman: float,
    ) -> Optional[AnomaliSonucu]:
        """
        Saldırı tespiti: Bir kişinin diğerine agresif yaklaşması.

        Kriterler:
        - Bir kişi yüksek hızla başka birine doğru koşuyor
        - Hedef kişi durağan veya çok daha yavaş
        - İkisi arasında belirli mesafe var (henüz çok yakın değil)
        """
        if len(kare_sonucu.tespitler) < 2:
            return None

        mesafe_esigi = self._esikler.saldiri_mesafe_esigi
        hiz_esigi = self._esikler.saldiri_hiz_esigi
        hiz_farki_esigi = self._esikler.saldiri_hiz_farki

        tespitler = kare_sonucu.tespitler
        for i in range(len(tespitler)):
            v1x, v1y = tespitler[i].hiz_vektoru
            h1 = np.sqrt(v1x ** 2 + v1y ** 2)

            if h1 < hiz_esigi:
                continue  # Yeterince hızlı değil

            m1 = bbox_merkez(tespitler[i].bbox)

            for j in range(len(tespitler)):
                if i == j:
                    continue

                m2 = bbox_merkez(tespitler[j].bbox)
                mesafe = oklid_mesafesi(m1, m2)

                if mesafe >= mesafe_esigi or mesafe < 30:
                    continue  # Çok uzak veya zaten çok yakın (kavga olur)

                v2x, v2y = tespitler[j].hiz_vektoru
                h2 = np.sqrt(v2x ** 2 + v2y ** 2)

                # Hız farkı: saldırgan çok daha hızlı
                if (h1 - h2) < hiz_farki_esigi:
                    continue

                # Yön kontrolü: saldırgan hedefe doğru mu koşuyor?
                dx = m2[0] - m1[0]
                dy = m2[1] - m1[1]
                norm = np.sqrt(dx ** 2 + dy ** 2)
                if norm > 0:
                    dx /= norm
                    dy /= norm

                yaklasma = v1x * dx + v1y * dy
                if yaklasma > hiz_esigi * 0.5:  # Hedefe doğru yeterli hız bileşeni
                    guven = min(1.0, 0.3 + (h1 / hiz_esigi) * 0.3 + (yaklasma / h1) * 0.3)

                    return AnomaliSonucu(
                        anomali_tipi=AnomaliTipi.SALDIRI,
                        guven_skoru=guven,
                        izgara_konumu=(int(m1[0]), int(m1[1])),
                        zaman_damgasi=zaman,
                        kisi_sayisi=2,
                        kare_no=kare_sonucu.kare_no,
                    )

        return None

    # ── ŞÜPHELİ DAVRANIŞ TESPİTİ ──────────────────────────────────────────

    def _suphe_davranis_kontrol(
        self,
        kare_sonucu: KareSonucu,
        orunt_sonucu: OruntSonucu,
        zaman: float,
    ) -> Optional[AnomaliSonucu]:
        """
        Şüpheli davranış tespiti.

        Kriterler:
        - Ani yön değişimi (hız vektörü ters dönme)
        - Uzun süre aynı yerde bekleme/pusuya yatma
        """
        if not kare_sonucu.tespitler:
            return None

        for tespit in kare_sonucu.tespitler:
            tid = tespit.id
            vx, vy = tespit.hiz_vektoru
            hiz = np.sqrt(vx ** 2 + vy ** 2)

            # --- Ani yön değişimi kontrolü ---
            if tid in self._onceki_hizlar:
                pvx, pvy = self._onceki_hizlar[tid]
                phiz = np.sqrt(pvx ** 2 + pvy ** 2)

                if phiz > 1.0 and hiz > 1.0:
                    # Hız vektörleri arasındaki açı değişimi
                    dot = vx * pvx + vy * pvy
                    cos_aci = dot / (hiz * phiz)

                    # Neredeyse ters yön (-1'e yakın cos) ve yüksek hız
                    if cos_aci < -0.5 and max(hiz, phiz) > self._esikler.suphe_hiz_degisim_esigi:
                        guven = min(1.0, 0.4 + abs(cos_aci) * 0.4)

                        if guven >= self._esikler.guven_minimum:
                            return AnomaliSonucu(
                                anomali_tipi=AnomaliTipi.SUPHE_DAVRANIS,
                                guven_skoru=guven,
                                izgara_konumu=(
                                    int(bbox_merkez(tespit.bbox)[0]),
                                    int(bbox_merkez(tespit.bbox)[1]),
                                ),
                                zaman_damgasi=zaman,
                                kisi_sayisi=1,
                                kare_no=kare_sonucu.kare_no,
                            )

            # --- Bekleme/pusu kontrolü ---
            if hiz < self._esikler.suphe_bekleme_hiz_esigi:
                self._bekleme_sayaci[tid] = self._bekleme_sayaci.get(tid, 0) + 1
            else:
                self._bekleme_sayaci[tid] = 0

            if self._bekleme_sayaci.get(tid, 0) >= self._esikler.suphe_min_bekleme_karesi:
                # Uzun süre hareketsiz bekleme + yakında başka kişiler var mı?
                m1 = bbox_merkez(tespit.bbox)
                yakin_kisi_var = False
                for diger in kare_sonucu.tespitler:
                    if diger.id == tid:
                        continue
                    m2 = bbox_merkez(diger.bbox)
                    if oklid_mesafesi(m1, m2) < 200:
                        yakin_kisi_var = True
                        break

                if yakin_kisi_var:
                    bekleme = self._bekleme_sayaci[tid]
                    guven = min(1.0, 0.3 + (bekleme / (self._esikler.suphe_min_bekleme_karesi * 3)) * 0.5)

                    if guven >= self._esikler.guven_minimum:
                        self._bekleme_sayaci[tid] = 0  # Sıfırla, tekrar tetiklemesin
                        return AnomaliSonucu(
                            anomali_tipi=AnomaliTipi.SUPHE_DAVRANIS,
                            guven_skoru=guven,
                            izgara_konumu=(int(m1[0]), int(m1[1])),
                            zaman_damgasi=zaman,
                            kisi_sayisi=1,
                            kare_no=kare_sonucu.kare_no,
                        )

        return None

    # ── KİŞİ DÜŞMESİ TESPİTİ ──────────────────────────────────────────────

    def _kisi_dusmesi_kontrol(
        self,
        kare_sonucu: KareSonucu,
        zaman: float,
    ) -> Optional[AnomaliSonucu]:
        """
        Kişi düşmesi: bbox yüksekliğinin dramatik düşüşü (saldırı sonucu).
        """
        mevcut_idler = {t.id for t in kare_sonucu.tespitler}

        for tespit in kare_sonucu.tespitler:
            if tespit.id in self._onceki_bbox:
                onceki = self._onceki_bbox[tespit.id]
                _, onceki_y1, _, onceki_y2 = onceki
                _, mevcut_y1, _, mevcut_y2 = tespit.bbox

                onceki_yukseklik = onceki_y2 - onceki_y1
                mevcut_yukseklik = mevcut_y2 - mevcut_y1

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

                    if guven >= self._esikler.guven_minimum:
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

        return None

    # ── YARDIMCI METODLAR ──────────────────────────────────────────────────

    def _yogun_bolge_bul(self, orunt_sonucu: OruntSonucu) -> tuple:
        if orunt_sonucu.yogunluk_izgarasi is None:
            return (0, 0)
        idx = np.unravel_index(
            orunt_sonucu.yogunluk_izgarasi.argmax(),
            orunt_sonucu.yogunluk_izgarasi.shape,
        )
        return (int(idx[1]), int(idx[0]))

    def _durumu_guncelle(self, kare_sonucu: KareSonucu) -> None:
        self._onceki_idler = {t.id for t in kare_sonucu.tespitler}
        self._onceki_bbox = {
            t.id: t.bbox for t in kare_sonucu.tespitler
        }
        self._onceki_hizlar = {
            t.id: t.hiz_vektoru for t in kare_sonucu.tespitler
        }
        # Kaybolmuş ID'lerin bekleme sayacını temizle
        mevcut = {t.id for t in kare_sonucu.tespitler}
        for tid in list(self._bekleme_sayaci.keys()):
            if tid not in mevcut:
                del self._bekleme_sayaci[tid]

    def sifirla(self) -> None:
        self._kayan_pencere.clear()
        self._onceki_idler.clear()
        self._onceki_bbox.clear()
        self._onceki_hizlar.clear()
        self._bekleme_sayaci.clear()
        logger.info("AnomalyAgent sıfırlandı.")

    def kapat(self) -> None:
        self.sifirla()
        self._baslatildi = False
        logger.info("AnomalyAgent kapatıldı.")
