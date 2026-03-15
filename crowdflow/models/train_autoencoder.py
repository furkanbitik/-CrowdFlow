"""
CrowdFlow Otoenkodör Eğitim Modülü

Normal kalabalık davranışı içeren video karelerinden otoenkodörü eğitir.
Eğitim sonucu kaydedilen ağırlıklar anomali tespitinde kullanılır.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from crowdflow.core.config import yapilandirma
from crowdflow.core.utils import logger_olustur
from crowdflow.models.autoencoder import KonvolusyonelOtoenkodor

logger = logger_olustur("OtoenkodorEgitim")


class VideoKareVeriSeti(Dataset):
    """
    Video dosyalarından kare çıkaran veri seti.

    Belirtilen dizindeki tüm video dosyalarından kareler çıkarır
    ve otoenkodör eğitimi için uygun formata dönüştürür.
    """

    def __init__(self, video_dizini: str, goruntu_boyutu: tuple = None):
        """
        Args:
            video_dizini: Video dosyalarının bulunduğu dizin yolu.
            goruntu_boyutu: (genişlik, yükseklik) hedef boyut.
        """
        if goruntu_boyutu is None:
            goruntu_boyutu = yapilandirma.otoenkodor.goruntu_boyutu

        self.goruntu_boyutu = goruntu_boyutu
        self.kareler = []

        self._videolari_yukle(video_dizini)
        logger.info(f"Toplam {len(self.kareler)} kare yüklendi.")

    def _videolari_yukle(self, video_dizini: str) -> None:
        """
        Dizindeki tüm video dosyalarından kareleri çıkarır.

        Args:
            video_dizini: Video dosyaları dizini.
        """
        dizin = Path(video_dizini)
        if not dizin.exists():
            logger.warning(f"Video dizini bulunamadı: {video_dizini}")
            return

        uzantilar = {".mp4", ".avi", ".mkv", ".mov"}
        video_dosyalari = [
            f for f in dizin.iterdir()
            if f.suffix.lower() in uzantilar
        ]

        if not video_dosyalari:
            logger.warning(f"Video dosyası bulunamadı: {video_dizini}")
            return

        for video_yolu in video_dosyalari:
            logger.info(f"Video yükleniyor: {video_yolu.name}")
            self._video_karelerini_cikart(str(video_yolu))

    def _video_karelerini_cikart(self, video_yolu: str) -> None:
        """
        Tek bir videodan kareleri çıkarır.

        Args:
            video_yolu: Video dosya yolu.
        """
        cap = cv2.VideoCapture(video_yolu)
        if not cap.isOpened():
            logger.error(f"Video açılamadı: {video_yolu}")
            return

        kare_sayaci = 0
        while True:
            basarili, kare = cap.read()
            if not basarili:
                break

            # Her 5 karede bir al (veri azaltma)
            if kare_sayaci % 5 == 0:
                kare = cv2.resize(kare, self.goruntu_boyutu)
                kare = kare.astype(np.float32) / 255.0
                kare = np.transpose(kare, (2, 0, 1))  # HWC -> CHW
                self.kareler.append(kare)

            kare_sayaci += 1

        cap.release()
        logger.info(
            f"  -> {len(self.kareler)} kare çıkarıldı (toplam: {kare_sayaci})."
        )

    def __len__(self) -> int:
        return len(self.kareler)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.kareler[idx])


class OtoenkodorEgitici:
    """
    Otoenkodör eğitim yöneticisi.

    Eğitim döngüsü, erken durdurma ve model kaydetme işlemlerini yönetir.
    """

    def __init__(self):
        """Eğiticiyi yapılandırma değerleriyle başlatır."""
        self.ayar = yapilandirma.otoenkodor
        self.cihaz = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = KonvolusyonelOtoenkodor().to(self.cihaz)
        self.optimizor = torch.optim.Adam(
            self.model.parameters(), lr=self.ayar.ogrenme_orani
        )
        self.kayip_fonksiyonu = nn.MSELoss()

        logger.info(f"Eğitim cihazı: {self.cihaz}")

    def egit(self, video_dizini: str = None) -> dict:
        """
        Otoenkodörü eğitir.

        Args:
            video_dizini: Eğitim videoları dizini (varsayılan: config'den).

        Returns:
            Eğitim istatistikleri sözlüğü.
        """
        if video_dizini is None:
            video_dizini = yapilandirma.video.video_dizini

        # Veri setini oluştur
        veri_seti = VideoKareVeriSeti(video_dizini)
        if len(veri_seti) == 0:
            logger.error("Eğitim verisi bulunamadı. Eğitim iptal edildi.")
            return {"durum": "basarisiz", "neden": "veri_yok"}

        yukleyici = DataLoader(
            veri_seti,
            batch_size=self.ayar.batch_boyutu,
            shuffle=True,
            num_workers=0,
        )

        # Eğitim döngüsü
        en_iyi_kayip = float("inf")
        sabir_sayaci = 0
        gecmis = []

        logger.info(
            f"Eğitim başlıyor: {self.ayar.epoch_sayisi} epoch, "
            f"{len(veri_seti)} örnek."
        )

        for epoch in range(self.ayar.epoch_sayisi):
            self.model.train()
            toplam_kayip = 0.0
            batch_sayisi = 0

            for batch in yukleyici:
                batch = batch.to(self.cihaz)

                # İleri geçiş
                yeniden = self.model(batch)
                kayip = self.kayip_fonksiyonu(yeniden, batch)

                # Geri yayılım
                self.optimizor.zero_grad()
                kayip.backward()
                self.optimizor.step()

                toplam_kayip += kayip.item()
                batch_sayisi += 1

            ortalama_kayip = toplam_kayip / max(batch_sayisi, 1)
            gecmis.append(ortalama_kayip)

            logger.info(
                f"Epoch {epoch + 1}/{self.ayar.epoch_sayisi} - "
                f"Kayıp: {ortalama_kayip:.6f}"
            )

            # Erken durdurma kontrolü
            if ortalama_kayip < en_iyi_kayip:
                en_iyi_kayip = ortalama_kayip
                sabir_sayaci = 0
                self._modeli_kaydet()
            else:
                sabir_sayaci += 1
                if sabir_sayaci >= self.ayar.erken_durdurma_sabri:
                    logger.info(
                        f"Erken durdurma: {self.ayar.erken_durdurma_sabri} "
                        f"epoch boyunca iyileşme olmadı."
                    )
                    break

        logger.info(f"Eğitim tamamlandı. En iyi kayıp: {en_iyi_kayip:.6f}")

        return {
            "durum": "basarili",
            "en_iyi_kayip": en_iyi_kayip,
            "toplam_epoch": len(gecmis),
            "gecmis": gecmis,
        }

    def _modeli_kaydet(self) -> None:
        """Modeli belirtilen dosya yoluna kaydeder."""
        kayit_yolu = self.ayar.model_kayit_yolu
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        torch.save(self.model.state_dict(), kayit_yolu)
        logger.info(f"Model kaydedildi: {kayit_yolu}")

    def modeli_yukle(self, yol: str = None) -> None:
        """
        Kaydedilmiş model ağırlıklarını yükler.

        Args:
            yol: Model dosya yolu (varsayılan: config'den).
        """
        if yol is None:
            yol = self.ayar.model_kayit_yolu

        if not os.path.exists(yol):
            logger.warning(f"Model dosyası bulunamadı: {yol}")
            return

        self.model.load_state_dict(
            torch.load(yol, map_location=self.cihaz, weights_only=True)
        )
        self.model.eval()
        logger.info(f"Model yüklendi: {yol}")


if __name__ == "__main__":
    egitici = OtoenkodorEgitici()
    sonuc = egitici.egit()
    logger.info(f"Eğitim sonucu: {sonuc}")
