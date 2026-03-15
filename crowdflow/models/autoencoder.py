"""
CrowdFlow Konvolüsyonel Otoenkodör Modeli

Normal kalabalık davranışını öğrenen konvolüsyonel otoenkodör.
Yeniden yapılandırma hatası yüksek olan kareler anomali olarak işaretlenir.
"""

import torch
import torch.nn as nn

from crowdflow.core.config import yapilandirma


class Enkoder(nn.Module):
    """
    Enkoder ağı: Girdi görüntüsünü sıkıştırılmış gizli temsile dönüştürür.
    """

    def __init__(self, giris_kanallari: int, gizli_boyut: int):
        """
        Args:
            giris_kanallari: Giriş görüntüsü kanal sayısı.
            gizli_boyut: Gizli katman boyutu.
        """
        super().__init__()

        self.ag = nn.Sequential(
            # Katman 1: giris -> 32 kanal
            nn.Conv2d(giris_kanallari, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Katman 2: 32 -> 64 kanal
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Katman 3: 64 -> 128 kanal
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Katman 4: 128 -> gizli_boyut kanal
            nn.Conv2d(128, gizli_boyut, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(gizli_boyut),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enkoder ileri geçişi."""
        return self.ag(x)


class Dekoder(nn.Module):
    """
    Dekoder ağı: Sıkıştırılmış temsilden orijinal görüntüyü yeniden oluşturur.
    """

    def __init__(self, gizli_boyut: int, cikis_kanallari: int):
        """
        Args:
            gizli_boyut: Gizli katman boyutu.
            cikis_kanallari: Çıkış görüntüsü kanal sayısı.
        """
        super().__init__()

        self.ag = nn.Sequential(
            # Katman 1: gizli -> 128 kanal
            nn.ConvTranspose2d(
                gizli_boyut, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Katman 2: 128 -> 64 kanal
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Katman 3: 64 -> 32 kanal
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Katman 4: 32 -> çıkış kanalları
            nn.ConvTranspose2d(
                32, cikis_kanallari, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dekoder ileri geçişi."""
        return self.ag(x)


class KonvolusyonelOtoenkodor(nn.Module):
    """
    Konvolüsyonel Otoenkodör: Normal kalabalık davranışını öğrenir.

    Eğitim sırasında normal kareler üzerinde düşük yeniden yapılandırma
    hatası üretirken, anomali içeren kareler yüksek hata üretir.
    """

    def __init__(
        self,
        giris_kanallari: int = None,
        gizli_boyut: int = None,
    ):
        """
        Args:
            giris_kanallari: Giriş kanal sayısı (varsayılan: config'den).
            gizli_boyut: Gizli katman boyutu (varsayılan: config'den).
        """
        super().__init__()

        if giris_kanallari is None:
            giris_kanallari = yapilandirma.otoenkodor.giris_kanallari
        if gizli_boyut is None:
            gizli_boyut = yapilandirma.otoenkodor.gizli_boyut

        self.enkoder = Enkoder(giris_kanallari, gizli_boyut)
        self.dekoder = Dekoder(gizli_boyut, giris_kanallari)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tam ileri geçiş: enkode -> dekode.

        Args:
            x: Giriş tensörü [batch, kanal, yükseklik, genişlik].

        Returns:
            Yeniden yapılandırılmış tensör.
        """
        gizli = self.enkoder(x)
        yeniden = self.dekoder(gizli)
        return yeniden

    def yeniden_yapilandirma_hatasi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi ile yeniden yapılandırma arasındaki MSE hatasını hesaplar.

        Args:
            x: Giriş tensörü.

        Returns:
            Örnek başına ortalama MSE hatası.
        """
        yeniden = self.forward(x)
        hata = torch.mean((x - yeniden) ** 2, dim=[1, 2, 3])
        return hata

    def anomali_skoru(self, x: torch.Tensor) -> torch.Tensor:
        """
        Anomali skoru hesaplar (yeniden yapılandırma hatası tabanlı).

        Args:
            x: Giriş tensörü.

        Returns:
            Anomali skoru tensörü.
        """
        self.eval()
        with torch.no_grad():
            skor = self.yeniden_yapilandirma_hatasi(x)
        return skor
