"""
CrowdFlow VisionAgent Modülü

YOLOv8 ile insan tespiti, DeepSORT ile çoklu nesne takibi ve
MediaPipe ile poz tahmini yaparak her kare için tespit sonuçları üretir.
"""

import time
from typing import Optional

import cv2
import numpy as np

from core.config import yapilandirma
from core.utils import (
    KareSonucu,
    TespitSonucu,
    bbox_merkez,
    hiz_hesapla,
    logger_olustur,
)

logger = logger_olustur("VisionAgent")


class VisionAgent:
    """
    Görüş Ajanı: İnsan tespiti, takibi ve poz tahmini.

    YOLOv8 modeli ile karede insan tespiti yapar, DeepSORT ile
    kalıcı kimlik atar ve MediaPipe ile poz noktalarını çıkarır.
    """

    def __init__(self):
        """VisionAgent'ı yapılandırma değerleriyle başlatır."""
        self._yolo_model = None
        self._tracker = None
        self._poz_tahminci = None
        self._onceki_konumlar: dict = {}  # {id: (cx, cy)}
        self._kare_sayaci: int = 0
        self._baslatildi: bool = False

    def baslat(self) -> None:
        """
        Modelleri ve takipçiyi yükler.

        İlk kullanımdan önce çağrılmalıdır.
        """
        if self._baslatildi:
            logger.warning("VisionAgent zaten başlatılmış.")
            return

        logger.info("VisionAgent başlatılıyor...")

        # YOLOv8 modelini yükle
        try:
            from ultralytics import YOLO

            self._yolo_model = YOLO(yapilandirma.yolo.model_yolu)
            logger.info(
                f"YOLOv8 modeli yüklendi: {yapilandirma.yolo.model_yolu}"
            )
        except Exception as e:
            logger.error(f"YOLOv8 modeli yüklenemedi: {e}")
            raise

        # DeepSORT takipçisini başlat
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            # embedder=None: pkg_resources uyumsuzluğunu önlemek için
            # yerleşik embedder kullanılmıyor, dummy embedding sağlanıyor.
            self._tracker = DeepSort(
                max_age=yapilandirma.deepsort.maks_yas,
                n_init=yapilandirma.deepsort.n_baslangic,
                max_iou_distance=yapilandirma.deepsort.maks_iou_mesafesi,
                embedder=None,
            )
            logger.info("DeepSORT takipçisi başlatıldı (IoU tabanlı takip).")
        except Exception as e:
            logger.error(f"DeepSORT başlatılamadı: {e}")
            raise

        # MediaPipe poz tahmincisini başlat
        try:
            import mediapipe as mp

            # Yeni ve eski mediapipe sürümlerini destekle
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
                self._poz_tahminci = mp.solutions.pose.Pose(
                    static_image_mode=yapilandirma.mediapipe.statik_goruntu_modu,
                    model_complexity=yapilandirma.mediapipe.model_karmasikligi,
                    smooth_landmarks=yapilandirma.mediapipe.duz_isaretler,
                    min_detection_confidence=yapilandirma.mediapipe.min_tespit_guveni,
                    min_tracking_confidence=yapilandirma.mediapipe.min_takip_guveni,
                )
                logger.info("MediaPipe poz tahmincisi başlatıldı.")
            else:
                self._poz_tahminci = None
                logger.warning(
                    "MediaPipe solutions API bulunamadı. "
                    "Poz tahmini devre dışı bırakıldı. "
                    "Uyumlu sürüm için: pip install mediapipe==0.10.9"
                )
        except Exception as e:
            self._poz_tahminci = None
            logger.warning(f"MediaPipe başlatılamadı: {e}. Poz tahmini devre dışı.")

        self._baslatildi = True
        logger.info("VisionAgent başarıyla başlatıldı.")

    def kareyi_isle(self, kare: np.ndarray) -> KareSonucu:
        """
        Tek bir video karesini işler.

        Args:
            kare: BGR formatında video karesi (numpy dizisi).

        Returns:
            KareSonucu nesnesi: tespit edilen kişiler ve bilgileri.
        """
        if not self._baslatildi:
            raise RuntimeError("VisionAgent başlatılmamış. Önce baslat() çağrın.")

        self._kare_sayaci += 1
        zaman = time.time()

        # 1. YOLOv8 ile insan tespiti
        tespitler_raw = self._insanlari_tespit_et(kare)

        # 2. DeepSORT ile takip
        takip_sonuclari = self._takip_et(tespitler_raw, kare)

        # 3. Her takip edilen kişi için poz tahmini ve hız hesabı
        tespitler = []
        for track in takip_sonuclari:
            track_id = track["id"]
            bbox = track["bbox"]
            merkez = bbox_merkez(bbox)

            # Hız hesapla
            onceki = self._onceki_konumlar.get(track_id)
            if onceki is not None:
                hiz = hiz_hesapla(onceki, merkez)
            else:
                hiz = (0.0, 0.0)
            self._onceki_konumlar[track_id] = merkez

            # Poz tahmini
            poz = self._poz_tahmin_et(kare, bbox)

            tespit = TespitSonucu(
                id=track_id,
                bbox=bbox,
                poz_noktalar=poz,
                hiz_vektoru=hiz,
            )
            tespitler.append(tespit)

        sonuc = KareSonucu(
            kare_no=self._kare_sayaci,
            zaman_damgasi=zaman,
            tespitler=tespitler,
            kare=kare,
        )

        logger.debug(
            f"Kare {self._kare_sayaci}: {len(tespitler)} kişi tespit edildi."
        )

        return sonuc

    def _insanlari_tespit_et(self, kare: np.ndarray) -> list:
        """
        YOLOv8 ile karede insan tespiti yapar.

        Args:
            kare: BGR formatında görüntü.

        Returns:
            Tespit listesi: [([x1, y1, x2, y2], güven, sınıf_id), ...]
        """
        sonuclar = self._yolo_model(
            kare,
            imgsz=yapilandirma.yolo.goruntu_boyutu,
            conf=yapilandirma.yolo.guven_esigi,
            device=yapilandirma.yolo.cihaz,
            verbose=False,
        )

        tespitler = []
        for sonuc in sonuclar:
            kutular = sonuc.boxes
            if kutular is None:
                continue

            for kutu in kutular:
                sinif_id = int(kutu.cls[0])
                if sinif_id not in yapilandirma.yolo.sinif_filtresi:
                    continue

                guven = float(kutu.conf[0])
                x1, y1, x2, y2 = kutu.xyxy[0].cpu().numpy()
                tespitler.append(
                    ([x1, y1, x2 - x1, y2 - y1], guven, sinif_id)
                )

        return tespitler

    def _takip_et(self, tespitler: list, kare: np.ndarray) -> list:
        """
        DeepSORT ile tespitleri takip eder ve kalıcı ID atar.

        Args:
            tespitler: YOLO tespit listesi.
            kare: Orijinal video karesi.

        Returns:
            Takip sonuçları: [{"id": int, "bbox": (x1,y1,x2,y2)}, ...]
        """
        if not tespitler:
            self._tracker.update_tracks([], frame=kare, embeds=[])
            return []

        # Embedder devre dışı; her tespit için rastgele birim embedding sağla
        # (sıfır vektör cosine similarity'de NaN üretir)
        embeds = []
        for _ in tespitler:
            v = np.random.randn(128).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            embeds.append(v)

        # DeepSORT formatına çevir: ([x, y, w, h], confidence, class)
        tracks = self._tracker.update_tracks(
            tespitler, frame=kare, embeds=embeds
        )

        sonuclar = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = (
                int(ltrb[0]),
                int(ltrb[1]),
                int(ltrb[2]),
                int(ltrb[3]),
            )
            sonuclar.append({"id": track_id, "bbox": bbox})

        return sonuclar

    def _poz_tahmin_et(
        self, kare: np.ndarray, bbox: tuple
    ) -> Optional[np.ndarray]:
        """
        MediaPipe ile belirli bir kişinin poz noktalarını çıkarır.

        Args:
            kare: BGR formatında tam kare.
            bbox: (x1, y1, x2, y2) kişi sınırlayıcı kutusu.

        Returns:
            33x3 numpy dizisi (x, y, visibility) veya tespit edilemezse None.
        """
        x1, y1, x2, y2 = bbox
        h, w = kare.shape[:2]

        # Sınırları kontrol et
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if self._poz_tahminci is None:
            return None

        if x2 <= x1 or y2 <= y1:
            return None

        kisi_karesi = kare[y1:y2, x1:x2]

        if kisi_karesi.size == 0:
            return None

        # MediaPipe RGB bekler
        rgb_kare = cv2.cvtColor(kisi_karesi, cv2.COLOR_BGR2RGB)
        sonuc = self._poz_tahminci.process(rgb_kare)

        if sonuc.pose_landmarks is None:
            return None

        noktalar = np.array(
            [
                [lm.x, lm.y, lm.visibility]
                for lm in sonuc.pose_landmarks.landmark
            ],
            dtype=np.float32,
        )

        return noktalar

    def sifirla(self) -> None:
        """Takip durumunu ve sayaçları sıfırlar."""
        self._onceki_konumlar.clear()
        self._kare_sayaci = 0
        logger.info("VisionAgent sıfırlandı.")

    def kapat(self) -> None:
        """Kaynakları serbest bırakır."""
        if self._poz_tahminci:
            self._poz_tahminci.close()
        self._baslatildi = False
        logger.info("VisionAgent kapatıldı.")
