"""
CrowdFlow ChromaDB Vektör Deposu Modülü

Anomali olaylarını vektör olarak saklayarak RAG tabanlı geçmiş olay
karşılaştırması yapılmasını sağlar. ReasoningAgent bu depoyu kullanarak
benzer geçmiş olayları bulur.
"""

import json
import time
from typing import Optional

from crowdflow.core.config import yapilandirma
from crowdflow.core.utils import (
    ANOMALI_TURKCE,
    AnomaliSonucu,
    AnomaliTipi,
    logger_olustur,
    zaman_damgasi_formatla,
)

logger = logger_olustur("ChromaDepo")


class ChromaDepo:
    """
    ChromaDB tabanlı olay hafıza deposu.

    Anomali olaylarını vektör gömme ile saklar ve
    RAG sorgulaması için benzerlik araması sağlar.
    """

    def __init__(self):
        """ChromaDepo'yu yapılandırma değerleriyle başlatır."""
        self._istemci = None
        self._koleksiyon = None
        self._olay_sayaci: int = 0
        self._baslatildi: bool = False

    def baslat(self) -> None:
        """
        ChromaDB istemcisini ve koleksiyonunu başlatır.
        """
        if self._baslatildi:
            logger.warning("ChromaDepo zaten başlatılmış.")
            return

        logger.info("ChromaDB deposu başlatılıyor...")

        try:
            import chromadb
            from chromadb.config import Settings

            ayar = yapilandirma.chroma

            self._istemci = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=ayar.kalici_dizin,
                    anonymized_telemetry=False,
                )
            )

            self._koleksiyon = self._istemci.get_or_create_collection(
                name=ayar.koleksiyon_adi,
                metadata={"hnsw:space": "cosine"},
            )

            mevcut = self._koleksiyon.count()
            self._olay_sayaci = mevcut
            logger.info(
                f"ChromaDB başlatıldı. Mevcut olay sayısı: {mevcut}"
            )

        except ImportError:
            logger.warning(
                "ChromaDB yüklenemedi. Bellek içi mod kullanılacak."
            )
            self._bellekici_mod_baslat()

        except Exception as e:
            logger.warning(
                f"ChromaDB başlatma hatası: {e}. Bellek içi mod kullanılacak."
            )
            self._bellekici_mod_baslat()

        self._baslatildi = True

    def _bellekici_mod_baslat(self) -> None:
        """ChromaDB kullanılamadığında bellek içi depo başlatır."""
        try:
            import chromadb

            self._istemci = chromadb.Client()
            self._koleksiyon = self._istemci.get_or_create_collection(
                name=yapilandirma.chroma.koleksiyon_adi,
            )
            logger.info("Bellek içi ChromaDB modu başlatıldı.")
        except Exception as e:
            logger.error(f"Bellek içi mod da başlatılamadı: {e}")
            self._koleksiyon = None

    def olay_kaydet(self, anomali: AnomaliSonucu, analiz_metni: str = "") -> str:
        """
        Anomali olayını depoya kaydeder.

        Args:
            anomali: AnomaliSonucu nesnesi.
            analiz_metni: ReasoningAgent tarafından üretilen analiz metni.

        Returns:
            Kaydedilen olayın benzersiz kimliği.
        """
        if self._koleksiyon is None:
            logger.warning("Koleksiyon mevcut değil. Olay kaydedilemedi.")
            return ""

        self._olay_sayaci += 1
        olay_id = f"olay_{self._olay_sayaci}_{int(time.time())}"

        tip_turkce = ANOMALI_TURKCE.get(
            anomali.anomali_tipi, str(anomali.anomali_tipi)
        )
        zaman_str = zaman_damgasi_formatla(anomali.zaman_damgasi)

        belge = (
            f"Anomali: {tip_turkce}. "
            f"Güven: %{anomali.guven_skoru * 100:.1f}. "
            f"Konum: {anomali.izgara_konumu}. "
            f"Kişi sayısı: {anomali.kisi_sayisi}. "
            f"Zaman: {zaman_str}. "
            f"Analiz: {analiz_metni}"
        )

        metadata = {
            "anomali_tipi": anomali.anomali_tipi.value,
            "guven_skoru": anomali.guven_skoru,
            "kisi_sayisi": anomali.kisi_sayisi,
            "zaman_damgasi": anomali.zaman_damgasi,
            "kare_no": anomali.kare_no,
            "izgara_x": anomali.izgara_konumu[0],
            "izgara_y": anomali.izgara_konumu[1],
        }

        try:
            self._koleksiyon.add(
                documents=[belge],
                metadatas=[metadata],
                ids=[olay_id],
            )
            logger.debug(f"Olay kaydedildi: {olay_id}")
        except Exception as e:
            logger.error(f"Olay kaydetme hatası: {e}")

        return olay_id

    def benzer_olaylari_bul(
        self,
        anomali: AnomaliSonucu,
        maks_sonuc: int = None,
    ) -> list:
        """
        Verilen anomaliye benzer geçmiş olayları bulur.

        Args:
            anomali: Karşılaştırma için referans anomali.
            maks_sonuc: Döndürülecek maksimum sonuç sayısı.

        Returns:
            Benzer olay sözlüklerinin listesi.
        """
        if self._koleksiyon is None or self._koleksiyon.count() == 0:
            return []

        if maks_sonuc is None:
            maks_sonuc = yapilandirma.chroma.maks_sonuc

        tip_turkce = ANOMALI_TURKCE.get(
            anomali.anomali_tipi, str(anomali.anomali_tipi)
        )

        sorgu = (
            f"Anomali: {tip_turkce}. "
            f"Güven: %{anomali.guven_skoru * 100:.1f}. "
            f"Kişi sayısı: {anomali.kisi_sayisi}."
        )

        try:
            sonuclar = self._koleksiyon.query(
                query_texts=[sorgu],
                n_results=min(maks_sonuc, self._koleksiyon.count()),
            )

            olaylar = []
            if sonuclar and sonuclar["documents"]:
                for i, belge in enumerate(sonuclar["documents"][0]):
                    olay = {
                        "belge": belge,
                        "metadata": (
                            sonuclar["metadatas"][0][i]
                            if sonuclar["metadatas"]
                            else {}
                        ),
                        "mesafe": (
                            sonuclar["distances"][0][i]
                            if sonuclar.get("distances")
                            else None
                        ),
                    }
                    olaylar.append(olay)

            return olaylar

        except Exception as e:
            logger.error(f"Benzer olay arama hatası: {e}")
            return []

    def tum_olaylari_al(self) -> list:
        """
        Depodaki tüm olayları döndürür.

        Returns:
            Tüm olay belgelerinin listesi.
        """
        if self._koleksiyon is None or self._koleksiyon.count() == 0:
            return []

        try:
            sonuc = self._koleksiyon.get()
            return sonuc.get("documents", [])
        except Exception as e:
            logger.error(f"Tüm olayları alma hatası: {e}")
            return []

    def olay_sayisi(self) -> int:
        """Depodaki toplam olay sayısını döndürür."""
        if self._koleksiyon is None:
            return 0
        return self._koleksiyon.count()

    def temizle(self) -> None:
        """Tüm olayları siler."""
        if self._istemci and self._koleksiyon:
            try:
                self._istemci.delete_collection(
                    yapilandirma.chroma.koleksiyon_adi
                )
                self._koleksiyon = self._istemci.get_or_create_collection(
                    name=yapilandirma.chroma.koleksiyon_adi,
                )
                self._olay_sayaci = 0
                logger.info("ChromaDB deposu temizlendi.")
            except Exception as e:
                logger.error(f"Depo temizleme hatası: {e}")

    def kapat(self) -> None:
        """Kaynakları serbest bırakır."""
        self._baslatildi = False
        logger.info("ChromaDepo kapatıldı.")
