"""
CrowdFlow ReasoningAgent Modülü

LangChain ReAct ajanı kullanarak anomali olaylarını analiz eder,
ChromaDB'den geçmiş olaylarla karşılaştırır ve Türkçe doğal dil
açıklamaları üretir.
"""

import time
from typing import Optional

from core.config import yapilandirma
from core.utils import (
    ANOMALI_TURKCE,
    RISK_EMOJILERI,
    AkillAnaliz,
    AnomaliSonucu,
    AnomaliTipi,
    RiskSeviyesi,
    anomali_raporu_formatla,
    logger_olustur,
    zaman_damgasi_formatla,
)
from memory.chroma_store import ChromaDepo

logger = logger_olustur("ReasoningAgent")


class ReasoningAgent:
    """
    Akıl Yürütme Ajanı: LangChain ReAct tabanlı anomali analizi.

    Anomali raporlarını alarak RAG ile geçmiş olaylarla karşılaştırır,
    risk seviyesi belirler ve Türkçe doğal dil açıklamaları üretir.
    """

    def __init__(self, chroma_depo: Optional[ChromaDepo] = None):
        """
        Args:
            chroma_depo: ChromaDB vektör deposu örneği (paylaşımlı kullanım için).
        """
        self._chroma_depo = chroma_depo or ChromaDepo()
        self._llm = None
        self._bellek = None
        self._ajan = None
        self._baslatildi: bool = False
        self._llm_kullanilabilir: bool = False

    def baslat(self) -> None:
        """
        ReasoningAgent'ı başlatır: LLM, bellek ve araçlar.
        """
        if self._baslatildi:
            logger.warning("ReasoningAgent zaten başlatılmış.")
            return

        logger.info("ReasoningAgent başlatılıyor...")

        # ChromaDB deposunu başlat
        if not self._chroma_depo._baslatildi:
            self._chroma_depo.baslat()

        # LangChain bileşenlerini başlat
        self._llm_baslat()

        self._baslatildi = True
        logger.info("ReasoningAgent başarıyla başlatıldı.")

    def _llm_baslat(self) -> None:
        """LangChain LLM ve ajan bileşenlerini başlatır."""
        try:
            from langchain_openai import ChatOpenAI

            try:
                from langchain.memory import ConversationBufferWindowMemory
            except ImportError:
                ConversationBufferWindowMemory = None

            ayar = yapilandirma.llm

            if not ayar.api_anahtari:
                logger.warning(
                    "OpenAI API anahtarı bulunamadı. "
                    "Kural tabanlı analiz kullanılacak."
                )
                self._llm_kullanilabilir = False
                return

            self._llm = ChatOpenAI(
                model=ayar.model_adi,
                api_key=ayar.api_anahtari,
                temperature=ayar.sicaklik,
                max_tokens=ayar.maks_token,
            )

            if ConversationBufferWindowMemory is not None:
                self._bellek = ConversationBufferWindowMemory(
                    k=ayar.bellek_pencere_boyutu,
                    memory_key="chat_history",
                    return_messages=True,
                )

            self._llm_kullanilabilir = True
            logger.info(f"LLM başlatıldı: {ayar.model_adi}")

        except ImportError as e:
            logger.warning(f"LangChain yüklenemedi: {e}")
            self._llm_kullanilabilir = False
        except Exception as e:
            logger.warning(f"LLM başlatma hatası: {e}")
            self._llm_kullanilabilir = False

    def analiz_et(self, anomali: AnomaliSonucu) -> AkillAnaliz:
        """
        Anomali olayını analiz eder ve detaylı rapor üretir.

        Args:
            anomali: AnomalyAgent'tan gelen anomali sonucu.

        Returns:
            AkıllıAnaliz: Risk seviyesi, açıklama ve öneriler.
        """
        if not self._baslatildi:
            raise RuntimeError(
                "ReasoningAgent başlatılmamış. Önce baslat() çağrın."
            )

        # 1. Risk seviyesini belirle
        risk = self._risk_seviyesi_belirle(anomali)

        # 2. Geçmiş olaylarla karşılaştır (RAG)
        gecmis = self._gecmis_olaylari_sorgula(anomali)

        # 3. Analiz metni oluştur
        if self._llm_kullanilabilir:
            analiz = self._llm_ile_analiz(anomali, gecmis, risk)
        else:
            analiz = self._kural_tabanlı_analiz(anomali, gecmis, risk)

        # 4. Olayı depoya kaydet
        self._chroma_depo.olay_kaydet(anomali, analiz.analiz_metni)

        # 5. Tam raporu oluştur
        analiz.tam_rapor = anomali_raporu_formatla(analiz)

        logger.info(
            f"Anomali analiz edildi: {anomali.anomali_tipi.value} - "
            f"Risk: {risk.value}"
        )

        return analiz

    def _risk_seviyesi_belirle(self, anomali: AnomaliSonucu) -> RiskSeviyesi:
        """
        Anomali tipine ve güven skoruna göre risk seviyesi belirler.

        Args:
            anomali: Anomali sonucu.

        Returns:
            RiskSeviyesi enum değeri.
        """
        guven = anomali.guven_skoru
        tip = anomali.anomali_tipi

        # Tip bazlı temel risk
        tip_risk = {
            AnomaliTipi.KAVGA: RiskSeviyesi.YUKSEK,
            AnomaliTipi.SALDIRI: RiskSeviyesi.KRITIK,
            AnomaliTipi.SUPHE_DAVRANIS: RiskSeviyesi.ORTA,
            AnomaliTipi.KISI_DUSMESI: RiskSeviyesi.ORTA,
        }

        temel_risk = tip_risk.get(tip, RiskSeviyesi.DUSUK)

        # Güven skoru ile ayarlama
        if guven >= 0.9:
            # Bir seviye yukarı
            siralama = [
                RiskSeviyesi.DUSUK,
                RiskSeviyesi.ORTA,
                RiskSeviyesi.YUKSEK,
                RiskSeviyesi.KRITIK,
            ]
            idx = siralama.index(temel_risk)
            return siralama[min(idx + 1, len(siralama) - 1)]
        elif guven < 0.4:
            # Bir seviye aşağı
            siralama = [
                RiskSeviyesi.DUSUK,
                RiskSeviyesi.ORTA,
                RiskSeviyesi.YUKSEK,
                RiskSeviyesi.KRITIK,
            ]
            idx = siralama.index(temel_risk)
            return siralama[max(idx - 1, 0)]

        return temel_risk

    def _gecmis_olaylari_sorgula(self, anomali: AnomaliSonucu) -> str:
        """
        ChromaDB'den benzer geçmiş olayları sorgular.

        Args:
            anomali: Referans anomali.

        Returns:
            Geçmiş olayların özet metni.
        """
        benzer_olaylar = self._chroma_depo.benzer_olaylari_bul(anomali)

        if not benzer_olaylar:
            return "Benzer geçmiş olay bulunamadı. Bu ilk kez karşılaşılan bir durum."

        ozet_parcalari = []
        for i, olay in enumerate(benzer_olaylar[:3], 1):
            meta = olay.get("metadata", {})
            tip = meta.get("anomali_tipi", "Bilinmiyor")
            guven = meta.get("guven_skoru", 0)
            ozet_parcalari.append(
                f"  {i}. Geçmiş olay: {tip} (güven: %{guven * 100:.0f})"
            )

        return (
            f"{len(benzer_olaylar)} benzer olay bulundu:\n"
            + "\n".join(ozet_parcalari)
        )

    def _llm_ile_analiz(
        self,
        anomali: AnomaliSonucu,
        gecmis: str,
        risk: RiskSeviyesi,
    ) -> AkillAnaliz:
        """
        LLM kullanarak detaylı analiz üretir.

        Args:
            anomali: Anomali sonucu.
            gecmis: Geçmiş olay karşılaştırma metni.
            risk: Belirlenen risk seviyesi.

        Returns:
            AkıllıAnaliz nesnesi.
        """
        tip_turkce = ANOMALI_TURKCE.get(
            anomali.anomali_tipi, str(anomali.anomali_tipi)
        )

        istem = f"""Sen bir kalabalık güvenlik analiz uzmanısın. Aşağıdaki anomali olayını Türkçe olarak analiz et.

Anomali Bilgileri:
- Tip: {tip_turkce}
- Güven Skoru: %{anomali.guven_skoru * 100:.1f}
- Konum: {anomali.izgara_konumu}
- Kişi Sayısı: {anomali.kisi_sayisi}
- Zaman: {zaman_damgasi_formatla(anomali.zaman_damgasi)}
- Risk Seviyesi: {risk.value}

Geçmiş Olaylar:
{gecmis}

Lütfen şunları üret:
1. ANALIZ: 2-3 cümlelik Türkçe açıklama (durumun ne olduğu ve neden tehlikeli olabileceği)
2. ONERI: Güvenlik ekibinin atması gereken somut adımlar (Türkçe)

Yanıtını şu formatta ver:
ANALIZ: [analiz metni]
ONERI: [öneri metni]"""

        try:
            yanit = self._llm.invoke(istem)
            icerik = yanit.content

            # Yanıtı ayrıştır
            analiz_metni = ""
            oneri_metni = ""

            for satir in icerik.split("\n"):
                if satir.startswith("ANALIZ:"):
                    analiz_metni = satir[7:].strip()
                elif satir.startswith("ONERI:") or satir.startswith("ÖNERİ:"):
                    oneri_metni = satir.split(":", 1)[1].strip()

            if not analiz_metni:
                analiz_metni = icerik[:200]

            # Belleğe ekle
            if self._bellek:
                self._bellek.save_context(
                    {"input": f"Anomali: {tip_turkce}"},
                    {"output": analiz_metni},
                )

            return AkillAnaliz(
                anomali=anomali,
                risk_seviyesi=risk,
                analiz_metni=analiz_metni,
                gecmis_karsilastirma=gecmis,
                oneri=oneri_metni,
            )

        except Exception as e:
            logger.error(f"LLM analiz hatası: {e}")
            return self._kural_tabanlı_analiz(anomali, gecmis, risk)

    def _kural_tabanlı_analiz(
        self,
        anomali: AnomaliSonucu,
        gecmis: str,
        risk: RiskSeviyesi,
    ) -> AkillAnaliz:
        """
        LLM kullanılamadığında kural tabanlı analiz üretir.

        Args:
            anomali: Anomali sonucu.
            gecmis: Geçmiş olay karşılaştırma metni.
            risk: Risk seviyesi.

        Returns:
            AkıllıAnaliz nesnesi.
        """
        tip = anomali.anomali_tipi
        tip_turkce = ANOMALI_TURKCE.get(tip, str(tip))

        analizler = {
            AnomaliTipi.KAVGA: (
                f"Bölgede {anomali.kisi_sayisi} kişi arasında fiziksel kavga "
                f"tespit edildi. Karşılıklı agresif hareketler ve yakın temas "
                f"gözlemleniyor. Ciddi yaralanma riski mevcut.",
                "Güvenlik personelini derhal olay yerine yönlendirin. "
                "Tarafları ayırın. Gerekirse kolluk kuvvetlerini bilgilendirin.",
            ),
            AnomaliTipi.SALDIRI: (
                f"Bir kişinin diğerine hızla yaklaşarak saldırı girişiminde "
                f"bulunduğu tespit edildi. Mağdurun hareket kabiliyeti sınırlı.",
                "Acil güvenlik müdahalesi gerekli. Saldırganı etkisiz hale getirin. "
                "Mağdura tıbbi yardım sağlayın. Kolluk kuvvetlerini arayın.",
            ),
            AnomaliTipi.SUPHE_DAVRANIS: (
                f"Bölgede şüpheli davranış tespit edildi. Ani yön değişiklikleri "
                f"veya belirli bir noktada uzun süreli bekleme gözlemleniyor.",
                "Güvenlik kameralarıyla kişiyi yakından takip edin. "
                "Sivil güvenlik personelini bilgilendirin. Gerekirse kimlik kontrolü yapın.",
            ),
            AnomaliTipi.KISI_DUSMESI: (
                "Takip edilen bir kişinin aniden yere düştüğü tespit edildi. "
                "Tıbbi acil durum veya saldırı sonucu olabilir.",
                "En yakın sağlık ekibini yönlendirin. Çevredeki kişileri "
                "uzaklaştırarak müdahale alanı oluşturun.",
            ),
        }

        analiz_metni, oneri = analizler.get(
            tip,
            (
                f"{tip_turkce} anomalisi tespit edildi.",
                "Güvenlik ekibini bilgilendirin.",
            ),
        )

        return AkillAnaliz(
            anomali=anomali,
            risk_seviyesi=risk,
            analiz_metni=analiz_metni,
            gecmis_karsilastirma=gecmis,
            oneri=oneri,
        )

    def sifirla(self) -> None:
        """Belleği ve durumu sıfırlar."""
        if self._bellek:
            self._bellek.clear()
        logger.info("ReasoningAgent sıfırlandı.")

    def kapat(self) -> None:
        """Kaynakları serbest bırakır."""
        self.sifirla()
        self._baslatildi = False
        logger.info("ReasoningAgent kapatıldı.")
