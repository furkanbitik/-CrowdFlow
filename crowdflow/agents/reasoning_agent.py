"""
CrowdFlow ReasoningAgent Modülü

LangChain ReAct ajanı kullanarak anomali olaylarını analiz eder,
ChromaDB'den geçmiş olaylarla karşılaştırır ve Türkçe doğal dil
açıklamaları üretir.
"""

import time
from typing import Optional

from crowdflow.core.config import yapilandirma
from crowdflow.core.utils import (
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
from crowdflow.memory.chroma_store import ChromaDepo

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
            from langchain.memory import ConversationBufferWindowMemory
            from langchain_openai import ChatOpenAI

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
            AnomaliTipi.PANIK_KACIS: RiskSeviyesi.YUKSEK,
            AnomaliTipi.KAVGA_KUMESI: RiskSeviyesi.YUKSEK,
            AnomaliTipi.DARBOGAZ: RiskSeviyesi.ORTA,
            AnomaliTipi.KISI_DUSMESI: RiskSeviyesi.ORTA,
            AnomaliTipi.ANI_DAGILMA: RiskSeviyesi.KRITIK,
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
            AnomaliTipi.PANIK_KACIS: (
                f"Bölgede {anomali.kisi_sayisi} kişilik bir grup ani ve hızlı bir "
                f"şekilde merkezden uzaklaşıyor. Bu durum panik kaçışına işaret "
                f"ediyor ve kalabalık ezilmesi riski taşıyor.",
                "Güvenlik ekibini alarma geçirin. Kaçış yollarını açık tutun. "
                "Anons sistemiyle sakin kalınması çağrısı yapın.",
            ),
            AnomaliTipi.KAVGA_KUMESI: (
                f"Lokalize bir bölgede {anomali.kisi_sayisi} kişi arasında yoğun "
                f"fiziksel etkileşim tespit edildi. Kavga veya arbede riski mevcut.",
                "Güvenlik personelini olay yerine yönlendirin. Kalabalığı "
                "bölgeden uzaklaştırın. Gerekirse kolluk kuvvetlerini bilgilendirin.",
            ),
            AnomaliTipi.DARBOGAZ: (
                f"Yüksek yoğunluklu bir bölgede {anomali.kisi_sayisi} kişi hareket "
                f"edemez durumda sıkışmış. Darboğaz oluşumu ezilme tehlikesi yaratıyor.",
                "Alternatif güzergahları açın. Kalabalık akışını yönlendirmek için "
                "bariyerler yerleştirin. İtfaiye/ambulans hazır beklesin.",
            ),
            AnomaliTipi.KISI_DUSMESI: (
                "Takip edilen bir kişinin aniden düştüğü veya görüş alanından "
                "kaybolduğu tespit edildi. Tıbbi acil durum olabilir.",
                "En yakın sağlık ekibini yönlendirin. Çevredeki kalabalığı "
                "uzaklaştırarak müdahale alanı oluşturun.",
            ),
            AnomaliTipi.ANI_DAGILMA: (
                f"{anomali.kisi_sayisi} kişilik grup aniden merkezden dışa doğru "
                f"dağılıyor. Bu patlama veya tehdit algısına bağlı olabilir.",
                "Bölgeyi derhal güvenlik kordonu altına alın. Tüm çıkışları "
                "kontrol edin. Bomba imha ekibini bilgilendirin.",
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
