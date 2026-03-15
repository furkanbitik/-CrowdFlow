"""
CrowdFlow Streamlit Dashboard

Gerçek zamanlı kalabalık anomali tespit sisteminin ana kontrol paneli.
Canlı video akışı, anomali logları, yoğunluk ısı haritası ve
risk zaman çizelgesi görüntülenir.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from crowdflow.core.config import yapilandirma
from crowdflow.core.orchestrator import Orkestrator
from crowdflow.core.utils import (
    ANOMALI_TURKCE,
    RISK_EMOJILERI,
    AkillAnaliz,
    RiskSeviyesi,
    VideoModu,
    zaman_damgasi_formatla,
)

# ── Sayfa Yapılandırması ────────────────────────────────────────────────────

st.set_page_config(
    page_title=yapilandirma.dashboard.sayfa_basligi,
    page_icon=yapilandirma.dashboard.sayfa_ikonu,
    layout=yapilandirma.dashboard.yerlesim,
)


def orkestrator_al() -> Orkestrator:
    """Streamlit oturum durumundan orkestratörü alır veya oluşturur."""
    if "orkestrator" not in st.session_state:
        st.session_state.orkestrator = Orkestrator()
    return st.session_state.orkestrator


def baslat_durumu():
    """Oturum durum değişkenlerini başlatır."""
    if "baslatildi" not in st.session_state:
        st.session_state.baslatildi = False
    if "calisiyor" not in st.session_state:
        st.session_state.calisiyor = False
    if "anomali_logu" not in st.session_state:
        st.session_state.anomali_logu = []


def ana_baslik():
    """Ana başlık ve açıklama."""
    st.title("🎯 CrowdFlow")
    st.caption("Ajanistik Kalabalık Anomali Tespit Sistemi")
    st.divider()


def kenar_cubugu():
    """Sol kenar çubuğu kontrolleri."""
    with st.sidebar:
        st.header("⚙️ Kontrol Paneli")

        # Video modu seçimi
        mod = st.radio(
            "Video Modu",
            options=["Webcam", "Video Dosyası"],
            index=0,
        )

        video_yolu = None
        if mod == "Video Dosyası":
            yukle = st.file_uploader(
                "Video Yükle",
                type=["mp4", "avi", "mkv", "mov"],
            )
            if yukle:
                kayit_yolu = Path(yapilandirma.video.video_dizini) / yukle.name
                kayit_yolu.parent.mkdir(parents=True, exist_ok=True)
                with open(kayit_yolu, "wb") as f:
                    f.write(yukle.getvalue())
                video_yolu = str(kayit_yolu)
                st.success(f"Video yüklendi: {yukle.name}")

            # Mevcut videolar
            video_dizini = Path(yapilandirma.video.video_dizini)
            if video_dizini.exists():
                videolar = list(video_dizini.glob("*.mp4")) + list(
                    video_dizini.glob("*.avi")
                )
                if videolar:
                    secilen = st.selectbox(
                        "Veya mevcut video seç",
                        options=[""] + [v.name for v in videolar],
                    )
                    if secilen:
                        video_yolu = str(video_dizini / secilen)

        st.divider()

        # Başlat / Durdur düğmeleri
        col1, col2 = st.columns(2)
        with col1:
            baslat_btn = st.button("▶️ Başlat", use_container_width=True)
        with col2:
            durdur_btn = st.button("⏹️ Durdur", use_container_width=True)

        st.divider()

        # Sistem durumu
        st.subheader("📊 Sistem Durumu")
        ork = orkestrator_al()
        durum = ork.durum_al()

        st.metric("Toplam Kare", durum.toplam_kare)
        st.metric("Toplam Anomali", durum.toplam_anomali)
        st.metric("FPS", f"{durum.fps:.1f}")
        st.metric("Hafıza Olayları", ork.olay_sayisi())

        return mod, video_yolu, baslat_btn, durdur_btn


def canli_video_paneli(kare_yeri):
    """Canlı video gösterim alanı."""
    st.subheader("📹 Canlı Video Akışı")
    return kare_yeri


def anomali_log_paneli():
    """Anomali log paneli."""
    st.subheader("📋 Anomali Logları")

    ork = orkestrator_al()
    analizler = ork.son_analizleri_al()

    if not analizler:
        st.info("Henüz anomali tespit edilmedi.")
        return

    for analiz in reversed(analizler[-10:]):
        if isinstance(analiz, AkillAnaliz):
            tip_turkce = ANOMALI_TURKCE.get(
                analiz.anomali.anomali_tipi,
                str(analiz.anomali.anomali_tipi),
            )
            risk_emoji = RISK_EMOJILERI.get(
                analiz.risk_seviyesi, str(analiz.risk_seviyesi)
            )

            with st.expander(
                f"⚠️ {tip_turkce} - {risk_emoji}", expanded=False
            ):
                st.markdown(f"**Güven:** %{analiz.anomali.guven_skoru * 100:.1f}")
                st.markdown(f"**Kişi Sayısı:** {analiz.anomali.kisi_sayisi}")
                st.markdown(f"**Analiz:** {analiz.analiz_metni}")
                st.markdown(f"**Öneri:** {analiz.oneri}")
                if analiz.tam_rapor:
                    st.code(analiz.tam_rapor, language=None)


def yogunluk_haritasi_paneli():
    """Plotly yoğunluk ısı haritası."""
    st.subheader("🗺️ Yoğunluk Isı Haritası")

    # Örnek yoğunluk verisi (gerçek veri orkestratörden gelecek)
    ork = orkestrator_al()
    durum = ork.durum_al()

    if durum.toplam_kare == 0:
        # Boş ısı haritası göster
        veri = np.zeros(yapilandirma.yogunluk.izgara_boyutu)
    else:
        veri = np.random.rand(*yapilandirma.yogunluk.izgara_boyutu) * 0.3

    fig = px.imshow(
        veri,
        color_continuous_scale=yapilandirma.dashboard.isitma_haritasi_renk_skalasi,
        labels={"color": "Yoğunluk"},
        title="Kalabalık Yoğunluk Dağılımı",
    )
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)


def risk_zaman_cizelgesi_paneli():
    """Risk zaman çizelgesi grafiği."""
    st.subheader("📈 Risk Zaman Çizelgesi")

    ork = orkestrator_al()
    veri = ork.risk_zaman_serisini_al()

    if not veri:
        st.info("Henüz risk verisi yok.")
        return

    zamanlar = [d["zaman"] for d in veri]
    riskler = [d["risk"] for d in veri]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(zamanlar))),
            y=riskler,
            mode="lines+markers",
            name="Risk Skoru",
            line=dict(color="red", width=2),
            fill="tozeroy",
            fillcolor="rgba(255,0,0,0.1)",
        )
    )

    # Risk seviye çizgileri
    fig.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="DÜŞÜK")
    fig.add_hline(y=0.5, line_dash="dash", line_color="yellow", annotation_text="ORTA")
    fig.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="YÜKSEK")
    fig.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="KRİTİK")

    fig.update_layout(
        height=300,
        xaxis_title="Kare",
        yaxis_title="Risk Skoru",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def akil_yurütme_paneli():
    """ReasoningAgent çıktı paneli."""
    st.subheader("🧠 Akıl Yürütme Çıktısı")

    ork = orkestrator_al()
    analizler = ork.son_analizleri_al()

    if not analizler:
        st.info("Henüz analiz raporu yok.")
        return

    # Son analizi göster
    son = analizler[-1]
    if isinstance(son, AkillAnaliz) and son.tam_rapor:
        st.code(son.tam_rapor, language=None)


def video_islem_dongusu(ork, video_kaynagi, kare_yeri):
    """Video işleme döngüsü."""
    if video_kaynagi == "Webcam":
        kaynak = yapilandirma.video.webcam_indeksi
    else:
        kaynak = video_kaynagi

    if kaynak is None:
        st.warning("Video kaynağı seçilmedi.")
        return

    ork.video_baslat(kaynak)

    while st.session_state.calisiyor:
        sonuc = ork.sonraki_kare()
        if sonuc is None:
            st.session_state.calisiyor = False
            st.info("Video sona erdi.")
            break

        cizilmis = sonuc.get("cizilmis_kare")
        if cizilmis is not None:
            # BGR -> RGB dönüşümü
            rgb = cv2.cvtColor(cizilmis, cv2.COLOR_BGR2RGB)
            kare_yeri.image(rgb, channels="RGB", use_container_width=True)

        time.sleep(1.0 / yapilandirma.video.maks_fps)


def main():
    """Ana uygulama fonksiyonu."""
    baslat_durumu()
    ana_baslik()

    # Kenar çubuğu
    mod, video_yolu, baslat_btn, durdur_btn = kenar_cubugu()

    # Orkestratör
    ork = orkestrator_al()

    # Başlat/Durdur işlemleri
    if baslat_btn:
        if not st.session_state.baslatildi:
            with st.spinner("Sistem başlatılıyor..."):
                ork.baslat()
                st.session_state.baslatildi = True
        st.session_state.calisiyor = True

    if durdur_btn:
        st.session_state.calisiyor = False
        ork.durdur()

    # Ana düzen
    col_video, col_sag = st.columns([2, 1])

    with col_video:
        st.subheader("📹 Canlı Video Akışı")
        kare_yeri = st.empty()

        if st.session_state.calisiyor:
            kaynak = video_yolu if mod == "Video Dosyası" else "Webcam"
            video_islem_dongusu(ork, kaynak, kare_yeri)
        else:
            kare_yeri.info(
                "Sistemi başlatmak için kenar çubuğundan '▶️ Başlat' butonuna basın."
            )

    with col_sag:
        anomali_log_paneli()
        akil_yurütme_paneli()

    st.divider()

    # Alt paneller
    col_harita, col_risk = st.columns(2)

    with col_harita:
        yogunluk_haritasi_paneli()

    with col_risk:
        risk_zaman_cizelgesi_paneli()


if __name__ == "__main__":
    main()
