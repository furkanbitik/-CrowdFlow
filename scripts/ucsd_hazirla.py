"""
UCSD Veri Seti Hazırlama Betiği

UCSD Anomaly Dataset'teki .tif görüntü karelerini
otoenkodör eğitimi için uygun video dosyalarına dönüştürür.

Kullanım:
    python scripts/ucsd_hazirla.py
"""

import glob
import os

import cv2

# ── Yapılandırma ────────────────────────────────────────────────────────────

UCSD_DIZIN = "crowdflow/data/ucsd_raw/UCSD_Anomaly_Dataset"
CIKIS_DIZIN = "crowdflow/data/videos"
FPS = 10

# ── Klasörleri Oluştur ──────────────────────────────────────────────────────

os.makedirs(CIKIS_DIZIN, exist_ok=True)

# ── Her Train Klibini Ayrı Video Yap ────────────────────────────────────────

for ped in ["UCSDped1", "UCSDped2"]:
    train_dizin = os.path.join(UCSD_DIZIN, ped, "Train")

    if not os.path.exists(train_dizin):
        print(f"[UYARI] Bulunamadı: {train_dizin}")
        continue

    # Her klibi bul (Train001, Train002, ...)
    klipler = sorted([
        d for d in os.listdir(train_dizin)
        if os.path.isdir(os.path.join(train_dizin, d))
    ])

    print(f"\n{ped}: {len(klipler)} normal klip bulundu.")

    for klip_adi in klipler:
        klip_yolu = os.path.join(train_dizin, klip_adi)

        # .tif karelerini bul
        kareler = sorted(glob.glob(os.path.join(klip_yolu, "*.tif")))

        if not kareler:
            print(f"  [ATLANDI] {klip_adi}: kare bulunamadı.")
            continue

        # İlk kareyi oku → boyutu öğren
        ilk = cv2.imread(kareler[0])
        if ilk is None:
            print(f"  [ATLANDI] {klip_adi}: kare okunamadı.")
            continue

        yukseklik, genislik = ilk.shape[:2]
        cikis_yolu = os.path.join(CIKIS_DIZIN, f"{ped}_{klip_adi}.avi")

        # VideoWriter başlat
        yazici = cv2.VideoWriter(
            cikis_yolu,
            cv2.VideoWriter_fourcc(*"XVID"),
            FPS,
            (genislik, yukseklik),
        )

        for kare_yolu in kareler:
            kare = cv2.imread(kare_yolu)
            if kare is not None:
                yazici.write(kare)

        yazici.release()
        print(f"  ✓ {klip_adi}: {len(kareler)} kare → {cikis_yolu}")

# ── Sonuç Özeti ─────────────────────────────────────────────────────────────

videolar = glob.glob(os.path.join(CIKIS_DIZIN, "*.avi"))
print(f"\n{'='*50}")
print(f"TAMAMLANDI: {len(videolar)} video oluşturuldu.")
print(f"Konum: {CIKIS_DIZIN}/")
print(f"{'='*50}")
print("\nArtık eğitimi başlatabilirsiniz:")
print("  python -m crowdflow.models.train_autoencoder")
