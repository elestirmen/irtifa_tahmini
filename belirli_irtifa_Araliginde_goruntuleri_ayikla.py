import os
import shutil
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif_data(image_path):
    """Verilen görüntü dosyasının EXIF verilerini alır."""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None
        return {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
    except Exception as e:
        print(f"Hata: {image_path} - {e}")
        return None

def get_gps_altitude(exif_data):
    """EXIF verilerinden GPS irtifasını çeker ve float formatına çevirir."""
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        gps_data = {GPSTAGS.get(tag, tag): value for tag, value in gps_info.items()}
        if "GPSAltitude" in gps_data and "GPSAltitudeRef" in gps_data:
            altitude = gps_data["GPSAltitude"]
            ref = gps_data["GPSAltitudeRef"]

            # Eğer altitude bir tuple (kesir) ise, float'a çevir
            if isinstance(altitude, tuple):
                altitude = altitude[0] / altitude[1]  # Örneğin: (4000, 1000) -> 4.0
            
            return -float(altitude) if ref == 1 else float(altitude)
    return None

def filter_and_copy_images(source_folder, target_folder):
    """Kaynak klasör ve alt klasörleri tarayarak 1500-1800m irtifadaki görüntüleri kopyalar."""
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"⚠️ Hata: '{source_folder}' klasörü bulunamadı!")

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    copied_count = 0

    for root, _, files in os.walk(source_folder):
        for file_name in files:
            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                continue  # Sadece görüntü dosyalarını işler

            file_path = os.path.join(root, file_name)
            exif_data = get_exif_data(file_path)

            if exif_data:
                altitude = get_gps_altitude(exif_data)
                if altitude is not None and 1650 <= altitude <= 1800:
                    target_path = os.path.join(target_folder, file_name)
                    shutil.copy(file_path, target_path)
                    copied_count += 1
                    print(f"✅ Kopyalandı: {file_name} - {altitude:.2f} metre")

    if copied_count == 0:
        print("⚠️ Hiçbir uygun görüntü bulunamadı.")
    else:
        print(f"✅ {copied_count} görüntü başarıyla kopyalandı!")

# 📂 Kaynak ve hedef klasörleri belirleyin
source_folder = r"G:\tez veri\tez__ayna\python_calismalar\guzergahlar\yeni"  # 🔄 Değiştirin: Taranacak klasör
target_folder = "hedef_klasor"    # 🔄 Değiştirin: Seçilen görüntülerin kopyalanacağı klasör

# 🚀 İşlemi başlat
try:
    filter_and_copy_images(source_folder, target_folder)
except FileNotFoundError as e:
    print(e)
