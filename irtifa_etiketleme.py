import os
import shutil
import rasterio
import requests
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np

# 📌 NASA SRTM Verisi URL (OpenTopography API)
SRTM_API_URL = "https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&outputFormat=GTiff&west={}&south={}&east={}&north={}"

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

def get_gps_coordinates(exif_data):
    """EXIF verilerinden GPS koordinatlarını çeker."""
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        gps_data = {GPSTAGS.get(tag, tag): value for tag, value in gps_info.items()}
        
        if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
            lat = gps_data["GPSLatitude"]
            lon = gps_data["GPSLongitude"]
            lat_ref = gps_data["GPSLatitudeRef"]
            lon_ref = gps_data["GPSLongitudeRef"]

            lat = (lat[0] + lat[1] / 60.0 + lat[2] / 3600.0) * (-1 if lat_ref == 'S' else 1)
            lon = (lon[0] + lon[1] / 60.0 + lon[2] / 3600.0) * (-1 if lon_ref == 'W' else 1)

            return lat, lon
    return None

def get_gps_altitude(exif_data):
    """EXIF verilerinden GPS irtifasını çeker."""
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        gps_data = {GPSTAGS.get(tag, tag): value for tag, value in gps_info.items()}
        if "GPSAltitude" in gps_data and "GPSAltitudeRef" in gps_data:
            altitude = gps_data["GPSAltitude"]
            ref = gps_data["GPSAltitudeRef"]
            altitude = altitude[0] / altitude[1] if isinstance(altitude, tuple) else altitude
            return -altitude if ref == 1 else altitude
    return None

def get_terrain_elevation(lat, lon):
    """SRTM veya OpenTopography API kullanarak arazi yüksekliğini alır."""
    try:
        response = requests.get(SRTM_API_URL.format(lon, lat, lon, lat))
        if response.status_code == 200:
            with open("dem.tif", "wb") as f:
                f.write(response.content)

            with rasterio.open("dem.tif") as dataset:
                row, col = dataset.index(lon, lat)
                elevation = dataset.read(1)[row, col]
                return elevation
    except Exception as e:
        print(f"Arazi yüksekliği alınamadı: {e}")
    return None

def filter_and_copy_images(source_folder, target_folder):
    """Görüntülerin uçuş irtifasını hesaplar ve belirlenen aralıktaki görüntüleri kopyalar."""
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
                gps_coords = get_gps_coordinates(exif_data)
                altitude = get_gps_altitude(exif_data)

                if gps_coords and altitude:
                    terrain_elevation = get_terrain_elevation(*gps_coords)
                    if terrain_elevation is not None:
                        flight_height = altitude - terrain_elevation  # 📌 Uçuş irtifası

                        # ✅ Eğer uçuş irtifası 1500-1800 metre arasındaysa dosyayı kopyala
                        if 480 <= flight_height <= 650:
                            target_path = os.path.join(target_folder, file_name)
                            shutil.copy(file_path, target_path)
                            copied_count += 1
                            print(f"✅ Kopyalandı: {file_name} - Uçuş İrtifası: {flight_height:.2f} m")

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
