import os
import pandas as pd
import piexif
from PIL import Image
from tqdm import tqdm

def get_image_altitude(image_path):
    try:
        img = Image.open(image_path)
        exif_data = piexif.load(img.info['exif'])
        gps_ifd = exif_data.get('GPS', {})
        
        # GPS Altitude tag
        gps_altitude = gps_ifd.get(piexif.GPSIFD.GPSAltitude)
        
        if gps_altitude:
            # GPSAltitude is in the form of a rational tuple (numerator, denominator)
            altitude = gps_altitude[0] / gps_altitude[1]
            return altitude
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
    return None

def scan_directory_for_images(directory):
    image_data = []
    # Dosya sayısını hesaplayın
    total_files = sum([len(files) for r, d, files in os.walk(directory)])
    # İlerleme çubuğu ile döngüyü başlatın
    with tqdm(total=total_files, desc="Processing images", unit="image") as pbar:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('jpg', 'jpeg', 'png')):
                    file_path = os.path.join(root, file)
                    altitude = get_image_altitude(file_path)
                    if altitude is not None:
                        image_data.append({"filename": file, "altitude": altitude})
                    pbar.update(1)
    return image_data

def save_to_csv(image_data, output_csv):
    df = pd.DataFrame(image_data)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    directory = r"\output_images_irtifa_full"  # Klasör yolunu burada belirtin
    output_csv = "output.csv"  # Çıkış CSV dosya ismini burada belirtin
    image_data = scan_directory_for_images(directory)
    if image_data:
        save_to_csv(image_data, output_csv)
        print(f"Data saved to {output_csv}")
    else:
        print("No images with altitude data found.")
