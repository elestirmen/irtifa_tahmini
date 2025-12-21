import cv2
import os


import fonksiyonlar as fk  


import os
from PIL import Image
import piexif

# Resim dosyalarının bulunduğu dizin
image_directory = "veri_hazirlama_etiketleme/parcalar"




# Döndürülen resimlerin kaydedileceği dizin
output_directory = "rotated_images/"

for filename in os.listdir(image_directory):
    if filename.endswith(".JPG") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Resmi aç
        image_path = os.path.join(image_directory, filename)
        with open(image_path, 'rb') as image_file:
            image = Image.open(image_file)

            # Exif bilgilerini oku
            exif_dict = piexif.load(image.info['exif'])

            # Resmi 30 derece döndür ve kaydet
            for angle in range(0, 360, 30):
                rotated_image = image.rotate(angle, expand=True)
                new_filename = f"{os.path.splitext(filename)[0]}_{angle}.jpg"
                new_image_path = os.path.join(output_directory, new_filename)
                rotated_image.save(new_image_path, exif=piexif.dump(exif_dict))
                
                """
                # Exif bilgilerini döndürülen resme aktar
                with open(new_image_path, 'rb+') as new_image_file:
                    exif_bytes = piexif.dump(exif_dict)
                    piexif.insert(exif_bytes, new_image_file)
                    new_image_file.seek(0)
                """