import cv2


import os
from PIL import Image
import piexif

import fonksiyonlar as fk  
import numpy as np



import requests

def get_elevation(latitude, longitude):
    url = "https://api.opentopodata.org/v1/eudem25m?locations={},{}".format(latitude, longitude)
    response = requests.get(url)
    data = response.json()
    
    #print(data)
    """
    elevation = None
    if data["results"][0]['elevation'] is not None:
        elevation = data["results"][0]['elevation']
    """
    #print(data["results"][0])
    elevation = data["results"][0]['elevation']
    return elevation



def get_gps_info(image):
    exif_data = piexif.load(image.info['exif'])
    if piexif.GPSIFD.GPSLatitude in exif_data['GPS']:
        latitude = exif_data['GPS'][piexif.GPSIFD.GPSLatitude]
        latitude_ref = exif_data['GPS'][piexif.GPSIFD.GPSLatitudeRef]
        longitude = exif_data['GPS'][piexif.GPSIFD.GPSLongitude]
        longitude_ref = exif_data['GPS'][piexif.GPSIFD.GPSLongitudeRef]

        # Convert from rational numbers to decimal degrees
        latitude = latitude[0][0] / latitude[0][1] + latitude[1][0] / (60 * latitude[1][1]) + latitude[2][0] / (3600 * latitude[2][1])
        longitude = longitude[0][0] / longitude[0][1] + longitude[1][0] / (60 * longitude[1][1]) + longitude[2][0] / (3600 * longitude[2][1])

        # Change the sign based on reference directions
        if latitude_ref == b'S':
            latitude *= -1
        if longitude_ref == b'W':
            longitude *= -1

        return latitude, longitude
    else:
        return None
    
    
    
def crop_around_center(image, width, height):
    """
    Given a PIL image, crops it to the given width and height,
    around it's centre point
    """

    # Convert the image to grayscale
    image = image.convert('L')
    
    image_size = (image.width, image.height)
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image.crop((x1, y1, x2, y2))







# Resim dosyalarının bulunduğu dizin
image_directory = "input_images"


# Döndürülen resimlerin kaydedileceği dizin
output_directory = "zoomed_images/"


train_images = np.zeros((0, 0))
irtifa = np.zeros((0))

train_images = train_images.astype(np.float32)
irtifa = irtifa.astype(np.float32)

hata=0




file_list = os.listdir(image_directory)
file_list = [file for file in file_list if file.endswith(('.JPG', '.jpeg', '.jpg'))]

train_images = []
irtifa = []

i = 0
limit = i + 1000

for j in range(len(file_list)):
    image_path = os.path.join(image_directory, file_list[i])
    
    with open(image_path, 'rb') as image_file:
        image = Image.open(image_file)
        original_image=image
        latitude, longitude = get_gps_info(image)
        exif_dict = piexif.load(image.info['exif'])
        gps_altitude = exif_dict['GPS'][6][0]
        t = str(exif_dict['0th'][272])
        if t[2:-1] == "L1D-20c":
            try:
                rakim = get_elevation(latitude, longitude)
            except:
                print("HATAA")
                hata+=1
                continue
            
            img_arr_list = []
            irtifa_list = []
            
            
            olcekler = np.linspace(0.5915, 0.92, 7)

            
            for olcek in olcekler:
                zoom_olcek = olcek
                width, height = image.size
                zoomed_image = image.resize((int(width*zoom_olcek), int(height*zoom_olcek)))
                
                image = crop_around_center(zoomed_image, 1024,1024)
                image = image.resize((512, 512), resample=Image.NEAREST)  
                
                image = image.convert('L')  # Grayscale yapmak isterseniz bu satırı ekleyin
                
                ucus_yuksekligi = gps_altitude - rakim
                
                exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(ucus_yuksekligi * 0.5915 / zoom_olcek), 1)
                
                print( exif_dict['GPS'][piexif.GPSIFD.GPSAltitude],(0.5915 / zoom_olcek))

                image.save("zoomed_images/_zoomed_"+str(zoom_olcek)+".jpg", exif=piexif.dump(exif_dict))

                image = image.point(lambda x: x / 255)
                
                img_arr_list.append(np.array(image))
                
                
                irtifa_list.append(ucus_yuksekligi * 0.5915 / zoom_olcek)
                image = original_image
                
            
            train_images.append(np.array(img_arr_list))
            
            irtifa.append(np.array(irtifa_list))
            
    i += 1
    print(i)
    if limit == i:
        break
"""
train_images = np.concatenate(train_images)
irtifa = np.concatenate(irtifa)

    
train_images = train_images.reshape(-1,512,512,1)
np.save('x_train_negatif_zoom14.npy', train_images)       
np.save('irtifa_negatif_zoom14.npy', irtifa)
    


#cv2.imshow("test",train_images[0])
    
"""