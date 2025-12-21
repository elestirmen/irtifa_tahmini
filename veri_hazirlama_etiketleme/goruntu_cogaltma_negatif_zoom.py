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
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    np_array = np.array(image)
    
    image =cv2.cvtColor(np_array, cv2.COLOR_BGR2GRAY)
    
    #print(image.shape)
    
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]






# Resim dosyalarının bulunduğu dizin
image_directory = "rotated_images"


# Döndürülen resimlerin kaydedileceği dizin
output_directory = "zoomed_images/"


train_images = np.zeros((0, 0))
irtifa = np.zeros((0))

train_images = train_images.astype(np.float32)
irtifa = irtifa.astype(np.float32)

hata=0




file_list=os.listdir(image_directory)

file_list = [file for file in file_list if file.endswith(('.JPG', '.jpeg', '.jpg'))]

i=3000
limit=i+3000
for j in range(len(file_list)):
    
    # Resmi aç
    image_path = os.path.join(image_directory, file_list[i])
    with open(image_path, 'rb') as image_file:
        image = Image.open(image_file)     
        
        
        latitude,longitude= get_gps_info(image)
        
        
        
        
        # Exif bilgilerini oku
        exif_dict = piexif.load(image.info['exif'])
        
        
        gps_altitude = exif_dict['GPS'][6][0]
        
       
        
        t = str(exif_dict['0th'][272])
      

        if t[2:-1]=="L1D-20c":             
            
            
            try:
                rakim = get_elevation(latitude, longitude)
            except:
                print("HATAA")
                hata+=1
                continue
            
            
            #Resmi 30 derece döndür ve kaydet
            for olcek in range(6100, 10000, 300):
                
                zoom_olcek=olcek/10000                                       
                
                width, height = image.size
                
                print(i,":  ",zoom_olcek)
                
               
                
                zoomed_image = image.resize((int(width*zoom_olcek),int(height*zoom_olcek)))
                
                
                ##############################################################################################
                imagecv = crop_around_center(zoomed_image, 1024, 1024)         
                
                imagecv=cv2.resize(imagecv,(512,512),interpolation=cv2.INTER_NEAREST)
                
                #cv2.imshow("goruntu",image)
                
                
                
                img_arr = (np.array(imagecv) / 255.0).astype(np.float32) # Normalizasyon
                
                
                
                #print(img_arr.shape)
                
                #cv2.imshow("test",img_arr)
                
                #input("pause")            
                
                train_images = np.append(train_images, img_arr)
                
                
                
                ucus_yuksekligi = gps_altitude - rakim
                
                
                irtifa=np.append(irtifa, ucus_yuksekligi*0.5915/zoom_olcek).astype(np.float32) 
                
                ###############################################################
                
                
                
                # new_filename = f"{os.path.splitext(filename)[0]}_{zoom_olcek}.jpg"
                # new_image_path = os.path.join(output_directory, new_filename)
                # zoomed_image.save(new_image_path, exif=piexif.dump(exif_dict))
                
            
        """
            # Exif bilgilerini döndürülen resme aktar
            with open(new_image_path, 'rb+') as new_image_file:
                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, new_image_file)
                new_image_file.seek(0)
        """
            
        
    #print(i,": ",filename)
    i+=1
    if limit==i:
        break
    
train_images = train_images.reshape(-1,512,512,1)
np.save('x_train_negatif_zoom.npy', train_images)       
np.save('irtifa_negatif_zoom.npy', irtifa)
    



    
