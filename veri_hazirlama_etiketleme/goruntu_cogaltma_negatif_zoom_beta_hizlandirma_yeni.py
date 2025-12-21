import os
from PIL import Image
import piexif
import numpy as np
import requests









def get_elevation(latitude, longitude):
    url = "https://api.opentopodata.org/v1/eudem25m?locations={},{}".format(latitude, longitude)
    response = requests.get(url)
    data = response.json()
    elevation = data["results"][0]['elevation']
    return elevation


def get_gps_info(image):
    exif_data = piexif.load(image.info['exif'])
    if piexif.GPSIFD.GPSLatitude in exif_data['GPS']:
        latitude = exif_data['GPS'][piexif.GPSIFD.GPSLatitude]
        latitude_ref = exif_data['GPS'][piexif.GPSIFD.GPSLatitudeRef]
        longitude = exif_data['GPS'][piexif.GPSIFD.GPSLongitude]
        longitude_ref = exif_data['GPS'][piexif.GPSIFD.GPSLongitudeRef]

        latitude = latitude[0][0] / latitude[0][1] + latitude[1][0] / (60 * latitude[1][1]) + latitude[2][0] / (3600 * latitude[2][1])
        longitude = longitude[0][0] / longitude[0][1] + longitude[1][0] / (60 * longitude[1][1]) + longitude[2][0] / (3600 * longitude[2][1])

        if latitude_ref == b'S':
            latitude *= -1
        if longitude_ref == b'W':
            longitude *= -1

        return latitude, longitude
    else:
        return None


def crop_around_center(image, width, height):
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


image_directory = "input_images"
output_directory = "output_images/"

train_images = []
irtifa = []

hata = 0

file_list = os.listdir(image_directory)
file_list = [file for file in file_list if file.endswith(('.JPG', '.jpeg', '.jpg'))]

for i, image_file in enumerate(file_list):
    image_path = os.path.join(image_directory, image_file)

    with open(image_path, 'rb') as image_file:
        image = Image.open(image_file)
        original_image = image
        latitude, longitude = get_gps_info(image)
        exif_dict = piexif.load(image.info['exif'])
        gps_altitude = exif_dict['GPS'][6][0]
        t = str(exif_dict['0th'][272])
        
        print(image_file)
        
        
        try:
            rakim = get_elevation(latitude, longitude)
        except Exception as e:
            print("HATAA:", e)
            hata+= 1
            continue
        
        
       
        
        
        
        
        for angle in range(0, 360, 45):            
            
            rotated_image = image.rotate(angle, expand=True)
            rotated_image_orginal=rotated_image
            #new_filename = f"{os.path.splitext(image_file.name)[0]}_{angle}.jpg"
            #new_image_path = os.path.join(output_directory, new_filename)    
            rotated_image = crop_around_center(rotated_image, 1536, 1536)
            rotated_image = rotated_image.resize((512, 512), resample=Image.NEAREST)

            rotated_image = rotated_image.convert('L')
            
            ucus_yuksekligi = gps_altitude - rakim
            
            if t[2:-1] == "L1D-20c":
                exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(ucus_yuksekligi * 0.5915 ), 1)
                train_images.append(np.array(image))

                irtifa.append(ucus_yuksekligi * 0.5915 )
            else:
                exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(ucus_yuksekligi), 1)
                train_images.append(np.array(image))

                irtifa.append(ucus_yuksekligi)
            
            rota=output_directory+str(angle)
            rotated_image.save(rota+"eski_"+str(file_list[i]), exif=piexif.dump(exif_dict))
            
        

            if t[2:-1] == "L1D-20c":
                
    
               
    
                olcekler = np.linspace(0.5915, 0.95, 4)
    
                for olcek in olcekler:
                    zoom_olcek = olcek
                    width, height = rotated_image_orginal.size
                    zoomed_image = rotated_image_orginal.resize((int(width * zoom_olcek), int(height * zoom_olcek)))
    
                    image = crop_around_center(zoomed_image, 1536, 1536)
                    image = image.resize((512, 512), resample=Image.NEAREST)
    
                    image = image.convert('L')
    
                    ucus_yuksekligi = gps_altitude - rakim
    
                    exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(ucus_yuksekligi * 0.5915 / zoom_olcek), 1)
    
                    print(exif_dict['GPS'][piexif.GPSIFD.GPSAltitude], (0.5915 / zoom_olcek))
    
                    image.save(rota+"eski_"+ str(zoom_olcek) +str(file_list[i]), exif=piexif.dump(exif_dict))
    
                    image = image.point(lambda x: x / 255)
    
                    train_images.append(np.array(image))
    
                    irtifa.append(ucus_yuksekligi * 0.5915 / zoom_olcek)
                    image = original_image
            

# train_images=np.array(train_images)
    
            
            

# train_images = np.concatenate(train_images)
# #♦irtifa = np.concatenate(irtifa)

# train_images = train_images.reshape(-1, 512, 512, 1)
# np.save('x_train_negatif_zoom14.npy', train_images)
# np.save('irtifa_negatif_zoom14.npy', irtifa)

