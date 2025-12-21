import os
import requests
import numpy as np
from PIL import Image
import piexif
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import rasterio
import numpy as np
from pyproj import Transformer
import cProfile


class ElevationModel:
    def __init__(self, dem_file):
        self.dem = rasterio.open(dem_file)
        self.transform = self.dem.transform
        self.elevation_data = self.dem.read(1)
        self.height, self.width = self.elevation_data.shape

    def get_elevation(self, lon, lat):
        # Enlem ve boylamı piksel koordinatlara dönüştür
        py, px = rasterio.transform.rowcol(self.transform, lon, lat)

        # Yükseklik bilgisini oku, eğer koordinatlar harita dışındaysa None döndür
        if 0 <= px < self.width and 0 <= py < self.height:
            elevation = self.elevation_data[py, px]

            # NaN değerleri kontrol et (eğer varsa)
            if np.isnan(elevation):
                return None
            else:
                return elevation
        else:
            return None
        
        
def wgs84_to_utm36(lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    return transformer.transform(lon, lat)


def wgs84_to_web_mercator(lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transformer.transform(lon, lat)



def _3395_to_web_mercator(lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)
    return transformer.transform(lon, lat)


def _32636_to_web_mercator(lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    return transformer.transform(lon, lat)

def check_dem_info(dem_file):
    with rasterio.open(dem_file) as dem:
        print("Bounds:", dem.bounds)
        print("CRS:", dem.crs)



"""            
            
def get_elevation_google(lat, long):
    query = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{long}&key=AIzaSyAAskunnv0rmhCF4t2XfaZlCIT002frX-U"

    response = requests.get(query)
    data = response.json()

    if data['status'] == 'OK':
        return data['results'][0]['elevation']
    else:
        return None


# ... (get_elevation ve get_gps_info fonksiyonları burada)
def get_elevation(latitude, longitude, retries=20):
    for _ in range(retries):
        try:
            url = "https://api.opentopodata.org/v1/eudem25m?locations={},{}".format(latitude, longitude)
            response = requests.get(url)
            
            data = response.json() 
            elevation = data["results"][0]['elevation']            
            return elevation
        
        except Exception as e:
            if _ < retries - 1:
                time.sleep(1)
                continue
            else:
                print("HATAA:", e)
                return None
            
            
def get_elevation_(latitude, longitude):
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {
        "locations": "{},{}".format(latitude, longitude)
    }

    response = requests.get(url, params=params, verify=False)  # SSL sertifika doğrulamasını devre dışı bırak
    data = response.json()

    elevation = data["results"][0]["elevation"]
    return elevation
 
"""

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
    image = image.convert('RGB')   #image.convert('L')
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




def process_image(args):
    image_path, output_directory, file_name, angle, zoom_olcek, t = args

    with open(image_path, 'rb') as image_file:
        image = Image.open(image_file)
        latitude, longitude = get_gps_info(image)
        
        mercator_lon, mercator_lat = wgs84_to_utm36(longitude, latitude)
        #mercator_lon, mercator_lat = _32636_to_web_mercator(longitude, latitude)


        
        exif_dict = piexif.load(image.info['exif'])
        gps_altitude = exif_dict['GPS'][6][0]
        t = str(exif_dict['0th'][272])
        
        focal_length = exif_dict['Exif'].get(piexif.ExifIFD.FocalLength, None)
        focal_length = focal_length[0] / focal_length[1]


        rakim = elevation_model.get_elevation(mercator_lon, mercator_lat)
        
        
        if rakim==None:
            mercator_lon, mercator_lat = _3395_to_web_mercator(longitude, latitude)
            #mercator_lon, mercator_lat = _32636_to_web_mercator(longitude, latitude)
            rakim=elevation_model_2.get_elevation(mercator_lon,mercator_lat)
            
        
        
        if rakim is None:
            print("Rakım değeri alınamadı, işlem gerçekleştirilmedi.")
            return None
        
        
        rotated_image = image.rotate(angle, expand=True)
        rotated_image_orginal = rotated_image
        rotated_image = crop_around_center(rotated_image, 1024, 1024)
        rotated_image = rotated_image.resize((512, 512), resample=Image.NEAREST)
        rotated_image = rotated_image.convert('RGB')      #rotated_image.convert('L')

        ucus_yuksekligi = gps_altitude - rakim

        if t[2:-1] == "L1D-20c":
            exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(ucus_yuksekligi * 0.669), 1)
            irtifa = ucus_yuksekligi * 0.669
        else:
            ucus_yuksekligi = ucus_yuksekligi * (4.386 / focal_length)
            exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(ucus_yuksekligi), 1)
            irtifa = ucus_yuksekligi

        rota = output_directory + str(angle)
        rotated_image.save(rota + "_" + str(file_name)+".jpg", exif=piexif.dump(exif_dict))

        if zoom_olcek is not None and t[2:-1] == "L1D-20c":
            
            width, height = rotated_image_orginal.size
            zoomed_image = rotated_image_orginal.resize((int(width * zoom_olcek), int(height * zoom_olcek)))
            image = crop_around_center(zoomed_image, 1024, 1024)
            image = image.resize((512, 512), resample=Image.NEAREST)
            image = image.convert('RGB')    #image.convert('L')
    
            ucus_yuksekligi = gps_altitude - rakim
            exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(ucus_yuksekligi * 0.669 / zoom_olcek), 1)
            image.save(rota + "_" + str(round(zoom_olcek, ndigits=3)) +"_"+ str(file_name), exif=piexif.dump(exif_dict))
            irtifa = ucus_yuksekligi * 0.669 / zoom_olcek            
        
        elif zoom_olcek is not None and t[2:-1] == "FC2204":
            ucus_yuksekligi = ucus_yuksekligi * (4.386 / focal_length)
            width, height = rotated_image_orginal.size
            zoomed_image = rotated_image_orginal.resize((int(width * zoom_olcek), int(height * zoom_olcek)))
            image = crop_around_center(zoomed_image, 1024, 1024)
            image = image.resize((512, 512), resample=Image.NEAREST)
            image = image.convert('RGB')    #image.convert('L')
    
            ucus_yuksekligi = gps_altitude - rakim
            exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(ucus_yuksekligi/ zoom_olcek), 1)
            image.save(rota + "_" + str(round(zoom_olcek, ndigits=3)) +"_"+ str(file_name), exif=piexif.dump(exif_dict))
            irtifa = ucus_yuksekligi/ zoom_olcek     

        return irtifa

def main():
    #image_directory = "D:/d_surucusu/m2pro_irtifa_icin/"
    image_directory = "input_images"
    output_directory = "output_images/"

    irtifa = []

    file_list = os.listdir(image_directory)
    file_list = [file for file in file_list if file.endswith(('.JPG', '.jpeg', '.jpg'))]
    dosya_adedi=len(file_list)
    
    for i, image_file in enumerate(file_list):
        image_path = os.path.join(image_directory, image_file)
        file_name = os.path.split(image_path)[1]  # Sadece dosya adını alın


        with open(image_path, 'rb') as image_file:
            image = Image.open(image_file)
            latitude, longitude = get_gps_info(image)
            exif_dict = piexif.load(image.info['exif'])
            t = str(exif_dict['0th'][272])

            print(image_file,str(i),"/",str(dosya_adedi))

            process_args = []

            for angle in range(0, 360, 30):
                olcekler_m2z = np.linspace(0.6, 0.95, 5)
                for zoom_olcek in olcekler_m2z:
                    process_args.append((image_path, output_directory, file_name, angle, zoom_olcek, t))
                    

            if t[2:-1] == "L1D-20c":
                for angle in range(0, 360, 30):
                    olcekler_m2p = np.linspace(0.4, 0.95, 10)
                    

                    for zoom_olcek in olcekler_m2p:
                        process_args.append((image_path, output_directory, file_name, angle, zoom_olcek, t))


            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_image, args) for args in process_args]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        irtifa.append(result)
                    except Exception as e:
                        print("İşlem sırasında hata oluştu:", e)                 


if __name__ == "__main__":
    dem_file = 'ana_harita_urgup_30_cm_utm_elevation.tif'
    elevation_model = ElevationModel(dem_file)
    dem_file2 = 'karlik_30_cm_bingmap_utm_elevation.tif'
    elevation_model_2 = ElevationModel(dem_file2)

    # check_dem_info(dem_file)
    # check_dem_info(dem_file2)

    main()
