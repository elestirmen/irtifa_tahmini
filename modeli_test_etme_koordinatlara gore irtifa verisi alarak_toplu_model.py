import requests
from PIL import Image, ExifTags
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
#import piexif
import exifread
from requests.exceptions import JSONDecodeError
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)  # Uyarıyı devre dışı bırak
import rasterio
from pyproj import Transformer


#import fonksiyonlar as fk 



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

def wgs84_to_web_mercator(lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transformer.transform(lon, lat)

def web_mercator_to_3395(lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)
    return transformer.transform(lon, lat)

def web_mercator_to_utm_zone_36n(lon, lat):
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    return transformer.transform(lon, lat)


def check_dem_info(dem_file):
    with rasterio.open(dem_file) as dem:
        print("Bounds:", dem.bounds)
        print("CRS:", dem.crs)



def get_elevation_from_coords(raster_file_path, lon, lat):
    """
    Verilen koordinatlardaki yükseklik değerini döndürür.

    Parametreler:
    - raster_file_path (str): Raster dosyasının yolu.
    - lon (float): Boylam (derece cinsinden, WGS84).
    - lat (float): Enlem (derece cinsinden, WGS84).

    Çıktı:
    - elevation (float): Verilen koordinatlardaki yükseklik değeri.
    """

    # Dönüştürücüyü oluştur
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)

    # Koordinatları dönüştür
    x, y = transformer.transform(lon, lat)

    # Raster dosyayı aç
    with rasterio.open(raster_file_path) as dataset:

        # Dönüştürülmüş koordinatların piksel konumlarını al
        row, col = dataset.index(x, y)

        # Elevation değerini oku
        elevation = dataset.read(1)[row, col]

    return elevation




"""

def get_elevation(lat, long):
    query = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{long}&key=AIzaSyAAskunnv0rmhCF4t2XfaZlCIT002frX-U"

    response = requests.get(query)
    data = response.json()

    if data['status'] == 'OK':
        return data['results'][0]['elevation']
    else:
        return None




def get_elevation__(latitude, longitude):
    url = "https://api.opentopodata.org/v1/eudem25m?locations={},{}".format(latitude, longitude)
    response = requests.get(url)
    data = response.json()
   

    elevation = data["results"][0]['elevation']
    return elevation


def get_elevation_(latitude, longitude):
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {
        "locations": "{},{}".format(latitude, longitude)
    }

    response = requests.get(url, params=params, verify=False)  # SSL sertifika doğrulamasını devre dışı bırak
    data = response.json()

    elevation = data["results"][0]["elevation"]
    return elevation



def get_elevation_usgs(latitude, longitude):
    url = "https://nationalmap.gov/epqs/pqs.php"
    params = {
        "x": longitude,
        "y": latitude,
        "units": "Meters",
        "output": "json"
    }

    response = requests.get(url, params=params)

    try:
        data = response.json()
    except JSONDecodeError as e:
        raise ValueError("API'dan alınan yanıt geçerli bir JSON değil: {}".format(response.text))

    elevation = data["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"]
    return elevation

"""

def crop_around_center(image, width, height):
    image = image.convert('RGB')            #image.convert('L')
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




def load_image2(path):
    
    image = Image.open(path)
    image = crop_around_center(image, 1024, 1024)
    image = image.resize((512, 512), resample=Image.Resampling.NEAREST)
    image = image.convert('RGB')            #image.convert('L')
    img_arr = np.array(image) / 255.0
    
    """
    image = cv2.imread(path,0)     
    
    image = crop_around_center(image,1024, 1024)         
    
    image=cv2.resize(image,(512, 512), interpolation=cv2.INTER_NEAREST) 
    
   
    img_arr = np.array(image) / 255.0 # Normalizasyon
    """
    return img_arr





def dms_to_decimal(value):
    d, m, s = value.values
    d = d.num / d.den
    m = m.num / m.den
    s = s.num / s.den
    return d + m / 60 + s / 3600

def get_gps_data(image_path):
    with open(image_path, 'rb') as img_file:
        exif_data = exifread.process_file(img_file, details=False)
        
    
        
    if exif_data is None:
        print("Exif verisi bulunamadı.")
        return None
    
    camera_model = str(exif_data['Image Model'])
 
    gps_info = {key: exif_data[key] for key in exif_data.keys() if key.startswith('GPS')}

    
    if len(gps_info) == 0:
        print("GPS verisi bulunamadı.")
        return None

    gps_latitude = gps_info.get('GPS GPSLatitude')
    gps_latitude_ref = gps_info.get('GPS GPSLatitudeRef')
    gps_longitude = gps_info.get('GPS GPSLongitude')
    gps_longitude_ref = gps_info.get('GPS GPSLongitudeRef')
    gps_altitude = gps_info.get('GPS GPSAltitude')

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        latitude = dms_to_decimal(gps_latitude)
        longitude = dms_to_decimal(gps_longitude)

        if gps_latitude_ref.values == 'S':
            latitude = -latitude
        if gps_longitude_ref.values == 'W':
            longitude = -longitude
    else:
        print("Koordinat bilgisi bulunamadı.")
        latitude, longitude = None, None

    if gps_altitude:
        altitude = gps_altitude.values[0].num / gps_altitude.values[0].den
                      
        
    else:
        print("GPS yükseklik bilgisi bulunamadı.")
        altitude = None

    return latitude, longitude, altitude, camera_model



dem_file = 'C:/d_surucusu/urgup_gmap_30_elevation_small.tif'
dem_file = 'ana_harita_urgup_30_cm_utm_elevation.tif'


elevation_model = ElevationModel(dem_file)

dem_file2 = 'karlik_30_cm_bingmap_utm_elevation.tif'
elevation_model_2 = ElevationModel(dem_file2)


#%%


#from egitim_sureci_dosyadan_okuma_heterojen_dagilimli import tf_weighted_mean_squared_error,tf_sqrt_weighted_mean_absolute_error

#model = load_model("epoch00012_relu.h5", custom_objects={'tf_weighted_mean_squared_error': tf_weighted_mean_squared_error})

#model = load_model("epoch00017_relu.h5", custom_objects={'tf_sqrt_weighted_mean_absolute_error': tf_sqrt_weighted_mean_absolute_error})


model_yol="modeller"

modeller=os.listdir(model_yol)

ortalama_farklar=[]

for model in modeller:
    print("\nmodel: "+model+"\n\n")
    
    model = tf.keras.models.load_model(model_yol+"/"+model, compile=False)

    #model=load_model("devam_edecek.h5")
    
    #adam = tf.keras.optimizers.Adam(lr=0.0001)  
    
    model.compile(optimizer='adam', loss='mean_absolute_error') 
    
    
    #model = load_model("_epoch_00006_relu_.h5")
    
    test=[]
    
    
    
    
    test_klasoru="test_goruntuleri"
    # test_klasoru="test_goruntuleri_m2p"
    # test_klasoru="test_goruntuleri"
    
    t=os.listdir(test_klasoru)
    
    
    
    
    
    
    gps_alt=[]
    rakim=[]
    
    gercek_irtifa=[]
    
    sonuc=[]
    elevation=0
    prediction=[]
    
    
    toplam=0
    for i,img in enumerate(t):
        bosluk=""
        gps_alt.append(get_gps_data(test_klasoru+"/"+t[i]))
        
        
        if elevation is not None and gps_alt[i] is not None:        
           
            mercator_lon, mercator_lat = web_mercator_to_utm_zone_36n(gps_alt[i][1], gps_alt[i][0])
            
            
            elevation=elevation_model.get_elevation(mercator_lon,mercator_lat)
            if elevation==None:
                mercator_lon, mercator_lat = web_mercator_to_utm_zone_36n(gps_alt[i][1], gps_alt[i][0])
                elevation=elevation_model_2.get_elevation(mercator_lon,mercator_lat)
                
            rakim.append(elevation)
            arazi_irtifa = int(gps_alt[i][2]-rakim[i])
            
            
            
            if str(gps_alt[i][3])=='L1D-20c':
                gercek_irtifa.append(int(arazi_irtifa*0.669))   
                #gercek_irtifa.append(int(arazi_irtifa))
                
            else:
                gercek_irtifa.append(arazi_irtifa)
        
                
        else:
            #print(f"{i}. görüntü için yükseklik bilgisi alınamadı.")
            gercek_irtifa.append(-9999)
        
        
        
        test.append(load_image2(test_klasoru+"/"+img))
        
        
             #sonuc.append(test[i].reshape(-1,512,512,1))
    
        try:
            sonuc.append(test[i].reshape(-1,512,512,3)) 
            prediction.append(model.predict(sonuc[i], verbose=0))  
        except:
            # Görüntüyü tek kanallı (siyah beyaz) hale getirin
            gray_image = np.mean(test[i], axis=2, keepdims=True)
            reshaped_image=gray_image.reshape(-1,512,512,1) 
            prediction.append(model.predict(reshaped_image, verbose=0))  
            
        print(prediction[i])
        print(gercek_irtifa[i])
        fark=int(prediction[i])-int(gercek_irtifa[i])
        
        toplam += np.absolute(fark)
        for b in range(16-len(t[i])):
            bosluk+=" "
        
        print(t[i],bosluk,"\t tahmin: ",int(prediction[i]),"   olculendeger: ","{:03d}".format(gercek_irtifa[i]),"||| fark: ",fark)
    
    ortalama_fark=(toplam/len(t))
    print("\nortalama fark: ",ortalama_fark,"\n\n")
    
    ortalama_farklar.append(ortalama_fark)
        
with open('sonuclar.txt', 'w') as dosya:
    for eleman1, eleman2 in zip(modeller, ortalama_farklar):
        dosya.write(f'{eleman1.ljust(40)} {eleman2:.3f}\n')
    
    
