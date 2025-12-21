import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from osgeo import gdal
import exiftool
from tensorflow.keras.models import load_model
import pickle
import multiprocessing
import warnings
import math
import time
warnings.filterwarnings("ignore")


# import rasterio as rio
# from rasterio.warp import transform 
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import load_img


import fonksiyonlar as fk    #fonksiyonların olduğu dosya çağrılır



#%%





#%%

import requests

def get_elevation(latitude, longitude):
    url = "https://api.opentopodata.org/v1/eudem25m?locations={},{}".format(latitude, longitude)
    response = requests.get(url)
    data = response.json()
   
    """
    elevation = None
    if data["results"][0]['elevation'] is not None:
        elevation = data["results"][0]['elevation']
    """
    print(data["results"][0])
    elevation = data["results"][0]['elevation']
    return elevation




# Resimleri yükleme ve boyutlandırma
def load_image(path):
    img = cv2.imread(path,0)
    img = cv2.resize(img,(512, 512)) # Resimleri boyutlandırma
    img_arr = np.array(img) / 255.0 # Normalizasyon
    return img_arr











dirname = os.path.dirname(os.path.abspath(__file__))




# directory = "parcalar" # görüntülerin bulunduğu dizin

# train_images = np.array([load_image("parcalar/"+path) for path in os.listdir(directory)])


# train_images = train_images.reshape(-1,512,512,1)

# np.save('x_train.npy', train_images)


train_images = np.zeros((0, 0))


if __name__ == '__main__':

  

    
    pool = multiprocessing.Pool()
    pool = multiprocessing.Pool(processes=4)
    
    
    
    #haritalar klasöründeki ilk görüntüde DEM verileri vardır. ikinci görüntü ise normal rgb görüntüdür.
    harita_yol=dirname+'/haritalar/'
    harita_yol_list=os.listdir(harita_yol)
    model_yol=dirname+'/model/'
    model_list=os.listdir(model_yol)
    #ana_harita_elevation = "urgup_genis_elevations.tif"
    #ana_harita_elevation="urgup_gmap_30_cm_elevations_560.tif"
    ana_harita_elevation="ana_harita_karlik_30_cm_bingmap_elevations_576.tif"
    
    
    
    
    
    # haritadaki piksellerin gps koordinatları bulunur ve koordinatlar olarak ayrı bri dosya olarak diske kaydedilir. bir kez çalıştırılması yeterlidir
    ###############################################################################
    #%%
    """
    import rasterio
    from affine import Affine
    from pyproj import Proj, transform
    
    #fname = 'urgup_gmap_georef.tif'
    fname = ana_harita_elevation    # harita_yol+harita_yol_list[0]
    
    # Read raster
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read()  # pixel values
    
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))
    
    def koordinat_bul(row,col):
        # Get affine transform for pixel centres
        T1 = T0 * Affine.translation(0.5, 0.5)
        # Function to convert pixel row/column index (from 0) to easting/northing at centre
        rc2en = lambda r, c: (c, r) * T1
        
        # All eastings and northings (there is probably a faster way to do this)
        eastings, northings = np.vectorize(rc2en, otypes=[float, float])(rows[row], cols[col])
        
        
        # Project all longitudes, latitudes
        p2 = Proj(proj='latlong',datum='WGS84')
        longs, lats = transform(p1, p2, eastings, northings)
       
        
        return (longs,lats)
    
    
    
    

    
    
    
    #%%

    
    #DEM verileri aktarılır
    
    filename = ana_harita_elevation
            
    dataset = gdal.Open(filename)
    
    gt = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)  #5. bant elevation bandı
    
    DEM_array = band.ReadAsArray()
    """
    ###############################################################################
    
    #%%
    
    
    
    hata=0
    
    
    konum=(0,0)
    konum_once=(0,0)
    kare=()  
    
    irtifa=[]
    for k in range(len(harita_yol_list)):
        
         
        dogru_tahmin=0
        yanlis_tahmin=0
        ana_harita=ana_harita_elevation
          
        #t_img = DEM_array  #haritalar klasöründeki ikinci görüntüyü okur
        #print(t_img.shape)
          
        #kenarx=int(t_img.shape[0]/512)
        
        #parcalar klasöründeki anlık görüntüleri getirir
        anlik_yol=dirname+'/rotated_images/'
        
        anlik_yol_list=os.listdir(anlik_yol)
        
        #anlik_goruntu=anlik_yol+anlik_yol_list[0]
        
        anlik_yol_list = sorted( anlik_yol_list,
                                key = lambda x: os.path.getmtime(os.path.join(anlik_yol, x))  # tarihe göre klasördeki dosyaları sıralar
                                )
        
        
        
      

        uzaklik=0
        
        i=0
        i=21000
        birak=i+3000
        for j in range(len(anlik_yol_list)): 
            
            #konum_once=konum
            
            
            
            
          
            anlik_goruntu = "rotated_images/"+anlik_yol_list[i]  #klasördeki ilk görüntüyü getir
            
            
            
            print("sira: "+str(i)+" "+anlik_yol_list[i] ) 
        
            #exif bilgileri okunur
            #####################################################
            with exiftool.ExifToolHelper() as et:
                metadata = et.get_metadata(anlik_goruntu)
            
            
                   
             
            
            altitude=metadata[0]["EXIF:GPSAltitude"]
            
            gps_latitude = metadata[0]["EXIF:GPSLatitude"]
            
            gps_longitude =metadata[0]["EXIF:GPSLongitude"]
            ######################################################
            
           
            
            
            
            
            
            
            
            
            
            try:
                if  "_0.jpg" in  anlik_goruntu:           #i%12==0:
                    rakim=get_elevation(gps_latitude, gps_longitude)
            except:
                print("HATAA")
                hata+=1
                i+=12
                continue
                
            """
            try:
                rakim=DEM_array[knm[1],knm[0]]
               
            except:
                print("dışarıda")
                continue
            """
            
            if metadata[0]["EXIF:Model"]=="L1D-20c":   
                #spatial çözünürlük elde etme
                #######################################################################
                camera_sensor_genislik=15.9 #mavic2pro için 13.2  milimetre sensör genişliği
                camera_focal_lenght=metadata[0]["EXIF:FocalLength"] #mavic2pro için 10.26 milimetre
                ucus_yuksekligi=altitude - rakim  #metre olarak yerden x"x""uçuş yüksekliği  35 dem dosyasındaki hatadan dolayı
                goruntu_piksel_genisligi = 5472 #pipksel olarak resmin genişliği
                goruntu_piksel_yuksekligi = 3648 #pipksel olarak resmin genişliği
                mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
                goruntunun_gercek_uzunlugu=(mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak
                
                #görüntünün hangi oranda küçültüleceğini belirler mekansal çözünürlüğe göre
                #olcek_scale_test=(mekansal_cozunurluk/(29.9 *(560/544)))
                
                #olcek_scale_test=(mekansal_cozunurluk/29.85 ) * (560/544)
                #olcek_scale_test=(mekansal_cozunurluk/29.85 ) * (576/544)
                olcek_scale_test=0.5916
                           
           
            
            elif metadata[0]["EXIF:Model"]=="FC2204":   
                #spatial çözünürlük elde etme
                #######################################################################
                camera_sensor_genislik =  8.407036405 #mavic2zoom için 6.17  milimetre sensör genişliği
                camera_focal_lenght= metadata[0]["EXIF:FocalLength"]  #mavic2zoom için 4 milimetre
                ucus_yuksekligi=altitude - rakim   #metre olarak yerden x"x""uçuş yüksekliği  33 dem dosyasındaki hatadan dolayı
                #←ucus_yuksekligi=726
                goruntu_piksel_genisligi = 4000 #pipksel olarak resmin genişliği
                goruntu_piksel_yuksekligi = 3000 #pipksel olarak resmin genişliği
                mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
                goruntunun_gercek_uzunlugu=(mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak
               
                #görüntünün hangi oranda küçültüleceğini belirler mekansal çözünürlüğe göre
                #olcek_scale_test=(mekansal_cozunurluk/29.85)  * (560/544)
                olcek_scale_test=(mekansal_cozunurluk/29.85 ) * (576/544)
                olcek_scale_test=1
                #######################################################################
                
                
            
                
            #continue
                
            
         
            #print(olcek_scale)
            print(olcek_scale_test)
       
            
           
            
            
            
          
            
            #################################################################################################
            
            # Reading the image
            image = cv2.imread(anlik_goruntu,0)

            height, width = image.shape[:2]
        
            
                
            
            ##########################################################
            """
            # Kare boyutunu hesaplayın
            if height > width:
                square_size = width
            else:
                square_size = height
            
            # Kareyi kırpın
            x = int((width - square_size) / 2)
            y = int((height - square_size) / 2)
            cropped_img = image[y:y+square_size, x:x+square_size]
            """
            
            
            
            ###########################################################
            
            
            image=cv2.resize(image, (int(width*olcek_scale_test),int(height*olcek_scale_test)),interpolation=cv2.INTER_NEAREST ) 
            
            image = fk.crop_around_center(image, 1024, 1024)         
            
            image=cv2.resize(image,(512, 512))
            
            #cv2.imshow("goruntu",image)
            
            
            
            img_arr = np.array(image) / 255.0 # Normalizasyon
            
            #print(img_arr.shape)
            
            #cv2.imshow("test",img_arr)
            
            #input("pause")            
            
            train_images = np.append(train_images, img_arr)
            
            irtifa.append(ucus_yuksekligi)    
            print(ucus_yuksekligi)
            print("\n")
            i+=1
            if i==birak:
                break
            
            
            
        
            
            

        #train_images = np.array(train_images)

        train_images = train_images.reshape(-1,512,512,1)
        np.save('x_train_'+str(birak)+'.npy', train_images)       
        np.save('irtifa_'+str(birak)+'.npy', irtifa)

        
        print(str(hata)+" adet görüntü hatalı olarak işlendi")


import numpy as np

x_train_7118=np.load("x_train_7118.npy")
irtifa_7118=np.load("irtifa_7118.npy")

x_train_9000 =np.load("x_train_9000.npy")
irtifa_9000 =np.load("irtifa_9000.npy")

x_train_12000 =np.load("x_train_12000.npy")
irtifa_12000 =np.load("irtifa_12000.npy")

x_train_15000 =np.load("x_train_15000.npy")
irtifa_15000 =np.load("irtifa_15000.npy")


x_train_18000 =np.load("x_train_18000.npy")
irtifa_18000 =np.load("irtifa_18000.npy")


x_train_21000 =np.load("x_train_21000.npy")
irtifa_21000 =np.load("irtifa_21000.npy")

x_train_24000 =np.load("x_train_24000.npy")
irtifa_24000 =np.load("irtifa_24000.npy")

x_train= np.concatenate((x_train_7118, x_train_9000, x_train_12000, x_train_15000, x_train_18000, x_train_21000, x_train_24000))
irtifa = np.concatenate((irtifa_7118, irtifa_9000, irtifa_12000, irtifa_15000, irtifa_18000, irtifa_21000, irtifa_24000))


np.save('x_train.npy', x_train)       
np.save('irtifa.npy', irtifa)





# cv2.imshow(" ",tmp[122])

#cv2.imshow(" ",train_images[16])


"""
import numpy as np
import matplotlib.pyplot as plt

# Veri seti oluştur


# Histogram verilerini hesapla
hist, bin_edges = np.histogram(irtifa, bins=20)

# Histogramı grafiğe dök
plt.bar(bin_edges[:-1], hist, width = 1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.title("Veri Seti Histogramı")
plt.show()

"""
