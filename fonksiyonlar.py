import math
import cv2
import numpy as np
from math import cos, sqrt
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import pickle
import rasterio as rio
from rasterio.warp import transform
import os,sys
from PIL import Image
from PIL.ExifTags import TAGS 
from pyproj import Transformer



def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)


def dosyaya_yaz_t(sonuclar,dogru_tahmin,yanlis_tahmin):    
    
    #model_name="sonuclar_"+model_name
    sonuclar_dosya = open("sonuclar.txt", "w")
    # sonuclar = np.vstack((sonuclar,dogru_tahmin, yanlis_tahmin)).T
    # print(sonuclar)
    
    df = pd.DataFrame(sonuclar, columns=['goruntu', 'sonuc', 'gercek_latitude', 'gercek_longitude', 'tahmini_latitude', 'tahmini_longitude','ucus_yuksekligi'])
    
    # df.loc[len(df.index)] = ["","",str(dogru_tahmin)+" dogru", str(yanlis_tahmin)+" yanlis"] 

    sonuclar_dosya.write(df.to_string())
    sonuclar_dosya.close()
    
    df.to_csv("sonuclar.csv", index=False)
    
    


import pandas as pd

def dosyaya_yaz(sonuclar, dogru_tahmin, yanlis_tahmin):
    
    # Veri çerçevesini oluştur
    df = pd.DataFrame(sonuclar, columns=['goruntu', 'sonuc', 'gercek_latitude', 'gercek_longitude', 'tahmini_latitude', 'tahmini_longitude','ucus_yuksekligi'])
    
    # Eğer her bir hücre bir liste içeriyorsa, bu listelerin ilk elemanını al
    for column in df.columns:
        df[column] = df[column].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Metin dosyasına yaz
    with open("sonuclar.txt", "w") as sonuclar_dosya:
        sonuclar_dosya.write(df.to_string())
    
    # CSV dosyasına kaydet
    df.to_csv("sonuclar.csv", index=False)



    
    
    
    
#exif bilgisi okur    
def get_field (exif,field) :
  for (k,v) in exif.items():
     if TAGS.get(k) == field:
        return v
 
 #gos coordinatını decimal sisteme çevirir
def conversion(yon,coord):
    direction = {'N':1, 'S':-1, 'E': 1, 'W':-1}  
    
    return (int(coord[0])+int(coord[1])/60.0+float(coord[2])/3600.0) * direction[yon]
"""
##lat ve long koordinatlarının görüntüdeki karşıklık gelen pikseli bulur
def piksel_bul_2(path,long,lat):
    # read image
    image_data = rio.open(path)
    # get crs objects for conversion
    to_crs = image_data.crs
    #print(to_crs)
    from_crs = rio.crs.CRS.from_epsg(4326) 
    
    #print(from_crs)
    new_x,new_y = transform(from_crs,to_crs,[long], [lat])
    
    # transform returns lists so unpack
    new_x = new_x[0]
    new_y = new_y[0]
    
    # get row and col
    row, col = image_data.index(new_x,new_y)
    return row,col
"""


def piksel_bul(path, longitude, latitude):
    """
    Find the row and column of a geographic coordinate in a raster file.
    
    Parameters:
    path (str): The path to the raster file.
    longitude (float): The longitude of the geographic coordinate.
    latitude (float): The latitude of the geographic coordinate.
    
    Returns:
    tuple: A tuple containing the row and column indices.
    """
    # Open the raster file
    with rio.open(path) as image_data:
        # Get the CRS for the raster file
        to_crs = image_data.crs

        # Initialize the transformer with high precision
        transformer = Transformer.from_crs("EPSG:4326", to_crs, always_xy=True)
        
        # Transform the coordinates
        new_x, new_y = transformer.transform(longitude, latitude)
        
        # Get the row and column index
        row, col = image_data.index(new_x, new_y)
        
    return row, col
    



def quick_distance(Lat1, Long1, Lat2, Long2):
    x = Lat2 - Lat1
    y = (Long2 - Long1) * cos((Lat2 + Lat1)*0.00872664626)  
    return 87.11 * sqrt(x*x + y*y)                     #return 111.319 * sqrt(x*x + y*y)
    
def quick_distance_utm(Lat1, Long1, Lat2, Long2):
    """
    Calculate the approximate distance between two points in UTM Zone 36.
    
    Parameters:
    - Lat1, Long1: Latitude and Longitude of the first point in decimal degrees.
    - Lat2, Long2: Latitude and Longitude of the second point in decimal degrees.
    
    Returns:
    - float: The approximate distance between the two points in meters.
    """
    
    # Create a Transformer object for WGS84 to UTM Zone 36 conversion
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    
    # Convert the coordinates to UTM Zone 36
    easting1, northing1 = transformer.transform(Long1, Lat1)
    easting2, northing2 = transformer.transform(Long2, Lat2)
    
    # Calculate the approximate distance using UTM coordinates
    x = easting2 - easting1
    y = northing2 - northing1
    
    return sqrt(x ** 2 + y ** 2)    
    
    
    
from pyproj import Proj, transform

def latlon_to_utm(latitude, longitude, zone_number=None, hemisphere=None):
    """
    Convert Latitude and Longitude to UTM coordinates.
    
    Parameters:
    - latitude (float): The latitude in decimal degrees.
    - longitude (float): The longitude in decimal degrees.
    - zone_number (int, optional): UTM zone number. If None, it will be calculated based on longitude.
    - hemisphere (str, optional): 'N' for Northern Hemisphere, 'S' for Southern Hemisphere. If None, it will be calculated based on latitude.
    
    Returns:
    - tuple: A tuple containing UTM Easting, UTM Northing, Zone Number, and Hemisphere.
    """
    
    # Calculate the UTM zone number if not provided
    if zone_number is None:
        zone_number = int((longitude + 180) / 6) + 1
    
    # Determine the hemisphere if not provided
    if hemisphere is None:
        hemisphere = 'N' if latitude >= 0 else 'S'
    
    # Create a Proj object for WGS84
    wgs84 = Proj(proj='latlong', datum='WGS84')
    
    # Create a Proj object for the UTM zone
    utm = Proj(proj='utm', zone=zone_number, datum='WGS84')
    
    # Perform the coordinate transformation
    easting, northing = transform(wgs84, utm, longitude, latitude)
    
    return easting, northing, zone_number, hemisphere


from math import sin, cos, sqrt, atan2, radians

def haversine_distance(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Compute differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Distance
    distance = R * c
    
    return distance




















def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

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


def rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """
    angle = math.radians(angle)
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x, bb_h - 2 * y)