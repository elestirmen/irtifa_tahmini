import os
import numpy as np
from PIL import Image
import piexif
from concurrent.futures import ThreadPoolExecutor, as_completed
import rasterio
from pyproj import Transformer
import argparse


class ElevationModel:
    """Class to handle elevation data from a DEM file."""
    def __init__(self, dem_file):
        self.dem = rasterio.open(dem_file)
        self.transform = self.dem.transform
        self.elevation_data = self.dem.read(1)
        self.height, self.width = self.elevation_data.shape

    def get_elevation(self, x, y):
        """
        Get the elevation at a specific coordinate.
        x, y: Coordinates in the DEM's CRS.
        """
        row, col = rasterio.transform.rowcol(self.transform, x, y)
        # Check if coordinates are within the DEM bounds
        if 0 <= col < self.width and 0 <= row < self.height:
            elevation = self.elevation_data[row, col]
            # Check for NaN values
            if np.isnan(elevation):
                return None
            else:
                return elevation
        else:
            return None


def wgs84_to_utm36(longitude, latitude):
    """Convert WGS84 coordinates to UTM Zone 36N."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    return transformer.transform(longitude, latitude)


def wgs84_to_epsg3395(longitude, latitude):
    """Convert WGS84 coordinates to World Mercator (EPSG:3395)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3395", always_xy=True)
    return transformer.transform(longitude, latitude)


def get_gps_info(image):
    """Extract GPS coordinates from image EXIF data."""
    exif_data = piexif.load(image.info['exif'])
    gps_info = exif_data.get('GPS', {})

    if piexif.GPSIFD.GPSLatitude in gps_info and piexif.GPSIFD.GPSLongitude in gps_info:
        latitude = gps_info[piexif.GPSIFD.GPSLatitude]
        latitude_ref = gps_info.get(piexif.GPSIFD.GPSLatitudeRef, b'N')
        longitude = gps_info[piexif.GPSIFD.GPSLongitude]
        longitude_ref = gps_info.get(piexif.GPSIFD.GPSLongitudeRef, b'E')

        latitude = convert_to_degrees(latitude)
        longitude = convert_to_degrees(longitude)

        if latitude_ref == b'S':
            latitude *= -1
        if longitude_ref == b'W':
            longitude *= -1

        return latitude, longitude
    else:
        return None


def convert_to_degrees(value):
    """Convert GPS coordinates stored in EXIF to degrees in float format."""
    d = value[0][0] / value[0][1]
    m = value[1][0] / value[1][1]
    s = value[2][0] / value[2][1]
    return d + (m / 60.0) + (s / 3600.0)


def crop_around_center(image, width, height):
    """Crop the image around its center to the specified width and height."""
    image = image.convert('RGB')
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
    image_path, output_directory, file_name, angle, zoom_scale, model_name = args

    with open(image_path, 'rb') as image_file:
        image = Image.open(image_file)
        gps_info = get_gps_info(image)
        if gps_info is None:
            print(f"GPS info not found for image {image_path}")
            return None
        latitude, longitude = gps_info

        # Transform coordinates to match DEM CRS
        x_coord, y_coord = wgs84_to_utm36(longitude, latitude)

        # Load EXIF data
        exif_dict = piexif.load(image.info['exif'])
        gps_altitude_rational = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
        gps_altitude = gps_altitude_rational[0] / gps_altitude_rational[1]

        focal_length_rational = exif_dict['Exif'].get(piexif.ExifIFD.FocalLength, (1, 1))
        focal_length = focal_length_rational[0] / focal_length_rational[1]

        # Get elevation from DEM
        elevation = elevation_model.get_elevation(x_coord, y_coord)

        if elevation is None:
            # Try alternative transformation and DEM
            x_coord, y_coord = wgs84_to_epsg3395(longitude, latitude)
            elevation = elevation_model_2.get_elevation(x_coord, y_coord)

        if elevation is None:
            print("Elevation value could not be obtained, process skipped.")
            return None

        # Rotate, crop, resize the image
        rotated_image = image.rotate(angle, expand=True)
        rotated_image_original = rotated_image
        rotated_image = crop_around_center(rotated_image, 1024, 1024)
        rotated_image = rotated_image.resize((512, 512), resample=Image.NEAREST)
        rotated_image = rotated_image.convert('RGB')

        # Calculate flight altitude
        flight_altitude = gps_altitude - elevation

        if "L1D-20c" in model_name:
            adjusted_altitude = flight_altitude * 0.669
            exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(adjusted_altitude), 1)
            altitude = adjusted_altitude
        else:
            adjusted_altitude = flight_altitude * (4.386 / focal_length)
            exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(adjusted_altitude), 1)
            altitude = adjusted_altitude

        # Save the rotated image
        rotation_output_path = os.path.join(output_directory, f"{angle}_{file_name}")
        rotated_image.save(rotation_output_path, exif=piexif.dump(exif_dict))

        # Process zoomed images if zoom_scale is provided
        if zoom_scale is not None and "L1D-20c" in model_name:
            width, height = rotated_image_original.size
            zoomed_image = rotated_image_original.resize((int(width * zoom_scale), int(height * zoom_scale)))
            image_cropped = crop_around_center(zoomed_image, 1024, 1024)
            image_cropped = image_cropped.resize((512, 512), resample=Image.NEAREST)
            image_cropped = image_cropped.convert('RGB')

            adjusted_altitude = (gps_altitude - elevation) * 0.669 / zoom_scale
            exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(adjusted_altitude), 1)
            zoom_output_path = os.path.join(output_directory, f"{angle}_{round(zoom_scale, 3)}_{file_name}")
            image_cropped.save(zoom_output_path, exif=piexif.dump(exif_dict))
            altitude = adjusted_altitude
        elif zoom_scale is not None and "FC2204" in model_name:
            adjusted_altitude = (gps_altitude - elevation) * (4.386 / focal_length) / zoom_scale
            width, height = rotated_image_original.size
            zoomed_image = rotated_image_original.resize((int(width * zoom_scale), int(height * zoom_scale)))
            image_cropped = crop_around_center(zoomed_image, 1024, 1024)
            image_cropped = image_cropped.resize((512, 512), resample=Image.NEAREST)
            image_cropped = image_cropped.convert('RGB')

            exif_dict['GPS'][piexif.GPSIFD.GPSAltitude] = (int(adjusted_altitude), 1)
            zoom_output_path = os.path.join(output_directory, f"{angle}_{round(zoom_scale, 3)}_{file_name}")
            image_cropped.save(zoom_output_path, exif=piexif.dump(exif_dict))
            altitude = adjusted_altitude

        return altitude


def main(image_directory: str, output_directory: str):

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List to store altitudes
    altitudes = []

    file_list = os.listdir(image_directory)
    file_list = [file for file in file_list if file.lower().endswith(('.jpg', '.jpeg'))]
    total_files = len(file_list)

    for i, image_file_name in enumerate(file_list):
        image_path = os.path.join(image_directory, image_file_name)
        file_name = os.path.basename(image_path)

        with open(image_path, 'rb') as image_file:
            image = Image.open(image_file)
            gps_info = get_gps_info(image)
            if gps_info is None:
                print(f"GPS info not found for image {image_file_name}")
                continue
            exif_dict = piexif.load(image.info['exif'])
            model_name = exif_dict['0th'][piexif.ImageIFD.Model].decode().strip()

            print(f"Processing {image_file_name} ({i+1}/{total_files})")

            process_args = []

            # Define angles and zoom scales
            angles = range(0, 360, 30)
            scales_m2z = np.linspace(0.6, 0.95, 5)
            scales_m2p = np.linspace(0.4, 0.95, 10)

            for angle in angles:
                for zoom_scale in scales_m2z:
                    process_args.append((image_path, output_directory, file_name, angle, zoom_scale, model_name))

                if "L1D-20c" in model_name:
                    for zoom_scale in scales_m2p:
                        process_args.append((image_path, output_directory, file_name, angle, zoom_scale, model_name))

            # Process images in parallel
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_image, args) for args in process_args]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            altitudes.append(result)
                    except Exception as e:
                        print(f"Error during processing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotate/zoom images and write adjusted altitude based on DEM.")
    parser.add_argument("--image-dir", dest="image_dir", default="input_images", help="Input images directory")
    parser.add_argument("--output-dir", dest="output_dir", default="output_images_irtifa_full", help="Output images directory")
    parser.add_argument("--dem1", dest="dem1", default="ana_harita_urgup_30_cm_utm_elevation.tif", help="Primary DEM file path")
    parser.add_argument("--dem2", dest="dem2", default="karlik_30_cm_bingmap_utm_elevation.tif", help="Fallback DEM file path")
    args = parser.parse_args()

    # Expose elevation models as globals for process_image
    global elevation_model, elevation_model_2
    dem_file = args.dem1
    elevation_model = ElevationModel(dem_file)
    dem_file2 = args.dem2
    elevation_model_2 = ElevationModel(dem_file2)

    main(args.image_dir, args.output_dir)
