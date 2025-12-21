import os
import pandas as pd
import piexif
from PIL import Image
from tqdm import tqdm
import argparse

def get_image_altitude(image_path):
    """
    Extracts the altitude from the image's EXIF data.

    Args:
        image_path (str): The file path to the image.

    Returns:
        float or None: The altitude if available, otherwise None.
    """
    try:
        img = Image.open(image_path)
        exif_data = piexif.load(img.info['exif'])
        gps_ifd = exif_data.get('GPS', {})
        gps_altitude = gps_ifd.get(piexif.GPSIFD.GPSAltitude)
        if gps_altitude:
            # GPSAltitude is a rational number represented as a tuple (numerator, denominator)
            altitude = gps_altitude[0] / gps_altitude[1]
            return altitude
    except KeyError:
        # EXIF data not found
        pass
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
    return None

def scan_directory_for_images(directory):
    """
    Scans the directory for images and extracts altitude information.

    Args:
        directory (str): The directory path to scan.

    Returns:
        list: A list of dictionaries containing filename and altitude.
    """
    image_data = []
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = []
    # Collect all image files
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                file_path = os.path.join(root, file)
                image_files.append(file_path)
    # Process images with a progress bar
    for file_path in tqdm(image_files, desc="Processing images", unit="image"):
        altitude = get_image_altitude(file_path)
        if altitude is not None:
            image_data.append({"filename": os.path.basename(file_path), "altitude": altitude})
    return image_data

def save_to_csv(image_data, output_csv):
    """
    Saves the image data to a CSV file.

    Args:
        image_data (list): The list of image data dictionaries.
        output_csv (str): The filename for the output CSV.
    """
    df = pd.DataFrame(image_data)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan images and extract EXIF altitude to CSV")
    parser.add_argument("--directory", default="output_images_irtifa_full", help="Directory to scan for images")
    parser.add_argument("--output-csv", dest="output_csv", default="veri_hazirlama_etiketleme/csv_file.csv", help="Output CSV path")
    args = parser.parse_args()

    image_data = scan_directory_for_images(args.directory)
    if image_data:
        save_to_csv(image_data, args.output_csv)
        print(f"Data saved to {args.output_csv}")
    else:
        print("No images with altitude data found.")
