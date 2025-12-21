# Import necessary libraries
import os  # For file and directory operations
import time  # For measuring throughput during testing
import numpy as np  # For numerical computations
import tensorflow as tf  # For building and running TensorFlow models
from PIL import Image  # For image processing
import exifread  # To read EXIF data from images
import rasterio  # For working with raster data (e.g., DEM files)
from pyproj import Transformer  # For coordinate transformations
import matplotlib.pyplot as plt  # For plotting
import argparse
import shutil
import tempfile
from model_zoo import get_model

# ---------------------------------------------------------------------------
# Compatibility patches for loading models saved with newer Keras versions
# (e.g., dtype policy objects like `DTypePolicy`) into older TF/Keras stacks.
# We register `DTypePolicy` as an alias for the legacy `Policy` class so that
# deserialization in tf.keras 2.x does not fail.
# ---------------------------------------------------------------------------
try:
    from tensorflow.keras.mixed_precision import policy as _mp_policy

    # In TF/Keras 2.x there is no DTypePolicy class, but models saved with
    # Keras 3 serialize the dtype policy with class_name="DTypePolicy".
    # Register this name globally so deserialization maps it to Policy.
    try:
        tf.keras.utils.get_custom_objects().setdefault("DTypePolicy", _mp_policy.Policy)
    except Exception:
        pass
except Exception:
    # If anything fails here, we silently skip; the rest of the script
    # (and newer TF/Keras stacks) will continue to use their native logic.
    pass

RGB_ARCHES = {
    "resnet50",
    "vgg16",
    "efficientnetv2b0",
    "efficientnetb0",
    "mobilenetv2",
    "mobilenetv3small",
}
SUPPORTED_ARCHES = (
    "resnet50",
    "vgg16",
    "efficientnetv2b0",
    "efficientnetb0",
    "mobilenetv2",
    "mobilenetv3small",
    "resnet18",
    "resnet34",
    "custom_cnn",
)


def detect_architecture(model_name: str) -> str:
    lower = model_name.lower()
    for arch in SUPPORTED_ARCHES:
        if arch in lower:
            return arch
    return "resnet50"


def color_mode_for_arch(arch: str) -> str:
    return "rgb" if arch in RGB_ARCHES else "grayscale"

# Define a class to handle elevation data from a DEM file
class ElevationModel:
    def __init__(self, dem_file):
        """
        Initialize the ElevationModel with a DEM file.

        Parameters:
        dem_file (str): Path to the DEM (Digital Elevation Model) file.
        """
        self.dem = rasterio.open(dem_file)
        self.transform = self.dem.transform
        self.elevation_data = self.dem.read(1)
        self.height, self.width = self.elevation_data.shape

    def get_elevation(self, x, y):
        """
        Get the elevation at a specific coordinate.

        Parameters:
        x (float): X coordinate (in the same CRS as the DEM).
        y (float): Y coordinate (in the same CRS as the DEM).

        Returns:
        float or None: Elevation value or None if out of bounds or invalid.
        """
        # Convert x and y to pixel coordinates
        py, px = rasterio.transform.rowcol(self.transform, x, y)

        # Check if the coordinates are within the DEM bounds
        if 0 <= px < self.width and 0 <= py < self.height:
            elevation = self.elevation_data[py, px]
            # Check for invalid data (e.g., NaN values)
            if np.isnan(elevation):
                return None
            else:
                return elevation
        else:
            return None

def wgs84_to_utm_zone_36n(lon, lat):
    """
    Transform coordinates from WGS84 to UTM Zone 36N.

    Parameters:
    lon (float): Longitude in WGS84.
    lat (float): Latitude in WGS84.

    Returns:
    (float, float): Coordinates in UTM Zone 36N.
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    return transformer.transform(lon, lat)

def crop_around_center(image, width, height):
    """
    Crop the image around its center to the specified width and height.

    Parameters:
    image (PIL.Image): The image to be cropped.
    width (int): The desired width.
    height (int): The desired height.

    Returns:
    PIL.Image: The cropped image.
    """
    image = image.convert('RGB')  # Ensure image is in RGB mode
    image_size = (image.width, image.height)
    #image_center = (int(image_size[0] * 0.5 +150 ), int(image_size[1] * 0.5 + 150))
    image_center = (int(image_size[0] * 0.5 ), int(image_size[1] * 0.5))

    # Ensure the crop dimensions do not exceed the image dimensions
    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    # Calculate crop boundaries
    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image.crop((x1, y1, x2, y2))

def load_image2(path, target_size=(512, 512), channels=3, crop_size=1024):
    """
    Load an image from the given path, crop it around the center,
    resize it to 512x512, and normalize pixel values.

    Parameters:
    path (str): Path to the image file.

    Returns:
    np.array: The processed image array.
    """
    # Open the image
    image = Image.open(path)
    # Crop around the center with dynamic crop size
    image = crop_around_center(image, crop_size, crop_size)
    # Resize the image to model's expected input
    image = image.resize(target_size, resample=Image.NEAREST)
    # Convert color mode
    if channels == 3:
        image = image.convert('RGB')
    else:
        image = image.convert('L')
    # Convert image to numpy array and normalize
    img_arr = np.array(image) / 255.0
    # Ensure channel axis exists
    if channels == 1 and img_arr.ndim == 2:
        img_arr = np.expand_dims(img_arr, axis=-1)
    return img_arr

def dms_to_decimal(value):
    """
    Convert GPS coordinates from degrees, minutes, seconds to decimal degrees.

    Parameters:
    value: EXIF GPS coordinate value.

    Returns:
    float: Coordinate in decimal degrees.
    """
    d, m, s = value.values
    d = d.num / d.den
    m = m.num / m.den
    s = s.num / s.den
    return d + m / 60 + s / 3600

def get_gps_data(image_path):
    """
    Extract GPS data from an image's EXIF metadata.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    tuple or None: (latitude, longitude, altitude, camera_model) or None if not available.
    """
    with open(image_path, 'rb') as img_file:
        exif_data = exifread.process_file(img_file, details=False)

    if not exif_data:
        print("No EXIF data found.")
        return None

    # Get camera model
    camera_model = str(exif_data.get('Image Model', 'Unknown'))

    # Extract GPS information
    gps_info = {key: exif_data[key] for key in exif_data.keys() if key.startswith('GPS')}

    if not gps_info:
        print("No GPS data found.")
        return None

    # Extract latitude, longitude, and altitude
    gps_latitude = gps_info.get('GPS GPSLatitude')
    gps_latitude_ref = gps_info.get('GPS GPSLatitudeRef')
    gps_longitude = gps_info.get('GPS GPSLongitude')
    gps_longitude_ref = gps_info.get('GPS GPSLongitudeRef')
    gps_altitude = gps_info.get('GPS GPSAltitude')

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        latitude = dms_to_decimal(gps_latitude)
        longitude = dms_to_decimal(gps_longitude)

        if gps_latitude_ref.values[0] == 'S':
            latitude = -latitude
        if gps_longitude_ref.values[0] == 'W':
            longitude = -longitude
    else:
        print("Coordinate information not found.")
        return None

    if gps_altitude:
        altitude = gps_altitude.values[0].num / gps_altitude.values[0].den
    else:
        print("No GPS altitude data found.")
        altitude = None

    return latitude, longitude, altitude, camera_model

parser = argparse.ArgumentParser(description="Evaluate altitude regression models on a test set with DEM-based ground truth")
parser.add_argument("--models-dir", dest="models_dir", default="modeller", help="Directory containing models (.h5)")
parser.add_argument("--test-dir", dest="test_dir", default="test_arazi", help="Directory containing test images")
parser.add_argument("--dem1", dest="dem1", default="ana_harita_urgup_30_cm_utm_elevation.tif", help="Primary DEM file path")
parser.add_argument("--dem2", dest="dem2", default="karlik_30_cm_bingmap_utm_elevation.tif", help="Fallback DEM file path")
parser.add_argument("--results-file", dest="results_file", default="results_sehir.txt", help="Results output text file")
parser.add_argument("--input-size", dest="input_size", type=int, default=512, help="Square input resolution used during training")
args = parser.parse_args()

# Initialize ElevationModel instances
elevation_model = ElevationModel(args.dem1)
elevation_model_2 = ElevationModel(args.dem2)

# Directory containing models
model_dir = args.models_dir
model_list = os.listdir(model_dir)

# List to store metrics for each model
model_metrics = []
metrics_header_printed = False
MODEL_NAME_COL_WIDTH = 35
NUMERIC_COL_WIDTH = 15


def _format_model_name(name: str, width: int = MODEL_NAME_COL_WIDTH) -> str:
    if len(name) <= width:
        return name
    return name[: width - 3] + "..."


def _format_header_row() -> str:
    return (
        f"{'Model Name'.ljust(MODEL_NAME_COL_WIDTH)}"
        f"{'MAE'.ljust(NUMERIC_COL_WIDTH)}"
        f"{'MSE'.ljust(NUMERIC_COL_WIDTH)}"
        f"{'RMSE'.ljust(NUMERIC_COL_WIDTH)}"
        f"{'R^2 Score'.ljust(NUMERIC_COL_WIDTH)}"
        f"{'Speed (img/s)'.ljust(NUMERIC_COL_WIDTH)}"
        "Samples"
    )


def load_full_model_with_inputlayer_patch(model_path: str):
    """
    Load a serialized Keras model and gracefully handle the new `batch_shape`
    argument emitted by newer TensorFlow versions. Older TF releases in the
    evaluation environment do not recognize that keyword, so we retry the load
    with a shimmed InputLayer that translates `batch_shape` into
    `batch_input_shape`.
    """

    def _normalize_batch_shape(value):
        if value is None:
            return None
        if hasattr(value, "as_list"):
            return tuple(value.as_list())
        return tuple(value)

    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except (ValueError, TypeError) as exc:
        msg = str(exc).lower()
        dtype_policy_error = "dtype policy" in msg or "dtypepolicy" in msg
        if (
            "batch_shape" not in msg
            and "keyword argument not understood" not in msg
            and not dtype_policy_error
        ):
            raise
        print(
            "Full model load failed due to unsupported config (e.g., 'batch_shape'/'data_format'). "
            "Retrying with compatibility patch."
        )

        class InputLayerCompat(tf.keras.layers.InputLayer):
            def __init__(self, *args, batch_shape=None, **kwargs):
                normalized_shape = _normalize_batch_shape(batch_shape)
                if normalized_shape and "batch_input_shape" not in kwargs:
                    kwargs["batch_input_shape"] = normalized_shape
                # Drop potential unknowns from newer Keras
                for k in ("data_format", "dtype_policy"):
                    if k in kwargs:
                        kwargs.pop(k)
                super().__init__(*args, **kwargs)

        def _drop_incompatible_kwargs(kwargs):
            for k in ("data_format", "dtype_policy", "synchronized"):
                if k in kwargs:
                    kwargs.pop(k)

        # Augmentation layer compatibility wrappers (ignore unknown kwargs like data_format)
        # Resolve preprocessing layer bases if available
        def _resolve(name):
            try:
                return getattr(tf.keras.layers, name)
            except Exception:
                return None

        _RandomFlipBase = _resolve("RandomFlip")
        _RandomRotationBase = _resolve("RandomRotation")
        _RandomZoomBase = _resolve("RandomZoom")
        _RandomContrastBase = _resolve("RandomContrast")
        _RandomTranslationBase = _resolve("RandomTranslation")
        _RescalingBase = _resolve("Rescaling")
        _ResizingBase = _resolve("Resizing")
        _CenterCropBase = _resolve("CenterCrop")
        _BatchNormBase = _resolve("BatchNormalization")
        _PolicyBase = getattr(tf.keras.mixed_precision, "Policy", None)

        class RandomFlipCompat(_RandomFlipBase or tf.keras.layers.Layer):
            def __init__(self, *args, **kwargs):
                _drop_incompatible_kwargs(kwargs)
                if _RandomFlipBase is None:
                    super().__init__()
                else:
                    super().__init__(*args, **kwargs)

        class RandomRotationCompat(_RandomRotationBase or tf.keras.layers.Layer):
            def __init__(self, *args, **kwargs):
                _drop_incompatible_kwargs(kwargs)
                if _RandomRotationBase is None:
                    super().__init__()
                else:
                    super().__init__(*args, **kwargs)

        class RandomZoomCompat(_RandomZoomBase or tf.keras.layers.Layer):
            def __init__(self, *args, **kwargs):
                _drop_incompatible_kwargs(kwargs)
                if _RandomZoomBase is None:
                    super().__init__()
                else:
                    super().__init__(*args, **kwargs)

        class RandomContrastCompat(_RandomContrastBase or tf.keras.layers.Layer):
            def __init__(self, *args, **kwargs):
                _drop_incompatible_kwargs(kwargs)
                if _RandomContrastBase is None:
                    super().__init__()
                else:
                    super().__init__(*args, **kwargs)

        class RandomTranslationCompat(_RandomTranslationBase or tf.keras.layers.Layer):
            def __init__(self, *args, **kwargs):
                _drop_incompatible_kwargs(kwargs)
                if _RandomTranslationBase is None:
                    super().__init__()
                else:
                    super().__init__(*args, **kwargs)

        class RescalingCompat(_RescalingBase or tf.keras.layers.Layer):
            def __init__(self, *args, **kwargs):
                _drop_incompatible_kwargs(kwargs)
                if _RescalingBase is None:
                    super().__init__()
                else:
                    super().__init__(*args, **kwargs)

        class ResizingCompat(_ResizingBase or tf.keras.layers.Layer):
            def __init__(self, *args, **kwargs):
                _drop_incompatible_kwargs(kwargs)
                if _ResizingBase is None:
                    super().__init__()
                else:
                    super().__init__(*args, **kwargs)

        class CenterCropCompat(_CenterCropBase or tf.keras.layers.Layer):
            def __init__(self, *args, **kwargs):
                _drop_incompatible_kwargs(kwargs)
                if _CenterCropBase is None:
                    super().__init__()
                else:
                    super().__init__(*args, **kwargs)

        class BatchNormalizationCompat(_BatchNormBase or tf.keras.layers.Layer):
            def __init__(self, *args, **kwargs):
                _drop_incompatible_kwargs(kwargs)
                if _BatchNormBase is None:
                    super().__init__()
                else:
                    super().__init__(*args, **kwargs)

        class DTypePolicyCompat(_PolicyBase or object):
            def __init__(self, name=None, **kwargs):
                if _PolicyBase is None:
                    self.name = name or "float32"
                else:
                    super().__init__(name or "float32", **kwargs)

            @classmethod
            def from_config(cls, config):
                name = (
                    config.get("name")
                    or config.get("policy_name")
                    or config.get("compute_dtype")
                    or "float32"
                )
                return cls(name=name)

            def get_config(self):
                return {"name": getattr(self, "name", "float32")}

        custom_objects = {
            "InputLayer": InputLayerCompat,
            "RandomFlip": RandomFlipCompat,
            "RandomRotation": RandomRotationCompat,
            "RandomZoom": RandomZoomCompat,
            "RandomContrast": RandomContrastCompat,
            "RandomTranslation": RandomTranslationCompat,
            "Rescaling": RescalingCompat,
            "Resizing": ResizingCompat,
            "CenterCrop": CenterCropCompat,
            "BatchNormalization": BatchNormalizationCompat,
        }
        # Register mixed-precision dtype policy classes so models saved with newer Keras versions load.
        dtype_policy_cls = getattr(tf.keras.mixed_precision, "DTypePolicy", None)
        policy_cls = getattr(tf.keras.mixed_precision, "Policy", None)
        if dtype_policy_cls:
            custom_objects["DTypePolicy"] = dtype_policy_cls
        if policy_cls:
            custom_objects.setdefault("Policy", policy_cls)
            # If we don't have a DTypePolicy class (old TF/Keras), still map the
            # serialized name to the legacy Policy class.
            custom_objects.setdefault("DTypePolicy", policy_cls)
        # Ensure we always have a handler for DTypePolicy
        custom_objects.setdefault("DTypePolicy", DTypePolicyCompat)
        custom_objects.setdefault("Policy", policy_cls or DTypePolicyCompat)

        # Also register globally to help legacy deserializers.
        tf.keras.utils.get_custom_objects().update(custom_objects)

        with tf.keras.utils.custom_object_scope(custom_objects):
            return tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects,
            )

# Loop over each model in the directory
for model_name in model_list:
    if not model_name.lower().endswith(".h5"):
        continue

    print(f"\nModel: {model_name}\n")
    model_path = os.path.join(model_dir, model_name)
    arch = detect_architecture(model_name)
    color_mode = color_mode_for_arch(arch)
    channels = 3 if color_mode == "rgb" else 1
    input_shape = (args.input_size, args.input_size, channels)

    model = None
    try:
        model = get_model(
            arch=arch,
            input_shape=input_shape,
            lr=0.0,
            train_base=False,
        )
        model.load_weights(model_path)
        model.compile(optimizer='adam', loss='mean_absolute_error')
    except Exception as e:
        print(f"Architecture-based load failed ({str(e)}), falling back to full model load.")
        model = load_full_model_with_inputlayer_patch(model_path)
        model.compile(optimizer='adam', loss='mean_absolute_error')
        _, h, w, c = model.input_shape
        input_shape = (h, w, c)
        channels = c
        color_mode = 'rgb' if c == 3 else 'grayscale'

    # Lists to store predictions and actual values
    predictions = []
    actual_values = []

    # Directory containing test images
    test_dir = args.test_dir
    test_images = os.listdir(test_dir)
    start_time = time.perf_counter()

    # Variables to accumulate total error
    num_samples = 0

    if model.input_shape:
        try:
            _, h, w, c = model.input_shape
        except Exception:
            h, w, c = input_shape
    else:
        h, w, c = input_shape
    target_size = (w, h)
    crop_size = max(1024, max(w, h) * 2)

    # Process each test image
    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)

        # Get GPS data from image
        gps_data = get_gps_data(img_path)

        if gps_data:
            latitude, longitude, altitude, camera_model = gps_data
            if latitude is not None and longitude is not None and altitude is not None:
                # Transform coordinates to DEM's CRS (UTM Zone 36N)
                x_coord, y_coord = wgs84_to_utm_zone_36n(longitude, latitude)

                # Get elevation from DEM
                elevation = elevation_model.get_elevation(x_coord, y_coord)

                if elevation is None:
                    # Try the second DEM if elevation not found in first
                    elevation = elevation_model_2.get_elevation(x_coord, y_coord)

                if elevation is not None:
                    # Calculate the terrain altitude (relative to ground level)
                    terrain_altitude = altitude - elevation

                    # Adjust altitude based on camera model
                    if camera_model == 'L1D-20c':
                        # Apply correction factor if needed
                        terrain_altitude *= 0.669  # Adjusted altitude
                    # Else, use terrain_altitude as is

                    actual_value = terrain_altitude
                else:
                    print(f"Elevation data not found for image {img_name}.")
                    actual_value = None
            else:
                print(f"Incomplete GPS data for image {img_name}.")
                actual_value = None
        else:
            print(f"No GPS data for image {img_name}.")
            actual_value = None

        # If actual value is available, proceed with prediction
        if actual_value is not None:
            # Load and preprocess the image according to model input
            img_array = load_image2(img_path, target_size=target_size, channels=c, crop_size=crop_size)
            # Reshape image for model input
            img_input = img_array.reshape(-1, h, w, c)

            # Make prediction
            prediction = model.predict(img_input, verbose=0)
            predicted_value = prediction[0][0]  # Assuming prediction is a scalar value

            # Store predictions and actual values
            predictions.append(predicted_value)
            actual_values.append(actual_value)

            # Print result for this image
            error = abs(predicted_value - actual_value)
            print(f"Image: {img_name.ljust(20)} Predicted: {predicted_value:.2f} Actual: {actual_value:.2f} Error: {error:.2f}")
            num_samples += 1
        else:
            print(f"Skipping image {img_name} due to missing data.")

    elapsed = max(time.perf_counter() - start_time, 1e-6)
    throughput = num_samples / elapsed if elapsed > 0 else 0.0

    print(f"Processed {num_samples} images in {elapsed:.2f}s ({throughput:.2f} img/s)")

    # Calculate metrics for this model if there are valid samples
    if num_samples > 0:
        # Convert lists to numpy arrays for metric calculations
        actual_values_np = np.array(actual_values)
        predictions_np = np.array(predictions)

        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(predictions_np - actual_values_np))

        # Mean Squared Error (MSE)
        mse = np.mean((predictions_np - actual_values_np) ** 2)

        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)

        # Coefficient of Determination (R^2 Score)
        ss_res = np.sum((actual_values_np - predictions_np) ** 2)
        ss_tot = np.sum((actual_values_np - np.mean(actual_values_np)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

        # Store metrics for this model
        current_metrics = {
            'model_name': model_name,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2_score,
            'num_samples': num_samples,
            'throughput': throughput,
        }
        model_metrics.append(current_metrics)

        # Print metrics summary
        print(f"\nMetrics for model {model_name}:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Coefficient of Determination (R^2 Score): {r2_score:.4f}\n")
        if not metrics_header_printed:
            print(_format_header_row())
            print("-" * (MODEL_NAME_COL_WIDTH + NUMERIC_COL_WIDTH * 5 + len("Samples")))
            metrics_header_printed = True
        display_name = _format_model_name(model_name)
        print(f"{display_name.ljust(MODEL_NAME_COL_WIDTH)}"
              f"{mae:<{NUMERIC_COL_WIDTH}.4f}"
              f"{mse:<{NUMERIC_COL_WIDTH}.4f}"
              f"{rmse:<{NUMERIC_COL_WIDTH}.4f}"
              f"{r2_score:<{NUMERIC_COL_WIDTH}.4f}"
              f"{throughput:<{NUMERIC_COL_WIDTH}.2f}"
              f"{num_samples}")

        # Generate and save visualizations
        # Create directory for plots if it doesn't exist
        from sklearn.metrics import r2_score, mean_absolute_error
        
        # Örnek olarak, R² ve MAE gibi değerleri hesaplamak isterseniz:
        # (actual_values_np, predictions_np) verilerini zaten tanımladığınızı varsayıyoruz.
        r2 = r2_score(actual_values_np, predictions_np)
        mae = mean_absolute_error(actual_values_np, predictions_np)
        
        plots_dir = "model_plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Actual vs. Predicted Values Plot
        plt.figure(figsize=(8, 6))
        # Daha iyi görünürlük için marker boyutu s=30 ve saydamlık alpha=0.7 kullanalım
        plt.scatter(actual_values_np, predictions_np, alpha=0.7, s=30, c='blue', label='Data Points')
        
        # İdeal doğru (y = x) çizgisini kırmızı kesikli çizgi olarak ekleyip etiketliyoruz
        plt.plot([actual_values_np.min(), actual_values_np.max()],
                 [actual_values_np.min(), actual_values_np.max()],
                 'r--', label='Ideal Fit (y = x)')
        
        # Ekseni ve grafiği detaylandırma
        plt.xlabel('Actual Altitude (m)')
        plt.ylabel('Predicted Altitude (m)')
        plt.title(f'Actual vs. Predicted Altitude\nModel: {model_name} | MAE: {mae:.2f} | R²: {r2:.3f}')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'actual_vs_predicted_{model_name}.png'))
        plt.close()
        
        
        # 2. Residual Plot
        residuals = actual_values_np - predictions_np
        plt.figure(figsize=(8, 6))
        plt.scatter(actual_values_np, residuals, alpha=0.7, s=30, c='green', label='Residuals')
        # Sıfır hata çizgisini ekleyip etiketliyoruz
        plt.axhline(y=0, color='r', linestyle='--', label='Zero Error Line')
        
        # Y ekseni sınırlarını +200 ve -200 arasında ayarlıyoruz
        plt.ylim(-200, 200)
        
        plt.xlabel('Actual Altitude (m)')  # X eksenini Actual Altitude olarak değiştirdik
        plt.ylabel('Residuals (m)')
        plt.title(f'Residual Plot\nModel: {model_name} | MAE: {mae:.2f} | R²: {r2:.3f}')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'residuals_{model_name}.png'))
        plt.close()


        
        # 3. Error Distribution Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error (m)')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution\nModel: {model_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'error_distribution_{model_name}.png'))
        plt.close()
        
        print(f"Plots saved in '{plots_dir}' directory.")

    else:
        print(f"No valid samples for model {model_name}.\n")

# Write results to a file
with open(args.results_file, 'w') as f:
    header = _format_header_row() + "\n"
    f.write(header)
    f.write("-" * (MODEL_NAME_COL_WIDTH + NUMERIC_COL_WIDTH * 5 + len("Samples")) + "\n")
    for metrics in model_metrics:
        display_name = _format_model_name(metrics['model_name'])
        f.write(f"{display_name.ljust(MODEL_NAME_COL_WIDTH)}"
                f"{metrics['mae']:<{NUMERIC_COL_WIDTH}.4f}"
                f"{metrics['mse']:<{NUMERIC_COL_WIDTH}.4f}"
                f"{metrics['rmse']:<{NUMERIC_COL_WIDTH}.4f}"
                f"{metrics['r2_score']:<{NUMERIC_COL_WIDTH}.4f}"
                f"{metrics['throughput']:<{NUMERIC_COL_WIDTH}.2f}"
                f"{metrics['num_samples']}\n")
