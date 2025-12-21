# Deep Learning-Based Altitude Estimation from Aerial Images Using DEM-Assisted Labeling

<div align="center">

**An End-to-End Deep Learning Pipeline for Flight Altitude Regression from Drone/Satellite Imagery**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Abstract

This research presents a comprehensive deep learning framework for estimating flight altitude from aerial images captured by drones or satellites. The proposed methodology addresses a fundamental challenge in remote sensing and autonomous navigation: determining the relative height above ground from single-view aerial imagery. Unlike traditional GPS-based methods that provide absolute elevation above sea level, our approach leverages Digital Elevation Model (DEM) data to automatically generate accurate ground-truth labels by computing the relative flight altitude as the difference between GPS altitude and terrain elevation at the image capture location.

The system employs a diverse set of deep learning architectures, including convolutional neural networks (CNNs) such as ResNet, EfficientNet, MobileNet, DenseNet, and Vision Transformers (ViT), along with custom lightweight architectures optimized for regression tasks. To enhance model generalization and robustness, extensive data augmentation techniques are applied, including geometric transformations (rotation at 30° intervals) and scale variations (negative zoom) that simulate different viewing angles and flight altitudes.

The framework provides a complete end-to-end pipeline from raw image preprocessing and DEM-assisted labeling to model training, evaluation, and deployment. Comprehensive experimental evaluation demonstrates the feasibility of automated altitude estimation from georeferenced aerial imagery, with support for multiple camera models and geographic regions. The system achieves robust performance across diverse terrain types, including urban and rural environments, making it suitable for applications in photogrammetry, autonomous navigation, and remote sensing.

**Keywords:** Deep Learning, Altitude Estimation, Aerial Imagery, Digital Elevation Model (DEM), Regression, Computer Vision, Remote Sensing, Transfer Learning, Vision Transformers

---

## Table of Contents

- [Introduction](#introduction)
  - [Problem Statement](#problem-statement)
  - [Research Objectives](#research-objectives)
  - [Contributions](#contributions)
  - [Related Work](#related-work)
- [Methodology](#methodology)
  - [Data Labeling Strategy](#data-labeling-strategy)
  - [Coordinate Transformation](#coordinate-transformation)
  - [Camera Model Calibration](#camera-model-calibration)
  - [Data Augmentation](#data-augmentation)
  - [Model Architectures](#model-architectures)
  - [Training Procedure](#training-procedure)
- [Experimental Setup](#experimental-setup)
  - [Dataset](#dataset)
  - [Hardware and Software](#hardware-and-software)
  - [Hyperparameters](#hyperparameters)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Test Sets](#test-sets)
- [Results and Evaluation](#results-and-evaluation)
  - [Model Performance](#model-performance)
  - [Architecture Comparison](#architecture-comparison)
  - [Ablation Studies](#ablation-studies)
  - [Visualization and Analysis](#visualization-and-analysis)
- [Installation and Usage](#installation-and-usage)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Manual Workflow](#manual-workflow)
  - [Advanced Configuration](#advanced-configuration)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
  - [DEM Processing](#dem-processing)
  - [EXIF Metadata Extraction](#exif-metadata-extraction)
  - [Model Compatibility](#model-compatibility)
- [Limitations and Future Work](#limitations-and-future-work)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Introduction

### Problem Statement

Accurate altitude estimation from aerial images is a critical task in remote sensing, photogrammetry, and autonomous navigation systems. Traditional methods rely on GPS altitude measurements, which provide absolute elevation above sea level (MSL) but do not directly indicate the relative height above ground level (AGL)—a crucial parameter for numerous applications including:

- **Obstacle Avoidance**: Autonomous drones require real-time altitude estimation to navigate safely above terrain
- **Terrain Following**: Military and civilian aircraft need precise ground clearance measurements
- **3D Reconstruction**: Photogrammetric workflows benefit from accurate altitude data for scale calibration
- **Remote Sensing**: Environmental monitoring and mapping applications depend on precise altitude measurements
- **Search and Rescue**: Emergency response operations require accurate altitude data for mission planning

The challenge lies in extracting altitude information from single-view aerial imagery without relying solely on GPS metadata, which may be unavailable, inaccurate, or insufficient for relative altitude determination. This research addresses this gap by developing a deep learning-based approach that learns visual cues from aerial imagery to estimate flight altitude.

### Research Objectives

This study aims to:

1. **Develop an Automated Labeling System**: Create a robust pipeline that uses DEM data to compute ground-truth flight altitudes from GPS metadata embedded in EXIF data, eliminating the need for manual annotation
2. **Comprehensive Architecture Evaluation**: Systematically investigate the performance of 30+ deep learning architectures (CNNs and Vision Transformers) for altitude regression from aerial imagery
3. **Data Augmentation Analysis**: Evaluate the impact of geometric and photometric augmentation strategies on model generalization and robustness
4. **Reproducible Research Pipeline**: Provide a complete, well-documented end-to-end pipeline for altitude estimation research that can be easily adapted to different datasets and geographic regions
5. **Camera Model Adaptation**: Develop calibration strategies for different camera models to account for sensor-specific characteristics

### Contributions

This research makes the following key contributions:

- **DEM-Assisted Automated Labeling**: A novel approach to ground-truth generation that combines GPS metadata with DEM data to automatically compute relative flight altitudes, eliminating manual annotation requirements
- **Comprehensive Architecture Comparison**: Extensive evaluation of 30+ deep learning architectures spanning CNNs (ResNet, EfficientNet, MobileNet, DenseNet, VGG, Xception, ConvNeXt) and Vision Transformers (ViT, PVT, EfficientFormer, EfficientViT), providing insights into architecture selection for altitude regression tasks
- **Robust Data Augmentation Pipeline**: Systematic application of rotation-based (30° intervals) and negative zoom augmentation strategies to enhance model generalization across different viewing angles and simulated flight altitudes
- **End-to-End Research Framework**: A complete, reproducible pipeline from raw image preprocessing and labeling to model training, evaluation, and deployment, facilitating future research in aerial image analysis
- **Camera Model Calibration**: Field-calibrated correction factors for different camera models (e.g., DJI Mavic 2 Pro, Mavic 2 Zoom) to account for sensor-specific altitude measurement characteristics
- **Custom Lightweight Architecture**: A novel lightweight CNN architecture with squeeze-and-excitation (SE) attention mechanisms optimized for altitude regression, demonstrating competitive performance with reduced computational requirements

### Related Work

Altitude estimation from aerial imagery has been addressed through various approaches:

- **Photogrammetric Methods**: Traditional stereo vision and structure-from-motion (SfM) techniques require multiple overlapping images and feature matching
- **GPS-Based Methods**: Direct use of GPS altitude data, limited by GPS accuracy and lack of relative altitude information
- **DEM Integration**: Previous work has integrated DEM data for terrain analysis but not specifically for single-image altitude estimation
- **Deep Learning for Remote Sensing**: Recent advances in CNNs and Vision Transformers for remote sensing tasks, but limited application to altitude regression

This work extends previous research by combining DEM-assisted labeling with deep learning regression, enabling accurate altitude estimation from single aerial images.

---

## Methodology

### Data Labeling Strategy

The core innovation of this research is the automated generation of ground-truth labels using DEM data. The relative flight altitude (AGL) is computed using the following formula:

```
flight_altitude_AGL = GPSAltitude_MSL - DEM_elevation_MSL
```

where:
- `GPSAltitude_MSL`: GPS altitude above mean sea level (extracted from EXIF metadata)
- `DEM_elevation_MSL`: Terrain elevation above mean sea level (extracted from DEM at GPS coordinates)
- `flight_altitude_AGL`: Relative flight altitude above ground level (the target label)

#### Mathematical Formulation

Given an aerial image $I$ with GPS coordinates $(lat, lon)$ and GPS altitude $h_{GPS}$, the ground-truth altitude label $y$ is computed as:

$$y = h_{GPS} - \text{DEM}(T(lat, lon))$$

where:
- $T: (lat, lon) \rightarrow (x_{UTM}, y_{UTM})$ is the coordinate transformation function
- $\text{DEM}(x, y)$ is the elevation value at UTM coordinates $(x, y)$
- $y$ represents the relative altitude above ground level

#### Label Generation Process

1. **EXIF Metadata Extraction**: GPS coordinates and altitude are extracted from image EXIF data using `piexif` and `exifread` libraries
2. **Coordinate Transformation**: GPS coordinates (WGS84, EPSG:4326) are transformed to DEM coordinate system (UTM Zone 36N, EPSG:32636)
3. **DEM Query**: Elevation value is retrieved from DEM raster at transformed coordinates using bilinear interpolation
4. **Altitude Calculation**: Relative altitude is computed as the difference between GPS altitude and DEM elevation
5. **Camera Calibration**: Model-specific correction factors are applied based on camera model extracted from EXIF

### Coordinate Transformation

Accurate spatial alignment between GPS coordinates and DEM data is critical for reliable label generation. The system implements coordinate transformation using the PROJ library via `pyproj`.

#### Coordinate Reference Systems

- **Input (GPS)**: WGS84 Geographic Coordinate System (EPSG:4326)
  - Latitude/Longitude in decimal degrees
  - Global coverage, standard for GPS devices
  
- **Output (DEM)**: UTM Zone 36N Projected Coordinate System (EPSG:32636)
  - Universal Transverse Mercator projection
  - Optimized for Turkey region (longitude 30°E to 36°E)
  - Units in meters, enabling direct distance calculations

#### Transformation Implementation

```python
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
x_utm, y_utm = transformer.transform(longitude, latitude)
```

The transformation accounts for:
- Ellipsoid differences between WGS84 and UTM
- Map projection distortions
- Coordinate system conventions (always_xy=True ensures longitude-first ordering)

### Camera Model Calibration

Different camera models exhibit systematic biases in altitude measurements due to sensor characteristics, lens distortion, and calibration differences. The system implements camera-specific correction factors:

#### Calibration Factors

- **L1D-20c (DJI Mavic 2 Pro)**: 
  - Scale factor: $\alpha = 0.669$
  - Calibrated through field experiments comparing GPS altitude with ground-truth measurements
  - Applied as: $y_{corrected} = y \times \alpha$

- **FC2204 (DJI Mavic 2 Zoom)**:
  - Focal length-dependent scaling: $y_{corrected} = y \times \frac{4.386}{f}$
  - Accounts for variable focal length in zoom cameras
  - $f$ is extracted from EXIF metadata

- **Other Models**:
  - Direct difference used: $y_{corrected} = y$
  - Assumes minimal systematic bias

#### Calibration Methodology

Calibration factors were determined through:
1. Controlled flight experiments at known altitudes
2. Comparison of GPS-derived altitudes with ground-truth measurements
3. Statistical analysis to identify systematic biases
4. Least-squares fitting to determine optimal correction factors

### Data Augmentation

Data augmentation is crucial for improving model generalization and robustness. The system implements multiple augmentation strategies:

#### 1. Rotation Augmentation

Images are rotated at regular intervals to simulate different viewing angles and camera orientations:

- **Rotation Angles**: $0°, 30°, 60°, 90°, 120°, 150°, 180°, 210°, 240°, 270°, 300°, 330°$ (12 variants per image)
- **Implementation**: PIL `Image.rotate()` with `expand=True` to preserve full image content
- **Rationale**: Aerial images may be captured at various orientations; rotation augmentation ensures the model learns orientation-invariant features

#### 2. Negative Zoom Augmentation

Scale variations simulate different flight altitudes:

- **Zoom Factors**: Multiple scale factors applied (e.g., 0.8x, 0.9x, 1.0x, 1.1x, 1.2x)
- **Implementation**: Center cropping followed by resizing to simulate zoom effects
- **Rationale**: Different zoom levels correspond to different effective flight altitudes; this augmentation helps the model generalize across altitude ranges

#### 3. Spatial Normalization

Consistent image preprocessing ensures uniform input dimensions:

- **Center Crop**: 1024×1024 pixels from image center
- **Resize**: Downsample to 512×512 pixels (model input size)
- **Color Conversion**: Ensure RGB format (3 channels) or grayscale (1 channel) based on model requirements
- **Normalization**: Pixel values scaled to [0, 1] range by dividing by 255.0

#### 4. Photometric Augmentation (Custom CNN)

For the custom CNN architecture, additional photometric augmentations are applied during training:

- **Random Horizontal Flip**: Probability 0.5
- **Random Rotation**: ±5% (small rotations for fine-tuning)
- **Random Zoom**: ±10% scale variation
- **Random Contrast**: ±10% contrast adjustment

These augmentations are implemented as TensorFlow/Keras preprocessing layers and applied on-the-fly during training.

### Model Architectures

The framework supports a comprehensive set of deep learning architectures, enabling systematic comparison and selection of optimal models for altitude regression.

#### Convolutional Neural Networks (CNNs)

##### ResNet Family
- **ResNet18**: 18-layer residual network, lightweight option
- **ResNet34**: 34-layer residual network, balanced performance
- **ResNet50**: 50-layer residual network, standard baseline
- **Architecture**: Residual blocks with skip connections, batch normalization, ReLU activations
- **Pre-trained Weights**: ImageNet (where applicable)
- **Input**: RGB (3 channels) or Grayscale (1 channel for ResNet18/34)

##### EfficientNet Family
- **EfficientNetB0**: Baseline EfficientNet with compound scaling
- **EfficientNetV2B0**: Improved EfficientNetV2 with progressive training strategies
- **Architecture**: Mobile inverted bottleneck convolutions (MBConv) with squeeze-and-excitation attention
- **Pre-trained Weights**: ImageNet
- **Input**: RGB (3 channels)

##### MobileNet Family
- **MobileNetV1**: Depthwise separable convolutions for efficiency
- **MobileNetV2**: Inverted residuals with linear bottlenecks
- **MobileNetV3Small**: Hardware-aware network architecture search (NAS)
- **MobileNetV3Large**: Larger variant of MobileNetV3
- **Architecture**: Optimized for mobile/edge deployment with reduced parameters
- **Pre-trained Weights**: ImageNet
- **Input**: RGB (3 channels)

##### DenseNet Family
- **DenseNet121**: 121-layer densely connected network
- **DenseNet169**: 169-layer variant
- **DenseNet201**: 201-layer variant
- **Architecture**: Dense blocks with feature reuse, transition layers
- **Pre-trained Weights**: ImageNet
- **Input**: RGB (3 channels)

##### Other CNNs
- **VGG16**: 16-layer VGG network, classical architecture
- **Xception**: Extreme version of Inception with depthwise separable convolutions
- **ConvNeXt**: Modern CNN architecture inspired by Vision Transformers
  - Variants: Tiny, Small, Base, Large
- **SqueezeNet**: Lightweight architecture with fire modules
- **Pre-trained Weights**: ImageNet (where applicable)
- **Input**: RGB (3 channels)

##### Custom CNN Architecture

A novel lightweight CNN architecture specifically designed for altitude regression:

**Architecture Details**:
- **Stem**: 3×3 convolutions with stride 2, batch normalization, ReLU
- **Stages**: 4 stages with bottleneck blocks
  - Stage 1: 32 filters, 2 blocks, stride 1
  - Stage 2: 64 filters, 3 blocks, stride 2
  - Stage 3: 96 filters, 4 blocks, stride 2
  - Stage 4: 128 filters, 2 blocks, stride 2
- **Bottleneck Blocks**: 1×1 conv → 3×3 depthwise conv → 1×1 conv with SE attention
- **Attention**: Squeeze-and-Excitation (SE) modules with reduction ratio 0.25
- **Regularization**: Spatial dropout (0.05-0.10) and weight decay
- **Head**: Global Average Pooling → Dense(512) → Dropout(0.35) → Dense(128) → Dropout(0.2) → Linear(1)
- **Loss Function**: Huber Loss (delta=50.0) for robust regression
- **Input**: Grayscale (1 channel) or RGB (3 channels)

#### Vision Transformers

##### ViT (Vision Transformer) Variants
- **ViT-Tiny**: 12 transformer layers, 192 embedding dimension
- **ViT-Small**: 12 transformer layers, 384 embedding dimension
- **ViT-Base**: 12 transformer layers, 768 embedding dimension
- **ViT-Large**: 24 transformer layers, 1024 embedding dimension
- **Architecture**: Patch embedding → Transformer encoder → Classification head
- **Pre-trained Weights**: ImageNet-21k (where available)
- **Input**: RGB (3 channels)

##### PVT (Pyramid Vision Transformer) Family
- **PVT v1**: Tiny, Small, Medium, Large variants
- **PVTv2**: Improved PVT with overlapping patch embedding
  - Variants: B0, B1, B2, B3, B4, B5
- **Architecture**: Hierarchical transformer with progressive downsampling
- **Pre-trained Weights**: ImageNet (where available)
- **Input**: RGB (3 channels)

##### EfficientFormer Family
- **EfficientFormer-L1**: Lightweight transformer variant
- **EfficientFormer-L3**: Medium variant
- **EfficientFormer-L7**: Large variant
- **Architecture**: MetaFormer architecture with token mixer and MLP blocks
- **Pre-trained Weights**: ImageNet (where available)
- **Input**: RGB (3 channels)

##### EfficientViT Family
- **EfficientViT-M0 to M5**: 6 variants with increasing capacity
- **Architecture**: Efficient vision transformer with depthwise convolutions
- **Pre-trained Weights**: ImageNet (where available)
- **Input**: RGB (3 channels)

#### Model Architecture Details

All models follow a consistent transfer learning approach:

##### Base Architecture
- **Pre-trained Weights**: ImageNet (for CNNs) or ImageNet-21k (for Vision Transformers)
- **Trainable Base**: Configurable (frozen or fine-tuned)
- **Feature Extraction**: Pre-trained backbone extracts visual features

##### Regression Head
- **Global Average Pooling**: Spatial feature aggregation (for CNNs)
- **Flatten/Token Aggregation**: For Vision Transformers
- **Dense Layers**: 
  - Standard: Dense(1024) → BatchNorm → ReLU → Dense(1024) → BatchNorm → ReLU → Linear(1)
  - VGG16: Dense(1024) → Dropout(0.5) → Dense(512) → Dropout(0.5) → Linear(1)
  - Custom CNN: Dense(512) → Dropout(0.35) → Dense(128) → Dropout(0.2) → Linear(1)
- **Output**: Single scalar value representing predicted altitude in meters

##### Loss Functions
- **Mean Squared Error (MSE)**: Standard regression loss for most models
  $$L_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
- **Huber Loss**: Robust regression loss for Custom CNN
  $$L_{Huber} = \begin{cases}
  \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
  \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
  \end{cases}$$
  where $\delta = 50.0$ meters

##### Optimizer
- **Adam Optimizer**: Adaptive learning rate optimization
- **Learning Rate**: Configurable (default: $5 \times 10^{-5}$)
- **Beta Parameters**: $\beta_1 = 0.9$, $\beta_2 = 0.999$ (default)
- **Epsilon**: $1 \times 10^{-7}$ (default)

### Training Procedure

The training pipeline implements best practices for deep learning regression:

#### 1. Data Splitting

- **Train/Validation Split**: 80/20 ratio using stratified or random splitting
- **Random Seed**: Fixed seed (42) for reproducibility
- **Shuffle**: Training data shuffled each epoch, validation data fixed order

#### 2. Data Loading

Two data loading strategies are supported:

##### CSV-Based Loading
- **Input**: CSV file with columns `filename` and `altitude`
- **Generator**: `ImageDataGenerator.flow_from_dataframe()`
- **Advantages**: Fast loading, pre-computed labels
- **Use Case**: When labels are already computed and stored

##### EXIF-Based Loading
- **Input**: Directory of images with EXIF metadata
- **Generator**: Custom `ExifSequence` class
- **Advantages**: No pre-computation required, labels computed on-the-fly
- **Use Case**: When working with raw images directly

#### 3. Training Callbacks

- **ModelCheckpoint**: Save best model based on validation loss
  - Monitor: `val_loss`
  - Save best only: `True`
  - File naming: `{arch}_{timestamp}_epoch_{epoch}.h5`

- **EarlyStopping**: Stop training if no improvement
  - Monitor: `val_loss`
  - Patience: 5-10 epochs
  - Restore best weights: `True`

- **ReduceLROnPlateau**: Reduce learning rate on plateau
  - Monitor: `val_loss`
  - Factor: 0.5
  - Patience: 3 epochs
  - Min learning rate: $1 \times 10^{-7}$

- **CSVLogger**: Log training metrics to CSV file
  - Filename: `training_log_{timestamp}.csv`
  - Metrics: loss, val_loss, MAE, val_MAE (if available)

#### 4. Evaluation Metrics

During training and evaluation, the following metrics are computed:

- **Mean Absolute Error (MAE)**: $MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- **Mean Squared Error (MSE)**: $MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **Root Mean Squared Error (RMSE)**: $RMSE = \sqrt{MSE}$
- **Coefficient of Determination (R²)**: $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$
  where $SS_{res} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ and $SS_{tot} = \sum_{i=1}^{n}(y_i - \bar{y})^2$

---

## Experimental Setup

### Dataset

#### Image Collection

- **Source**: Aerial images captured by DJI drones
  - **DJI Mavic 2 Pro**: L1D-20c camera, 20MP sensor
  - **DJI Mavic 2 Zoom**: FC2204 camera, 12MP sensor with optical zoom
- **Geographic Coverage**: 
  - Ürgüp region, Turkey (Cappadocia)
  - Karlık region, Turkey
- **Image Format**: JPEG with embedded EXIF GPS metadata
- **Image Resolution**: Variable (typically 4000×3000 pixels or higher)
- **Total Images**: Dataset-dependent (count your `input_images/`); the provided `veri_hazirlama_etiketleme/csv_file.csv` lists 287,196 labeled/augmented samples in this snapshot

#### DEM Data

- **Resolution**: 30 cm per pixel (high-resolution elevation data)
- **Coordinate System**: UTM Zone 36N (EPSG:32636)
- **Coverage**: 
  - `ana_harita_urgup_30_cm_utm_elevation.tif`: Ürgüp region DEM
  - `karlik_30_cm_bingmap_utm_elevation.tif`: Karlık region DEM
- **Data Source**: Not specified (provider/provenance may vary)
- **Accuracy**: Not specified in this repository

#### Data Preprocessing

1. **EXIF Extraction**: GPS coordinates and altitude extracted from image metadata
2. **Coordinate Transformation**: WGS84 → UTM Zone 36N
3. **DEM Query**: Elevation retrieved at GPS coordinates
4. **Label Computation**: Relative altitude = GPS altitude - DEM elevation
5. **Image Preprocessing**: Center crop (1024×1024) → Resize (512×512) → Normalize [0, 1]

#### Data Augmentation

- **Rotation**: 12 variants per image (30° intervals)
- **Zoom**: Multiple scale factors per image
- **Total Augmented Dataset**: Determined by augmentation settings; current `veri_hazirlama_etiketleme/csv_file.csv` contains 287,196 samples

### Hardware and Software

#### Hardware

- **CPU**: Any modern multi-core CPU
- **GPU**: Optional; NVIDIA CUDA GPU recommended for faster training
- **RAM**: 16GB+ recommended (more for large datasets)
- **Storage**: Depends on dataset size (augmented datasets can be large)

#### Software

- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), macOS
- **Python**: 3.9 or higher
- **TensorFlow**: 2.x (tested with 2.8+)
- **CUDA**: Required only for NVIDIA GPU (version depends on TensorFlow build)
- **CuDNN**: Required only for NVIDIA GPU (version depends on TensorFlow build)

#### Key Libraries

- **TensorFlow/Keras**: Deep learning framework
- **PIL/Pillow**: Image processing
- **piexif**: EXIF metadata manipulation
- **rasterio**: DEM raster data reading
- **pyproj**: Coordinate transformation
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **matplotlib**: Visualization
- **scikit-learn**: Evaluation metrics

### Hyperparameters

| Parameter | Default Value | Range Tested | Description |
|-----------|---------------|--------------|-------------|
| Input Size | 512×512 | 256, 512, 1024 | Square input resolution |
| Batch Size | 16 | 8, 16, 32, 64 | Training batch size |
| Learning Rate | 5×10⁻⁵ | 1×10⁻⁶ to 1×10⁻³ | Adam optimizer learning rate |
| Epochs | 10-20 | 5-50 | Training epochs |
| Train/Val Split | 80/20 | 70/30, 80/20, 90/10 | Data split ratio |
| Base Trainable | True | True/False | Whether to fine-tune pre-trained base |
| Dropout Rate | 0.0-0.5 | 0.0-0.7 | Dropout probability (model-dependent) |
| Weight Decay | 0.0 | 0.0-1×10⁻⁴ | L2 regularization (Custom CNN) |

#### Hyperparameter Selection

Hyperparameters were selected through:
1. **Literature Review**: Standard values from similar regression tasks
2. **Grid Search**: Limited grid search on learning rate and batch size
3. **Validation Performance**: Selection based on validation set performance
4. **Computational Constraints**: Batch size adjusted based on GPU memory

### Evaluation Metrics

#### Regression Metrics

- **Mean Absolute Error (MAE)**: 
  - Interpretation: Average prediction error in meters
  - Units: Meters
  - Lower is better
  
- **Mean Squared Error (MSE)**:
  - Interpretation: Average squared prediction error
  - Units: Meters²
  - Lower is better
  - Sensitive to outliers

- **Root Mean Squared Error (RMSE)**:
  - Interpretation: Standard deviation of prediction errors
  - Units: Meters
  - Lower is better
  - More interpretable than MSE

- **Coefficient of Determination (R²)**:
  - Interpretation: Proportion of variance explained by model
  - Range: $(-\infty, 1]$
  - Perfect fit: R² = 1.0
  - Higher is better
  - Can be negative if model performs worse than baseline (mean prediction)

#### Inference Speed

- **Throughput**: Images processed per second (img/s)
- **Latency**: Time per image prediction (milliseconds)
- **Batch Processing**: Throughput measured with batch size = 1 and batch size = 16

### Test Sets

Two independent test sets are used for evaluation:

#### Urban Test Set (`test_sehir`)
- **Environment**: City/urban environment images
- **Characteristics**: 
  - Buildings, roads, infrastructure
  - Higher altitude variation
  - More complex scene geometry
- **Size**: 762 images
- **Altitude Range**: Derived from EXIF+DEM (AGL); varies by dataset

#### Rural Test Set (`test_arazi`)
- **Environment**: Rural/terrain environment images
- **Characteristics**:
  - Natural terrain, vegetation
  - More uniform altitude distribution
  - Simpler scene geometry
- **Size**: 1024 images
- **Altitude Range**: Derived from EXIF+DEM (AGL); varies by dataset

---

## Results and Evaluation

### Model Performance

The framework evaluates all trained models on independent test sets, generating comprehensive performance metrics and visualizations.

#### Quantitative Results

Quantitative metrics are written to result files during evaluation:

- `results.txt`: default pipeline output
- `results_sehir.txt`: urban test set example output
- `results_arazi.txt`: rural test set example output

#### Performance by Test Set

**Urban Test Set Performance**:
- See `results_sehir.txt` for the current snapshot

**Rural Test Set Performance**:
- See `results_arazi.txt` for the current snapshot

### Architecture Comparison

#### CNN vs. Vision Transformer

See the `results_*.txt` files and `model_plots/` outputs for per-architecture comparisons.

- **CNNs**: Generally faster inference, lower memory requirements
- **Vision Transformers**: Potentially better feature representation, higher computational cost

#### Model Size vs. Performance Trade-off

See the `results_*.txt` files for speed/accuracy trade-offs across architectures.

- **Lightweight Models** (MobileNet, EfficientNetB0): Suitable for edge deployment
- **Large Models** (ResNet50, ViT-Large): Higher accuracy, higher computational cost

### Ablation Studies

#### Data Augmentation Impact

Ablation study results are not tracked in this repository snapshot.

#### Transfer Learning Impact

Ablation study results are not tracked in this repository snapshot.

#### Camera Calibration Impact

Ablation study results are not tracked in this repository snapshot.

### Visualization and Analysis

The evaluation pipeline generates comprehensive visualizations for each model:

#### 1. Actual vs. Predicted Scatter Plot

- **X-axis**: Actual altitude (ground truth)
- **Y-axis**: Predicted altitude (model output)
- **Ideal Line**: y = x (perfect predictions)
- **Interpretation**: 
  - Points close to ideal line indicate accurate predictions
  - Systematic deviations indicate bias
  - Scatter indicates prediction variance

#### 2. Residual Plot

- **X-axis**: Actual altitude
- **Y-axis**: Residuals (Actual - Predicted)
- **Zero Line**: y = 0 (no error)
- **Interpretation**:
  - Random scatter around zero indicates good fit
  - Patterns indicate systematic errors or heteroscedasticity
  - Outliers indicate difficult samples

#### 3. Error Distribution Histogram

- **X-axis**: Prediction error (meters)
- **Y-axis**: Frequency
- **Interpretation**:
  - Normal distribution centered at zero indicates good model
  - Skewness indicates systematic bias
  - Wide distribution indicates high variance

#### 4. Model Comparison Plots

- Side-by-side comparison of multiple models
- Performance metrics visualization
- Inference speed vs. accuracy trade-off plots

---

## Installation and Usage

### Prerequisites

#### System Requirements

- **Python**: 3.9 or higher (tested with 3.9, 3.10, 3.11)
- **Operating System**: 
  - Windows 10/11 (PowerShell 5.1+)
  - Linux (Ubuntu 20.04+, Debian 10+)
  - macOS (10.15+)
- **GPU** (Optional but Recommended): 
  - NVIDIA GPU with CUDA support
  - CUDA 11.2+ and CuDNN 8.1+ for TensorFlow GPU acceleration
  - Minimum 4GB VRAM for training, 2GB for inference

#### Required System Libraries

For `rasterio` and `pyproj`, system-level dependencies are required:

**Windows**:
- GDAL library (recommended: install via Conda)
- PROJ library (included with Conda installation)

**Linux**:
```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev proj-data proj-bin
```

**macOS**:
```bash
brew install gdal proj
```

### Installation

#### Method 1: Using Conda (Recommended for Windows)

Conda provides pre-compiled binaries for GDAL and PROJ, simplifying installation:

```bash
# Create a new conda environment
conda create -n altitude_estimation python=3.9
conda activate altitude_estimation

# Install system dependencies
conda install -c conda-forge rasterio pyproj gdal

# Install Python packages
pip install pillow piexif numpy pandas tqdm tensorflow exifread matplotlib scikit-learn
```

#### Method 2: Using pip (Linux/macOS)

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install system dependencies first (see above)
# Then install Python packages
pip install pillow piexif rasterio pyproj numpy pandas tqdm tensorflow exifread matplotlib scikit-learn
```

#### Method 3: Using requirements.txt

Install Python packages from the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Note: GDAL/PROJ system dependencies may still be required for `rasterio`/`pyproj` (see above).

#### Verification

Verify installation:

```bash
python -c "import PIL, piexif, rasterio, pyproj, numpy, pandas, tqdm, tensorflow, exifread, matplotlib, sklearn; print('All packages installed successfully')"
```

### Quick Start

#### End-to-End Pipeline (PowerShell - Windows)

The easiest way to run the complete pipeline is using the provided PowerShell script:

1. **Place DEM files** in the repository root directory:
   - `ana_harita_urgup_30_cm_utm_elevation.tif`
   - `karlik_30_cm_bingmap_utm_elevation.tif`

2. **Place input images** in `input_images/` directory

3. **Run the pipeline**:

```powershell
.\run_pipeline.ps1 -Dem1 "ana_harita_urgup_30_cm_utm_elevation.tif" -Dem2 "karlik_30_cm_bingmap_utm_elevation.tif"
```

#### Custom Parameters

You can customize all pipeline parameters:

```powershell
.\run_pipeline.ps1 `
  -Arch resnet50 `
  -Epochs 20 `
  -BatchSize 32 `
  -InputSize 512 `
  -Lr 0.0001 `
  -Dem1 "ana_harita_urgup_30_cm_utm_elevation.tif" `
  -Dem2 "karlik_30_cm_bingmap_utm_elevation.tif" `
  -InputImages "input_images" `
  -OutputImages "output_images_irtifa_full" `
  -TestDir "test_sehir"
```

#### Pipeline Steps

The script automatically executes:

1. **Data Preparation**: Augmentation and DEM-assisted labeling
2. **CSV Generation**: Create training CSV from labeled images
3. **Model Training**: Train selected architecture
4. **Model Evaluation**: Evaluate on test set and generate results

### Manual Workflow

For more control or debugging, you can run each step manually:

#### Step 1: Data Preparation and Labeling

This step performs image augmentation (rotation, zoom) and computes altitude labels using DEM data:

```bash
python veri_hazirlama_etiketleme/veri_hazirlama_goruntu_cogaltma_negatif_zoom_ve_rotate_parallel_program_o1.py \
  --image-dir input_images \
  --output-dir output_images_irtifa_full \
  --dem1 ana_harita_urgup_30_cm_utm_elevation.tif \
  --dem2 karlik_30_cm_bingmap_utm_elevation.tif
```

**Parameters**:
- `--image-dir`: Directory containing input images
- `--output-dir`: Directory for augmented/labeled images
- `--dem1`: Primary DEM file path
- `--dem2`: Fallback DEM file path

**Output**: Augmented images with altitude labels embedded in filenames or EXIF

#### Step 2: CSV Generation

Create a CSV file mapping image filenames to altitude labels:

```bash
python veri_hazirlama_etiketleme/goruntuleri_csv_dosyasina_cevir_o1.py \
  --directory output_images_irtifa_full \
  --output-csv veri_hazirlama_etiketleme/csv_file.csv
```

**Parameters**:
- `--directory`: Directory containing labeled images
- `--output-csv`: Output CSV file path

**Output**: CSV file with columns `filename` and `altitude`

#### Step 3: Model Training

Train a deep learning model for altitude regression:

```bash
python egitim_sureci_dosyadan_okuma_o1.py \
  --arch resnet50 \
  --image-folder output_images_irtifa_full \
  --csv veri_hazirlama_etiketleme/csv_file.csv \
  --epochs 20 \
  --batch-size 16 \
  --input-size 512 \
  --lr 0.00005
```

**Parameters**:
- `--arch`: Model architecture (see available architectures below)
- `--image-folder`: Directory containing training images
- `--csv`: CSV file with labels
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--input-size`: Input image size (square)
- `--lr`: Learning rate
- `--freeze-base`: Freeze pre-trained base (optional)
- `--label-source`: "csv" or "exif" (default: "csv")
- `--color-mode`: "rgb" or "grayscale" (default: auto-detect)

**Available Architectures**:
- CNNs: `resnet18`, `resnet34`, `resnet50`, `vgg16`, `efficientnetb0`, `efficientnetv2b0`, `mobilenet`, `mobilenetv1`, `mobilenetv2`, `mobilenetv3small`, `mobilenetv3large`, `densenet121`, `densenet169`, `densenet201`, `xception`, `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`, `squeezenet`, `custom_cnn`
- Vision Transformers: `vit_tiny`, `vit_small`, `vit_base`, `vit_large`, `pvt_tiny`, `pvt_small`, `pvt_medium`, `pvt_large`, `pvtv2_b0`, `pvtv2_b1`, `pvtv2_b2`, `pvtv2_b3`, `pvtv2_b4`, `pvtv2_b5`, `efficientformer_l1`, `efficientformer_l3`, `efficientformer_l7`, `efficientvit_m0`, `efficientvit_m1`, `efficientvit_m2`, `efficientvit_m3`, `efficientvit_m4`, `efficientvit_m5`

**Output**: Trained model saved as `{arch}_{timestamp}_epoch_{epoch}.h5` in `modeller/` directory

#### Step 4: Model Evaluation

Evaluate trained models on a test set:

```bash
python "modeli_test_etme_koordinatlara gore irtifa verisi alarak_toplu_model_o1.py" \
  --models-dir modeller \
  --test-dir test_sehir \
  --dem1 ana_harita_urgup_30_cm_utm_elevation.tif \
  --dem2 karlik_30_cm_bingmap_utm_elevation.tif \
  --results-file results.txt \
  --input-size 512
```

**Parameters**:
- `--models-dir`: Directory containing trained models (.h5 files)
- `--test-dir`: Directory containing test images
- `--dem1`: Primary DEM file path
- `--dem2`: Fallback DEM file path
- `--results-file`: Output file for results
- `--input-size`: Input image size used during training

**Output**:
- Results text file with metrics for each model
- Visualization plots in `model_plots/` directory:
  - `actual_vs_predicted_{model_name}.png`
  - `residuals_{model_name}.png`
  - `error_distribution_{model_name}.png`

### Advanced Configuration

#### Training with EXIF Labels

Instead of pre-computing labels in CSV, read labels directly from EXIF:

```bash
python egitim_sureci_dosyadan_okuma_o1.py \
  --arch resnet50 \
  --image-folder output_images_irtifa_full \
  --label-source exif \
  --epochs 20 \
  --batch-size 16
```

#### Grayscale Training

Some architectures support grayscale input (1 channel):

```bash
python egitim_sureci_dosyadan_okuma_o1.py \
  --arch custom_cnn \
  --color-mode grayscale \
  --image-folder output_images_irtifa_full \
  --csv veri_hazirlama_etiketleme/csv_file.csv
```

#### Freezing Base Layers

Freeze pre-trained base and only train regression head:

```bash
python egitim_sureci_dosyadan_okuma_o1.py \
  --arch resnet50 \
  --freeze-base \
  --image-folder output_images_irtifa_full \
  --csv veri_hazirlama_etiketleme/csv_file.csv
```

---

## Project Structure

```
irtifa_tahmini/
|-- README.md
|-- README_EXIF.md
|-- LICENSE
|-- requirements.txt
|-- run_pipeline.ps1
|-- egitim_sureci_dosyadan_okuma_o1.py
|-- model_zoo.py
|-- fonksiyonlar.py
|-- exif_data_generator.py
|-- modeli_test_etme_koordinatlara gore irtifa verisi alarak_toplu_model_o1.py
|-- veri_hazirlama_etiketleme/
|   |-- veri_hazirlama_goruntu_cogaltma_negatif_zoom_ve_rotate_parallel_program_o1.py
|   |-- goruntuleri_csv_dosyasina_cevir_o1.py
|   `-- csv_file.csv
|-- models/
|   |-- resnet_small.py
|   `-- advanced_backbones.py
|-- input_images/
|-- output_images_irtifa_full/
|-- modeller/
|-- test_sehir/
|-- test_arazi/
|-- model_plots/
|-- results.txt
|-- results_sehir.txt
|-- results_arazi.txt
|-- ana_harita_urgup_30_cm_utm_elevation.tif
`-- karlik_30_cm_bingmap_utm_elevation.tif
```

### Key Files Description

#### `egitim_sureci_dosyadan_okuma_o1.py`
Main training script that:
- Loads data from CSV or EXIF
- Builds model from architecture name
- Trains with callbacks (checkpointing, early stopping, LR reduction)
- Saves trained models

#### `model_zoo.py`
Centralized model architecture definitions:
- Factory function `get_model()` to build any supported architecture
- Architecture-specific configurations
- Transfer learning setup (pre-trained weights, fine-tuning)

#### `exif_data_generator.py`
Custom Keras Sequence generator:
- Reads images and extracts labels from EXIF on-the-fly
- Supports caching for performance
- Handles missing/invalid EXIF data gracefully

#### `modeli_test_etme_koordinatlara gore irtifa verisi alarak_toplu_model_o1.py`
Evaluation script that:
- Loads multiple trained models
- Evaluates on test set with DEM-based ground truth
- Computes metrics (MAE, MSE, RMSE, R²)
- Generates visualization plots
- Writes results to text file

#### `veri_hazirlama_goruntu_cogaltma_negatif_zoom_ve_rotate_parallel_program_o1.py`
Data preparation script that:
- Reads images and EXIF metadata
- Performs rotation and zoom augmentation
- Queries DEM for elevation
- Computes altitude labels
- Saves augmented images with updated EXIF

---

## Technical Details

### DEM Processing

#### DEM File Format

- **Format**: GeoTIFF (.tif)
- **Coordinate System**: UTM Zone 36N (EPSG:32636)
- **Resolution**: 30 cm per pixel
- **Data Type**: Float32 (elevation in meters)
- **NoData Value**: NaN (handled gracefully)

#### Elevation Query

Elevation values are retrieved using bilinear interpolation:

1. **Coordinate Transformation**: GPS (WGS84) → UTM Zone 36N
2. **Pixel Coordinates**: UTM coordinates converted to DEM pixel indices
3. **Bounds Checking**: Verify coordinates within DEM bounds
4. **Value Retrieval**: Read elevation from DEM raster
5. **NoData Handling**: Return None if elevation is NaN or out of bounds

#### Multi-DEM Support

The system supports multiple DEM files for extended coverage:

- **Primary DEM**: First DEM file (e.g., Ürgüp region)
- **Fallback DEM**: Second DEM file (e.g., Karlık region)
- **Query Strategy**: Try primary DEM first, fallback to second if elevation not found

### EXIF Metadata Extraction

#### GPS Data Extraction

GPS information is extracted from EXIF metadata:

- **Latitude**: `GPS GPSLatitude` + `GPS GPSLatitudeRef`
- **Longitude**: `GPS GPSLongitude` + `GPS GPSLongitudeRef`
- **Altitude**: `GPS GPSAltitude`
- **Format**: Degrees/Minutes/Seconds (DMS) converted to decimal degrees

#### Camera Information

Camera model and settings extracted:

- **Camera Model**: `Image Model`
- **Focal Length**: `Exif FocalLength` (for zoom cameras)
- **Used for**: Camera-specific calibration

#### Libraries Used

- **piexif**: EXIF manipulation and writing
- **exifread**: EXIF reading (alternative library)
- **PIL/Pillow**: Image metadata access

### Model Compatibility

#### TensorFlow/Keras Version Compatibility

The codebase includes compatibility patches for loading models across different TensorFlow/Keras versions:

- **DTypePolicy Compatibility**: Handles newer Keras 3.x dtype policy serialization
- **InputLayer Compatibility**: Translates `batch_shape` to `batch_input_shape` for older TF versions
- **Preprocessing Layer Compatibility**: Wrappers for augmentation layers to handle version differences

#### Model Loading Strategy

Two loading strategies are implemented:

1. **Architecture-Based Loading** (Preferred):
   - Build model architecture from name
   - Load weights only (faster, more compatible)
   - Requires architecture name detection from filename

2. **Full Model Loading** (Fallback):
   - Load complete serialized model
   - Includes compatibility patches for version differences
   - Slower but more robust

---

## Limitations and Future Work

### Current Limitations

1. **Geographic Scope**: 
   - System optimized for UTM Zone 36N (Turkey region)
   - Adaptation required for other UTM zones or coordinate systems
   - DEM coordinate system must match or be transformable

2. **Camera Calibration**: 
   - Calibration factors derived from field experiments with specific camera models
   - May require recalibration for new camera models
   - Limited to DJI Mavic 2 Pro and Mavic 2 Zoom currently

3. **DEM Resolution Dependency**: 
   - Performance depends on DEM resolution and accuracy
   - Lower resolution DEMs may reduce label accuracy
   - DEM errors propagate to training labels

4. **Platform Support**: 
   - PowerShell script designed for Windows
   - Linux/macOS support via direct Python commands
   - Some system dependencies may vary by platform

5. **Single-Image Limitation**: 
   - Current approach uses single images
   - Does not leverage temporal or multi-view information
   - May miss contextual cues from image sequences

6. **Altitude Range**: 
   - Performance may vary across altitude ranges
   - Training data distribution affects model performance
   - Limited generalization to extreme altitudes

### Future Research Directions

1. **Multi-Scale DEM Integration**: 
   - Incorporate DEM data at multiple resolutions
   - Hierarchical elevation queries for improved accuracy
   - Adaptive resolution selection based on terrain complexity

2. **Temporal Consistency**: 
   - Leverage temporal sequences for altitude estimation
   - Video-based approaches for smoother altitude trajectories
   - Temporal smoothing and consistency constraints

3. **Uncertainty Quantification**: 
   - Implement Bayesian neural networks for uncertainty estimation
   - Ensemble methods for prediction confidence intervals
   - Calibrated uncertainty estimates for safety-critical applications

4. **Real-Time Inference**: 
   - Optimize models for real-time altitude estimation
   - Model quantization and pruning for edge devices
   - Efficient architectures for mobile/embedded deployment

5. **Cross-Domain Generalization**: 
   - Evaluate performance across different geographic regions
   - Domain adaptation techniques for new environments
   - Transfer learning from synthetic to real data

6. **Multi-Modal Fusion**: 
   - Combine visual features with additional sensors (IMU, barometer)
   - Sensor fusion for improved accuracy and robustness
   - Handling sensor failures and missing data

7. **Explainability**: 
   - Visual attention mechanisms to identify altitude-relevant features
   - Interpretability analysis of model decisions
   - Feature importance visualization

8. **Large-Scale Evaluation**: 
   - Benchmark on public aerial image datasets
   - Comparison with traditional photogrammetric methods
   - Standardized evaluation protocols

---

## Contributing

This is an academic research project. Contributions, suggestions, and bug reports are welcome and appreciated.

### How to Contribute

1. **Bug Reports**: Open an issue describing the bug, steps to reproduce, and expected behavior
2. **Feature Requests**: Propose new features or improvements via issues
3. **Code Contributions**: Submit pull requests with clear descriptions of changes
4. **Documentation**: Improve documentation, add examples, or fix typos
5. **Testing**: Test on different platforms and report compatibility issues

### Code Style

- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Include comments for complex logic

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

**Note**: If you use this code in your research, please cite the work appropriately (see Citation section).

---

## Citation

If you use this code, methodology, or results in your research, please cite:

```bibtex
@software{altitude_estimation_2025,
  title = {Deep Learning-Based Altitude Estimation from Aerial Images Using DEM-Assisted Labeling},
  author = {Ahmet Ertugrul Arik},
  year = {2025},
  url = {https://github.com/elestirmen/irtifa_tahmini},
  note = {Academic Research Project},
  version = {1.0}
}
```

### BibTeX for Paper (if published)

If you publish a paper, add its BibTeX entry here.

---

## Acknowledgments

We gratefully acknowledge:

- **DEM Data Providers**: Data sources used to generate the DEM rasters referenced in this repository
- **Open-Source Community**: 
  - TensorFlow and Keras teams for the deep learning framework
  - Contributors to open-source libraries (PIL, rasterio, pyproj, etc.)
- **Hardware Support**: Local compute resources used for training and evaluation
- **Testers and Contributors**: Community members who tested and improved the codebase

### Related Projects and Libraries

- **TensorFlow**: https://www.tensorflow.org/
- **Keras**: https://keras.io/
- **rasterio**: https://rasterio.readthedocs.io/
- **pyproj**: https://pyproj4.github.io/pyproj/
- **PIL/Pillow**: https://pillow.readthedocs.io/

---

<div align="center">

**For questions, collaboration inquiries, or research discussions, please contact the research team.**

**Last Updated**: 2025-12-13

</div>
