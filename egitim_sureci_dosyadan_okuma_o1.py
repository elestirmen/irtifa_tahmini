import os
import argparse
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from model_zoo import get_model
from exif_data_generator import ExifSequence


AVAILABLE_ARCHES = [
    "resnet18", "resnet34", "resnet50",
    "vgg16",
    "xception",
    "densenet121", "densenet169", "densenet201",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    "efficientnetb0", "efficientnetv2b0",
    "mobilenet", "mobilenetv1", "mobilenetv2", "mobilenetv3small", "mobilenetv3large",
    "squeezenet",
    "vit_tiny", "vit_small", "vit_base", "vit_large",
    "pvt_tiny", "pvt_small", "pvt_medium", "pvt_large",
    "pvtv2_b0", "pvtv2_b1", "pvtv2_b2", "pvtv2_b3", "pvtv2_b4", "pvtv2_b5",
    "efficientformer_l1", "efficientformer_l3", "efficientformer_l7",
    "efficientvit_m0", "efficientvit_m1", "efficientvit_m2", "efficientvit_m3", "efficientvit_m4", "efficientvit_m5",
    "custom_cnn",
]
GRAYSCALE_CAPABLE = {"resnet18", "resnet34", "custom_cnn"}


def main():
    parser = argparse.ArgumentParser(description="Train altitude regression with selectable architectures")
    parser.add_argument("--image-folder", default="output_images_irtifa_full/output_images_irtifa_full", help="Folder with training images")
    parser.add_argument("--csv", dest="csv_path", default="veri_hazirlama_etiketleme/csv_file.csv", help="CSV with filename,altitude")
    parser.add_argument("--label-source", choices=["csv", "exif"], default="csv", help="Use labels from CSV or read EXIF on the fly")
    parser.add_argument("--arch", default="vgg16", choices=AVAILABLE_ARCHES, help="Model architecture (see README for details)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--freeze-base", action="store_true", help="Freeze base (for pretrained models)")
    parser.add_argument("--input-size", type=int, default=512, help="Input resolution (square)")
    parser.add_argument("--color-mode", default=None, choices=["rgb", "grayscale"], help="Force color mode; default auto by arch")
    args = parser.parse_args()

    # Decide channels by arch or explicit flag

    arch_key = args.arch.lower()
    if args.color_mode is not None:
        color_mode = args.color_mode
    else:
        color_mode = 'grayscale' if arch_key in GRAYSCALE_CAPABLE else 'rgb'

    if color_mode == 'grayscale' and arch_key not in GRAYSCALE_CAPABLE:
        raise ValueError(f"{args.arch} requires RGB inputs; grayscale mode is not supported.")

    channels = 3 if color_mode == 'rgb' else 1

    target_size = (args.input_size, args.input_size)
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    if args.label_source == "csv":
        dataframe = pd.read_csv(args.csv_path)
        train_df, val_df = train_test_split(dataframe, test_size=0.2, random_state=42)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=args.image_folder,
            x_col="filename",
            y_col="altitude",
            target_size=target_size,
            batch_size=args.batch_size,
            class_mode='raw',
            color_mode=color_mode,
            shuffle=True
        )

        val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            directory=args.image_folder,
            x_col="filename",
            y_col="altitude",
            target_size=target_size,
            batch_size=args.batch_size,
            class_mode='raw',
            color_mode=color_mode,
            shuffle=True
        )
    else:
        # Build file lists directly from folder and split; labels read from EXIF per batch
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF'}
        all_files = [
            os.path.join(args.image_folder, f)
            for f in os.listdir(args.image_folder)
            if os.path.splitext(f)[1] in exts
        ]
        if len(all_files) == 0:
            raise RuntimeError(f"No images found in {args.image_folder}")

        train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

        train_generator = ExifSequence(
            filepaths=train_files,
            target_size=target_size,
            batch_size=args.batch_size,
            color_mode=color_mode,
            rescale=1.0/255.0,
            shuffle=True,
            cache_labels=True,
        )
        val_generator = ExifSequence(
            filepaths=val_files,
            target_size=target_size,
            batch_size=args.batch_size,
            color_mode=color_mode,
            rescale=1.0/255.0,
            shuffle=False,
            cache_labels=True,
        )

    input_shape = (args.input_size, args.input_size, channels)
    model = get_model(
        arch=args.arch,
        input_shape=input_shape,
        lr=args.lr,
        train_base=(not args.freeze_base)
    )

    model.summary()

    os.makedirs('modeller', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    # Save every epoch (history), plus a best checkpoint
    ckpt_path = os.path.join('modeller', f'{timestamp}_{args.arch}_epoch_{{epoch:02d}}.h5')
    best_path = os.path.join('modeller', f'{timestamp}_{args.arch}_best.h5')
    checkpoint_all = ModelCheckpoint(filepath=ckpt_path, save_best_only=False, save_freq='epoch')
    checkpoint_best = ModelCheckpoint(filepath=best_path, monitor='val_loss', mode='min', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=max(args.lr*1e-3, 1e-7), mode='min', verbose=1)
    csv_log = CSVLogger(os.path.join('modeller', f'{timestamp}_{args.arch}_log.csv'))

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=[checkpoint_all, checkpoint_best, early_stop, reduce_lr, csv_log]
    )

    final_path = os.path.join('modeller', f'{timestamp}_{args.arch}_final.h5')
    model.save(final_path)
    arch_output = f"son_model_{args.arch}.h5"
    model.save(arch_output)
    weight_path = os.path.join('modeller', f'{timestamp}_{args.arch}_weights.h5')
    model.save_weights(weight_path)
    config_path = os.path.join('modeller', f'{timestamp}_{args.arch}_config.json')
    with open(config_path, 'w', encoding='utf-8') as cfg:
        cfg.write(model.to_json())
    # Legacy filename for compatibility
    model.save("son_model.h5")


if __name__ == "__main__":
    main()
