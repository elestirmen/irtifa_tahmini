import os
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
import piexif

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np





# Veri üreteci sınıfını tanımlayın
class MyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size, image_folder, target_size, sample_weight_func):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.image_folder = image_folder
        self.target_size = target_size
        self.sample_weight_func = sample_weight_func

    def __len__(self):
        return int(tf.math.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_df = self.dataframe[index*self.batch_size:(index+1)*self.batch_size]
        images = []
        labels = []
        sample_weights = []

        for _, row in batch_df.iterrows():
            image_path = os.path.join(self.image_folder, row['filename'])
            image = Image.open(image_path)
            image = image.convert('L')  # convert to grayscale
            image = image.resize(self.target_size)
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            images.append(image_array)
            labels.append(row['altitude'])
            sample_weights.append(self.sample_weight_func(row['altitude']))

        return (
            tf.stack(images),
            tf.stack(labels),
            tf.stack(sample_weights)
        )

def compute_sample_weights(df):
    weights = {}
    value_counts = df['altitude'].value_counts()
    max_freq = value_counts.max()
    for value, freq in value_counts.items():
        weights[value] = max_freq / freq
    return df['altitude'].map(weights)


#t1=tf.convert_to_tensor(compute_sample_weights(dataframe))


def tf_weighted_mean_squared_error(y_true, y_pred):
    weights = tf.convert_to_tensor(np.float32(compute_sample_weights(dataframe)))
    mse = tf.square(y_true - y_pred)
    weighted_mse = tf.multiply(mse, weights)
    return tf.reduce_sum(weighted_mse) / tf.reduce_sum(weights)



if __name__ == '__main__':
    # CSV dosyasını oku ve eğitim ve doğrulama setlerine böl
    image_folder = 'veri_hazirlama_etiketleme/output_images'
    csv_path = 'veri_hazirlama_etiketleme/csv_file.csv'
    
    dataframe = pd.read_csv(csv_path)

    
    
    image_files = os.listdir(image_folder)
    data = []
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data:
            altitude = exif_data[34853][6]
            data.append({'filename': image_file, 'altitude': altitude})
        altitude = None
    
    df = pd.DataFrame(data)
    
    if not df.empty:
        df.to_csv(csv_path, index=False)
    else:
        print("Veri çerçevesi boş, lütfen görüntülerinizi kontrol edin.")


    #tmp=(compute_sample_weights(dataframe),dataframe['altitude'])

#%%

    
    
    # Veri kümesini okuyun ve eğitim/doğrulama setlerine ayırın
    #dataframe = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(dataframe, test_size=0.1, random_state=42)
    
    # ImageDataGenerator örneklerini oluştur
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = ImageDataGenerator(rescale=1./255)
    
    
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_folder,
        x_col="filename",
        y_col="altitude",
        target_size=(512, 512),
        batch_size=16,
        class_mode='raw',
        color_mode='grayscale',
        shuffle=True
    )
    
    val_generator = val_generator.flow_from_dataframe(
        dataframe=val_df,
        directory=image_folder,
        x_col="filename",
        y_col="altitude",
        target_size=(512, 512),
        batch_size=16,
        class_mode='raw',
        color_mode='grayscale',
        shuffle=True
    )
    
    #%%
    



    activation_func = 'relu'
    strds = 1
    
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=strds, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(512, 512, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    model = load_model("son_model.h5", custom_objects={'tf_weighted_mean_squared_error': tf_weighted_mean_squared_error})

    #model=load_model("son_model.h5")
    
    adam = tf.keras.optimizers.Adam(lr=0.00005)
    model.compile(optimizer=adam, loss=tf_weighted_mean_squared_error, metrics=[tf_weighted_mean_squared_error])
    model.summary
    
    #%%
    
    checkpoint = ModelCheckpoint('epoch{epoch:05d}_' + activation_func + '.h5', period=1, save_best_only=False)
    
    
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=50,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint],
        use_multiprocessing=False,
        workers=8,
        max_queue_size=32,
        verbose=1
    )
