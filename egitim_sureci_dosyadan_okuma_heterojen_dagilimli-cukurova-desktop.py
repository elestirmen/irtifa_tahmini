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
import time
import datetime

timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")



class PeriodicSave(tf.keras.callbacks.Callback):
    def __init__(self, save_every=1000):
        super(PeriodicSave, self).__init__()
        self.save_every = save_every
        self.batch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % self.save_every == 0:
            self.model.save(f'{timestamp}_model_step_{self.batch_count}.h5')
            
    def on_epoch_end(self, epoch, logs=None):        
        self.model.save(f'{timestamp}_model_epoch_{epoch}.h5')


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
            #image = image.convert('L')  # convert to grayscale
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
    value_counts = df['altitude'].value_counts()
    max_freq = value_counts.max()
    weights = max_freq / value_counts
    return df['altitude'].map(weights)


def tf_weighted_mean_squared_error(y_true, y_pred):
    weights = tf.convert_to_tensor(np.float32(compute_sample_weights(dataframe)))
    mse = tf.square(y_true - y_pred)
    weighted_mse = tf.multiply(mse, weights)
    return tf.reduce_sum(weighted_mse) / tf.reduce_sum(weights)


def tf_weighted_mean_absolute_error(y_true, y_pred):
    weights = tf.convert_to_tensor(np.float32(compute_sample_weights(dataframe)))
    mae = tf.abs(y_true - y_pred)
    weighted_mae = tf.multiply(mae, weights)
    return tf.reduce_sum(weighted_mae) / tf.reduce_sum(weights)



def tf_sqrt_weighted_mean_absolute_error(y_true, y_pred):
    weights = tf.convert_to_tensor(np.float32(compute_sample_weights(dataframe)))
    sqrt_weights = tf.sqrt(weights)
    mae = tf.abs(y_true - y_pred)
    weighted_mae = tf.multiply(mae, sqrt_weights)
    return tf.reduce_sum(weighted_mae) / tf.reduce_sum(sqrt_weights)

#t1=tf.convert_to_tensor(compute_sample_weights(dataframe))


#%%



if __name__ == '__main__':
    # CSV dosyasını oku ve eğitim ve doğrulama setlerine böl
    image_folder = "C:/d_surucusu/output_images_irtifa_full"
    csv_path = 'veri_hazirlama_etiketleme/csv_file.csv'
    
    dataframe = pd.read_csv(csv_path)

    """
    
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

    """
    #tmp=(compute_sample_weights(dataframe),dataframe['altitude'])
    #t1=compute_sample_weights(df)
    #t2=compute_sample_sqrt_weights(df)

#%%

    
    
    # Veri kümesini okuyun ve eğitim/doğrulama setlerine ayırın
    #dataframe = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(dataframe, test_size=0.2, random_state=42)
    
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
        color_mode='rgb',
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
        color_mode='rgb',
        shuffle=True
    )
    
    #%%
    



    activation_func = 'relu'
    strds = 1
    
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=strds, activation='relu',  kernel_initializer='he_normal', input_shape=(512, 512, 3)))
    model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(16, (3, 3), activation='relu',  kernel_initializer='he_normal'))
    model.add(Conv2D(8, (3, 3), activation='relu',  kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    #model = load_model("55.28.h5", custom_objects={'tf_weighted_mean_squared_error': tf_weighted_mean_squared_error})
    #model = load_model("55.28.h5", custom_objects={'tf_sqrt_weighted_mean_absolute_error': tf_sqrt_weighted_mean_absolute_error})

    model=load_model("son_model.h5",compile=False)
    
    adam = tf.keras.optimizers.Adam(lr=0.00005)
    #model.compile(optimizer=adam, loss=tf_weighted_mean_squared_error, metrics=[tf_weighted_mean_squared_error])
    model.compile(optimizer=adam, loss=tf_weighted_mean_squared_error, metrics=[tf_weighted_mean_squared_error])
    
    
    model.summary()
    
    
    #%%
    from tensorflow.keras.callbacks import ModelCheckpoint

    checkpoint = ModelCheckpoint('epoch{epoch:05d}_' + activation_func + '.h5', period=1, save_best_only=False)
    periodic_save = PeriodicSave(save_every=500)

    
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=50,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[periodic_save],
        use_multiprocessing=False,
        workers=16,
        max_queue_size=32,
        verbose=1,
        
    )
    
    hist=model.fit(train_generator, validation_data=val_generator, epochs=10)