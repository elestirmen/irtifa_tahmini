import os
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
import piexif

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense,BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import datetime
import time

an = datetime.datetime.now()
timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")


image_folder = "output_images_irtifa_full"
csv_path = 'veri_hazirlama_etiketleme/csv_file.csv'



class PeriodicSave_(tf.keras.callbacks.Callback):
    def __init__(self, save_every=1000):
        self.save_every = save_every
        super(PeriodicSave, self).__init__()

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.save_every == 0:
            self.model.save(f'{timestamp}_model_step_{batch}.h5')
            
    def on_epoch_end(self, epoch, logs=None):        
        self.model.save(f'model_epoch_{epoch}.h5')
        
        
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





image_files = os.listdir(image_folder)
data = []

"""

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)
    exif_data = image._getexif()
    if exif_data:
        altitude = exif_data[34853][6]        
        
        data.append({'filename': image_file, 'altitude': altitude})
    altitude=None
df = pd.DataFrame(data)

if not df.empty:
    df.to_csv(csv_path, index=False)
else:
    print("Veri çerçevesi boş, lütfen görüntülerinizi kontrol edin.")
"""

#%%

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# CSV dosyasını oku ve eğitim ve doğrulama setlerine böl
dataframe = pd.read_csv(csv_path)
train_df, val_df = train_test_split(dataframe, test_size=0.2, random_state=42)

# ImageDataGenerator örneklerini oluştur
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Data generator'ları ayarla
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_folder,
    x_col="filename",
    y_col="altitude",
    target_size=(512, 512),
    batch_size=16,
    class_mode='raw',
    color_mode='rgb',   #color_mode='grayscale'  
    shuffle=True

)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=image_folder,
    x_col="filename",
    y_col="altitude",
    target_size=(512, 512),
    batch_size=16,
    class_mode='raw',
    color_mode='rgb',    #color_mode='grayscale'  
    shuffle=True
)


#%%

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense,BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model




# Modelinizi tanımlayın, derleyin ve eğitin
activation_func='relu'

#%%
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Model

# Encoder (EfficientNetB0 based)
input_img = Input(shape=(512, 512, 3))
base_model = EfficientNetV2B0 (weights='imagenet', include_top=False, input_tensor=input_img)
encoded = base_model.output
encoded = GlobalAveragePooling2D()(encoded)

# Fully connected layers for altitude prediction
encoded = Dense(1024, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.5)(encoded)
encoded = Dense(512, activation='relu')(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dropout(0.5)(encoded)

# Altitude Prediction Layer
altitude_output = Dense(1, activation='linear', name='altitude_output')(encoded)

# Model for altitude prediction
model = Model(inputs=input_img, outputs=altitude_output)

# adam = tf.keras.optimizers.Adam(lr=0.00005)  

#model = tf.keras.models.load_model("son_model.h5", compile=False)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()









#%%

from tensorflow.keras.callbacks import ModelCheckpoint

# Özel geri çağırma işlevini ayarlayın
periodic_save = PeriodicSave(save_every=3000)


input_shape = (544, 544, 1)
activation_func='sigmoid'
filtre_adet=32
kernel_size=(3, 3)
strds=(1, 1)
hizlandirici="GPU"



strds = str(strds).replace(",", "_")
checkpoint = ModelCheckpoint('_'+str(timestamp)+"_"+hizlandirici+'_model_f'+str(filtre_adet)+"_k"+str(kernel_size[0])+'_epoch_{epoch:05d}_'+activation_func+'_'+str(strds)+'_.h5', period=1, save_best_only=False)

# Modeli eğitme
# hist=model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[periodic_save])

hist=model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[periodic_save])

model.save("son_model.h5")





#%%





"""
strds=1
model0 = Sequential()
model0.add(Conv2D(64, (3, 3), strides=strds, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(512, 512, 1)))
model0.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model0.add(MaxPooling2D(pool_size=(2, 2)))
model0.add(BatchNormalization())

model0.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model0.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model0.add(MaxPooling2D(pool_size=(2, 2)))
model0.add(BatchNormalization())


model0.add(Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model0.add(Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model0.add(MaxPooling2D(pool_size=(2, 2)))
model0.add(BatchNormalization())

model0.add(Flatten())
model0.add(Dense(128, activation='relu'))
model0.add(Dense(64, activation='relu'))
#model.add(Dropout(0.25))
model0.add(Dense(1, activation='linear'))


model0=load_model("son_model.h5")

adam = tf.keras.optimizers.Adam(lr=0.0001)  

model0.compile(optimizer='adam', loss='mean_absolute_error')   # metrics='mean_absolute_error')   
model0.summary()


checkpoint = ModelCheckpoint('_epoch_{epoch:05d}_'+activation_func+'_'+'.h5', period=1, save_best_only=False)


hist=model0.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[checkpoint])


model0.save("son_model.h5")



"""

#%%

"""

# Modelinizi tanımlayın, derleyin ve eğitin
activation_func='relu'


strds=1
model2 = Sequential()
model2.add(Conv2D(64, (3,3 ), strides=strds, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(512, 512, 1)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(BatchNormalization())

model2.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(BatchNormalization())


model2.add(Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model2.add(Conv2D(8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(BatchNormalization())

model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(1, activation='linear'))


model2=load_model("son_model.h5")


#model2=load_model("422.h5")

adam = tf.keras.optimizers.Adam(lr=0.0001)  

model2.compile(optimizer='adam', loss='mean_absolute_error')   # metrics='mean_absolute_error')   
model2.summary()


# checkpoint = ModelCheckpoint('_epoch_{epoch:05d}_'+activation_func+'_'+'.h5', period=1, save_best_only=True)


# hist=model2.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[checkpoint])

"""


#%%  model transfer

"""
from egitim_sureci_dosyadan_okuma_heterojen_dagilimli import tf_weighted_mean_squared_error

model1=load_model("ls_1809_vls_19050.h5", custom_objects={'tf_weighted_mean_squared_error': tf_weighted_mean_squared_error})

model2=model0

first_conv_layers = [layer for layer in model1.layers if isinstance(layer, Conv2D) or isinstance(layer, Dense)]
second_conv_layers = [layer for layer in model2.layers if isinstance(layer, Conv2D) or isinstance(layer, Dense)]


# Ağırlıkları transfer etme
for first_layer, second_layer in zip(first_conv_layers, second_conv_layers):
    first_weights = first_layer.get_weights()
    second_weights = second_layer.get_weights()

    if isinstance(first_layer, Conv2D) and isinstance(second_layer, Conv2D):
        # Farklı filtre ve kanal sayısı durumunda, boyutları uyumlu hale getirip transfer eder
        min_channels = min(first_weights[0].shape[-2], second_weights[0].shape[-2])
        min_filters = min(first_weights[0].shape[-1], second_weights[0].shape[-1])
        
        # Çekirdek boyutlarına göre yayma işlemi
        kernel_shape1 = first_weights[0].shape[:2]
        kernel_shape2 = second_weights[0].shape[:2]
        max_kernel_shape = np.maximum(kernel_shape1, kernel_shape2)
        new_kernel_weights = np.zeros((*max_kernel_shape, min_channels, min_filters))
        
        # Çekirdek boyutlarındaki farkları hesapla ve ortalanmış dilimleme için başlangıç ve bitiş indekslerini belirle
        start_indices = (np.array(max_kernel_shape) - np.array(kernel_shape1)) // 2
        end_indices = start_indices + np.array(kernel_shape1)
        
        new_kernel_weights[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], :min_channels, :min_filters] = first_weights[0][:, :, :min_channels, :min_filters].copy()
        new_bias_weights = first_weights[1][:min_filters].copy()
    
        second_weights[0][:, :, :min_channels, :min_filters] = new_kernel_weights
        second_weights[1][:min_filters] = new_bias_weights
    
        second_layer.set_weights(second_weights)

        second_layer.set_weights(second_weights)
    elif isinstance(first_layer, Dense) and isinstance(second_layer, Dense):
        # Farklı nöron sayısı durumunda, boyutları uyumlu hale getirip transfer eder
        min_units = min(first_weights[0].shape[-1], second_weights[0].shape[-1])
        new_kernel_weights = first_weights[0][:min_units, :min_units].copy()
        new_bias_weights = first_weights[1][:min_units].copy()

        second_weights[0][:min_units, :min_units] = new_kernel_weights
        second_weights[1][:min_units] = new_bias_weights

        second_layer.set_weights(second_weights)
# İkinci modelin özeti
model2.summary()

# İkinci modelin derlenmesi ve eğitilmesi (fine-tuning)
#model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf_weighted_mean_squared_error, metrics=[tf_weighted_mean_squared_error])


model2.save("son_model.h5")

"""
#%%
"""

checkpoint = ModelCheckpoint('_epoch_{epoch:05d}_'+activation_func+'_'+'.h5', period=1, save_best_only=False)


hist=model0.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[checkpoint])




model0.save("son_model.h5")
"""


#model = tf.keras.models.load_model("son_model.h5")

#%%




"""
import matplotlib.pyplot as plt

# Histogramı oluşturalım
hist, bins = np.histogram(y, bins=100)

# Histogramı grafiğe döküp gösterelim
plt.hist(y, bins=bins)
plt.show()
"""
























