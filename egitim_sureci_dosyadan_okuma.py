import os
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
import piexif

image_folder = 'veri_hazirlama_etiketleme/output_images'
csv_path = 'veri_hazirlama_etiketleme/csv_file.csv'

image_files = os.listdir(image_folder)
data = []



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


#%%

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# CSV dosyasını oku ve eğitim ve doğrulama setlerine böl
dataframe = pd.read_csv(csv_path)
train_df, val_df = train_test_split(dataframe, test_size=0.1, random_state=42)

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
    color_mode='grayscale'  

)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=image_folder,
    x_col="filename",
    y_col="altitude",
    target_size=(512, 512),
    batch_size=16,
    class_mode='raw',
    color_mode='grayscale'  
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


# Modelinizi tanımlayın, derleyin ve eğitin
activation_func='relu'


strds=1
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
#model.add(Dropout(0.25))
model.add(Dense(1, activation='linear'))


model=load_model("son_model.h5")

adam = tf.keras.optimizers.Adam(lr=0.0001)  

model.compile(optimizer='adam', loss='mean_squared_error')   # metrics='mean_absolute_error')   
model.summary()


checkpoint = ModelCheckpoint('_epoch_{epoch:05d}_'+activation_func+'_'+'.h5', period=1, save_best_only=True)


hist=model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[checkpoint])






#%%


model.save("son_model.h5")



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
























