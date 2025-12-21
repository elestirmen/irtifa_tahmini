
import exifread
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint

import fonksiyonlar as fk 


# Resimleri yükleme ve boyutlandırma
def load_image(path):
    image = cv2.imread(path,0)
    
    height, width = image.shape[:2]

    
    height,width= (image.shape[0],image.shape[1])           
    
    ##########################################################
    
    # Kare boyutunu hesaplayın
    if height > width:
        square_size = width
    else:
        square_size = height
    
    # Kareyi kırpın
    x = int((width - square_size) / 2)
    y = int((height - square_size) / 2)
    cropped_img = image[y:y+square_size, x:x+square_size]
            
    
    
    
    
    
    img = cv2.resize(cropped_img,(512, 512)) # Resimleri boyutlandırma
    img_arr = np.array(img) / 255.0 # Normalizasyon
    return img_arr



def load_image2(path):
    image = cv2.imread(path,0)
    
    height, width = image.shape[:2]

    
    height,width= (image.shape[0],image.shape[1])           
    
    ##########################################################
    
    image = fk.crop_around_center(image,1024, 1024)         
      
    image=cv2.resize(image,(512, 512))

      
      
      
    img_arr = np.array(image) / 255.0 # Normalizasyon
    return img_arr




"""
irtifa=[]


directory = "goruntuler" # görüntülerin bulunduğu dizin



train_images = np.array([load_image("goruntuler/"+path) for path in os.listdir(directory)])



for filename in os.listdir(directory):
    if filename.endswith(".JPG") or filename.endswith(".jpeg") or filename.endswith(".png"): # sadece görüntü dosyalarını işle
        with open(os.path.join(directory, filename), 'rb') as f:
            tags = exifread.process_file(f)
            altitude = tags.get('GPS GPSAltitude')
            if altitude is not None:
                irtifa.append(int(altitude.values[0]))
            
            print(f"Image: {filename}, Altitude: {altitude}")
        


irtifa=np.array(irtifa)

X=train_images 
y=irtifa

"""
#%%








from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense,BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential




y = np.load("irtifa_full.npy", mmap_mode='r')
X = np.load("x_train_full.npy", mmap_mode='r')




# X = X['data']
# y = y['data']


train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



batch_size=8

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)






"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=21)


X_train=X_train.reshape(-1,512,512,1)
X_test=X_test.reshape(-1,512,512,1)


"""


"""

activation_func='relu'

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(512, 512, 1)),
    #tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(4, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D((2,2)),
    

    
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
])




#♦model = load_model("son_model.h5")

adam = tf.keras.optimizers.Adam(lr=0.00003)  

model.compile(optimizer='adam', loss='mean_squared_error')      #, metrics=['val_loss'])
model.summary()
"""

#%%


activation_func='relu'
input_shape =(512,512,1)

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


hist=model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(X_test) // batch_size,
    callbacks=[checkpoint]
)





#%%
"""

# Modeli eğitme
#history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))


checkpoint = ModelCheckpoint('_epoch_{epoch:05d}_'+activation_func+'_'+'.h5', period=1, save_best_only=True)







hist = model.fit(X_train, y_train,epochs=50,batch_size=1,validation_data=(X_test, y_test), callbacks=[checkpoint])

"""


model.save("son_model.h5")



model = tf.keras.models.load_model("son_model.h5")

#%%




"""
import matplotlib.pyplot as plt

# Histogramı oluşturalım
hist, bins = np.histogram(y, bins=100)

# Histogramı grafiğe döküp gösterelim
plt.hist(y, bins=bins)
plt.show()
"""