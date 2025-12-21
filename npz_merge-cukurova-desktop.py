



"""import numpy as np

# birleştirmek istediğiniz dosyaların isimlerini bir liste olarak belirtin
file_names = ['maps_0_544.npz', 'maps_1_544.npz', 'maps_2_544.npz', 'maps_3_544.npz', 'maps_4_544.npz']


# Yeni bir dizi oluşturun ve tüm verileri bu diziye ekleyin
merged_arr_0 = np.empty(0)
merged_arr_1 = np.empty(0)

for file_name in file_names:
    # .npz dosyasını yükleyin
    data = np.load(file_name)
    # 'arr_0' ve 'arr_1' sütunlarını alın
    arr_0 = data['arr_0']
    arr_1 = data['arr_1']
    # Tüm verileri yeni bir diziye ekleyin
    if merged_arr_0.size == 0:
        merged_arr_0 = arr_0
        merged_arr_1 = arr_1
    else:
        merged_arr_0 = np.concatenate((merged_arr_0, arr_0), axis=0)
        merged_arr_1 = np.concatenate((merged_arr_1, arr_1), axis=0)

# Yeni bir .npz dosyası oluşturun
np.savez_compressed("merged_data.npz", arr_0=merged_arr_0, arr_1=merged_arr_1)

"""

#%% büyük npy dosyasını sıkıştır npz olarak kaydet
"""
import numpy as np



# 64 bitlik diziyi yükle
data_64 = np.load('irtifa_full.npy', mmap_mode='r')

# Dizinin veri türünü 32 bitlik float türüne dönüştür
data_32 = data_64.astype(np.float32)

# 32 bitlik diziyi kaydet
np.savez_compressed('irtifa_full.npz', data=data_32)
"""

#%%  Sıkıştırılmış npz dosyasını açılmış npy haline getirir
"""
import numpy as np

# İlk önce .npz dosyalarını mmap_mode='r' ile yükleyin
y_npz = np.load("irtifa_full.npz", mmap_mode='r')
X_npz = np.load("x_train_full.npz", mmap_mode='r')

# Dosya boyutunu bölerek küçük parçalar halinde verileri okuyun ve .npy dosyalarına kaydedin
chunk_size = 1000

for i in range(0, len(y_npz['data']), chunk_size):
    y_chunk = y_npz['data'][i:i + chunk_size]
    np.save(f"irtifa_full_{i}-{i + chunk_size}.npy", y_chunk)

for i in range(0, len(X_npz['data']), chunk_size):
    X_chunk = X_npz['data'][i:i + chunk_size]
    np.save(f"x_train_full_{i}-{i + chunk_size}.npy", X_chunk)
    
""" 



#%%

import numpy as np
import os

# İlk önce .npz dosyalarını mmap_mode='r' ile yükleyin
y_npz = np.load("irtifa_full.npz", mmap_mode='r')
X_npz = np.load("x_train_full.npz", mmap_mode='r')

# Verilerinizi kaç parçaya bölmek istediğinizi belirleyin
num_splits = 10

# Verilerinizi parçalara bölün
y_splits = np.array_split(y_npz['data'], num_splits)
X_splits = np.array_split(X_npz['data'], num_splits)

# Parçaları ayrı .npy dosyalarına kaydedin
os.makedirs("y_parts", exist_ok=True)
os.makedirs("X_parts", exist_ok=True)

for i, (y_part, X_part) in enumerate(zip(y_splits, X_splits)):
    np.save(f"y_parts/y_part_{i}.npy", y_part)
    np.save(f"X_parts/X_part_{i}.npy", X_part)


