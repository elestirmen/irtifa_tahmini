import numpy as np





"""




x_train1=np.load("x_train_negatif_zoom1.npy", mmap_mode='r')
irtifa1=np.load("irtifa_negatif_zoom1.npy", mmap_mode='r')

x_train2=np.load("x_train_negatif_zoom2.npy", mmap_mode='r')
irtifa2=np.load("irtifa_negatif_zoom2.npy", mmap_mode='r')

x_train3=np.load("x_train_negatif_zoom3.npy", mmap_mode='r')
irtifa3=np.load("irtifa_negatif_zoom3.npy", mmap_mode='r')

x_train4=np.load("x_train_negatif_zoom4.npy", mmap_mode='r')
irtifa4=np.load("irtifa_negatif_zoom4.npy", mmap_mode='r')

x_train5=np.load("x_train_negatif_zoom5.npy", mmap_mode='r')
irtifa5=np.load("irtifa_negatif_zoom5.npy", mmap_mode='r')

x_train6=np.load("x_train_negatif_zoom6.npy", mmap_mode='r')
irtifa6=np.load("irtifa_negatif_zoom6.npy", mmap_mode='r')

x_train7=np.load("x_train_negatif_zoom7.npy", mmap_mode='r')
irtifa7=np.load("irtifa_negatif_zoom7.npy", mmap_mode='r')

x_train8=np.load("x_train_negatif_zoom8.npy", mmap_mode='r')
irtifa8=np.load("irtifa_negatif_zoom8.npy", mmap_mode='r')

x_train9=np.load("x_train_negatif_zoom9.npy", mmap_mode='r')
irtifa9=np.load("irtifa_negatif_zoom9.npy", mmap_mode='r')

x_train10=np.load("x_train_negatif_zoom10.npy", mmap_mode='r')
irtifa10=np.load("irtifa_negatif_zoom10.npy", mmap_mode='r')

x_train11=np.load("x_train_negatif_zoom11.npy", mmap_mode='r')
irtifa11=np.load("irtifa_negatif_zoom11.npy", mmap_mode='r')

x_train12=np.load("x_train_negatif_zoom12.npy", mmap_mode='r')
irtifa12=np.load("irtifa_negatif_zoom12.npy", mmap_mode='r')

x_train13=np.load("x_train_negatif_zoom13.npy", mmap_mode='r')
irtifa13=np.load("irtifa_negatif_zoom13.npy", mmap_mode='r')
"""

"""
x_train1=np.memmap("x_train_negatif_zoom1.npy", dtype='float32', mode='r')

x_train2=np.memmap("x_train_negatif_zoom2.npy", dtype='float32', mode='r')

x_train3=np.memmap("x_train_negatif_zoom3.npy", dtype='float32', mode='r')

x_train4=np.memmap("x_train_negatif_zoom4.npy", dtype='float32', mode='r')

x_train5=np.memmap("x_train_negatif_zoom5.npy", dtype='float32', mode='r')

x_train6=np.memmap("x_train_negatif_zoom6.npy", dtype='float32', mode='r')

x_train7=np.memmap("x_train_negatif_zoom7.npy", dtype='float32', mode='r')

x_train8=np.memmap("x_train_negatif_zoom8.npy", dtype='float32', mode='r')

x_train9=np.memmap("x_train_negatif_zoom9.npy", dtype='float32', mode='r')

x_train10=np.memmap("x_train_negatif_zoom10.npy", dtype='float32', mode='r')

x_train11=np.memmap("x_train_negatif_zoom11.npy", dtype='float32', mode='r')

x_train12=np.memmap("x_train_negatif_zoom12.npy", dtype='float32', mode='r')

x_train13=np.memmap("x_train_negatif_zoom13.npy", dtype='float32', mode='r')




x_train1=x_train1[32:].reshape(-1,512,512,1)
x_train2=x_train2[32:].reshape(-1,512,512,1)
x_train3=x_train3[32:].reshape(-1,512,512,1)
x_train4=x_train4[32:].reshape(-1,512,512,1)
x_train5=x_train5[32:].reshape(-1,512,512,1)
x_train6=x_train6[32:].reshape(-1,512,512,1)
x_train7=x_train7[32:].reshape(-1,512,512,1)
x_train8=x_train8[32:].reshape(-1,512,512,1)
x_train9=x_train9[32:].reshape(-1,512,512,1)
x_train10=x_train10[32:].reshape(-1,512,512,1)
x_train11=x_train11[32:].reshape(-1,512,512,1)
x_train12=x_train12[32:].reshape(-1,512,512,1)
x_train13=x_train13[32:].reshape(-1,512,512,1)











x_train= np.concatenate((x_train1, x_train2,x_train3,x_train4,x_train5,x_train6,x_train7,x_train8,x_train9,x_train10,x_train11,x_train12,x_train13))
irtifa = np.concatenate((irtifa1[32:],irtifa2[32:],irtifa3[32:],irtifa4[32:],irtifa5[32:],irtifa6[32:],irtifa7[32:],irtifa8[32:],irtifa9[32:],irtifa10[32:],irtifa11[32:],irtifa12[32:],irtifa13))
print("8")


np.savez('x_train_zoom.npz', x_train)       
np.save('irtifa_zoom.npy', irtifa)
print("9")

"""


#%%


irtifa1=np.load("irtifa_negatif_zoom1.npy", mmap_mode='r')

irtifa2=np.load("irtifa_negatif_zoom2.npy", mmap_mode='r')

irtifa3=np.load("irtifa_negatif_zoom3.npy", mmap_mode='r')

irtifa4=np.load("irtifa_negatif_zoom4.npy", mmap_mode='r')

irtifa5=np.load("irtifa_negatif_zoom5.npy", mmap_mode='r')

irtifa6=np.load("irtifa_negatif_zoom6.npy", mmap_mode='r')

irtifa7=np.load("irtifa_negatif_zoom7.npy", mmap_mode='r')

irtifa8=np.load("irtifa_negatif_zoom8.npy", mmap_mode='r')

irtifa9=np.load("irtifa_negatif_zoom9.npy", mmap_mode='r')

irtifa10=np.load("irtifa_negatif_zoom10.npy", mmap_mode='r')

irtifa11=np.load("irtifa_negatif_zoom11.npy", mmap_mode='r')

irtifa12=np.load("irtifa_negatif_zoom12.npy", mmap_mode='r')

irtifa13=np.load("irtifa_negatif_zoom13.npy", mmap_mode='r')


irtifa_zoom = np.concatenate((irtifa1,irtifa2,irtifa3,irtifa4,irtifa5,irtifa6,irtifa7,irtifa8,irtifa9,irtifa10,irtifa11,irtifa12,irtifa13))

irtifa1=np.load("irtifa.npy", mmap_mode='r')


irtifa = np.concatenate((irtifa1,irtifa_zoom))

np.save('irtifa_full.npy', irtifa)



input("")

import numpy as np

x_train1=np.memmap("x_train.npy", dtype='float32', mode='r')
x_train2=np.memmap("x_train_zoom.npy", dtype='float32', mode='r')
print("1")






irtifa1=np.memmap("irtifa.npy", dtype='float32', mode='r')
irtifa2=np.memmap("irtifa_zoom.npz", dtype='float32', mode='r')
print("2")

irtifa1=irtifa1[32:]

irtifa2=irtifa2[32:]




x_train1=x_train1[32:].reshape(-1,512,512,1)
x_train2=x_train2[32:].reshape(-1,512,512,1)
print("3")

input("pad")
x_train = np.vstack((x_train1, x_train2))
print("4")






print("11")

irtifa1=np.load("irtifa.npy", mmap_mode='r')
irtifa2=np.load("irtifa_zoom.npy")

irtifa = np.vstack((irtifa1,irtifa2))

#np.save('x_train_full.npy', x_train)
print(5)
np.save('irtifa_full.npy', irtifa)
print("12")
