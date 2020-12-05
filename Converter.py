import numpy as np
import scipy.io
import os

a = np.array([[1,2,3],[4,5,6]]);
print(np.amax(a))

# print(max(max(a)))
# df = scipy.io.loadmat('C:/Users/ibrahim/Desktop/DataSets/braintumordataset/AllMasksMatFiles/image1.mat', mdict=None, appendmat=True)['img']
# df = np.array(df)
# print(df[200,:])
# print(df.shape)
# np.save('C:/Users/ibrahim/Desktop/DataSets/braintumordataset/AllImagesMatFiles/my', df)



# dataDir = "C:/Users/ibrahim/Desktop/DataSets/braintumordataset/AllMasksMatFiles/"
# mats = []
# num = 1
# arr = scipy.io.loadmat( dataDir+"mask1.mat",mdict=None,appendmat=True)['mask']
# arr = np.array(arr)
# print(arr[200,:])

# for file in os.listdir( dataDir ) :
#     arr = scipy.io.loadmat( dataDir+file,mdict=None,appendmat=True)['mask']
#     df = np.array(arr)
#     name = "C:/Users/ibrahim/Desktop/DataSets/braintumordataset/MasksNumpy/"+'mask'+str(num)
#     np.save(name,df)
#     num += 1




