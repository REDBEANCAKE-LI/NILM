import numpy as np
from scipy import io
import matplotlib.image as mpimg

#constants
IMGNUM = 5  #number of images
IMGSEG = 500    #number of segments for each image
IMGHEIGHT = 736 #height of each image
PATH = 'image_gray_'    #path


#initialize
features = np.mat(np.zeros(IMGHEIGHT, dtype=np.int))    ###!!!pay attention to dtype: np.int
labels = np.mat(np.zeros(1, dtype=np.int))

#add data
for dirIndex in range(1,IMGNUM+1):   ###!!!pay attention to naming rules
    dirCur = PATH + '%d'%dirIndex
    for imgIndex in range(1,IMGSEG+1):  ###!!!pay attention to naming rules
        img = mpimg.imread(dirCur+'\\%d_%d.jpg'%(dirIndex, imgIndex))
        imgMat = np.mat(img).T
        features = np.row_stack((features, imgMat))
        labMat = np.mat(dirIndex)   ###!!! labels: from 1 to IMGNUM
        labels = np.row_stack((labels, labMat))

#delete the first line
features = np.delete(features, 0, 0)
labels = np.delete(labels, 0, 0)


##### test #####
print('features:')
print(features.shape)
print(features)
print()
print('labels:')
print(labels.shape)
print(labels)

#save data as a .mat file
io.savemat('features.mat', {'values':features})
io.savemat('labels.mat', {'values':labels})
print('\n*********************************************************\nEND.')
