from PIL import Image
import os

#constants
IMGNUM = 5  #number of images
IMSEG = 500 #number of segments for each image
PATH = 'image_' #path

for dirIndex in range(1,IMGNUM+1):  ###!!!pay attention to naming rules
    #create new directory
    dirCur = PATH + '%d'%dirIndex
    dirCur_new = dirCur + '_gray'
    os.mkdir(dirCur_new)
    
    #open images, convert rgb to gray and resave
    for imgIndex in range(1,IMSEG+1):   ###!!!pay attention to naming rules
        imgName = '%d_%d.jpg'%(dirIndex, imgIndex)
        img = Image.open(dirCur+'\\'+imgName).convert('L')
        img.save(dirCur_new+'\\'+imgName)

print('END.')
