import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
image_shape = (160, 576)
image_paths = glob("./runs/1544681386.8921719/")
#print(image_paths)
label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join("../data_road/training/", 'gt_image_2', '*_road_*.png'))}
#print(label_paths)
background_color = np.array([255, 0, 0])
# Loop through batches and grab images, yielding each batch
tp = 0
tn = 0
fp = 0
fn = 0
for key,value in label_paths.items():
        
        gt_image_file = value
        image_file = image_paths[0]+key
        # Re-size to image_shape
        try:
                
            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
            gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)



            # Create "one-hot-like" labels by class
            gt_bg = np.all(gt_image == background_color, axis=2)
            gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
            gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
            image = np.array(image)
            gt_image = np.array(gt_image)


            print(image[:5,:5,0])
            print(gt_image[:5,:5,0])
            #for i in range(image.shape[0]):
            #    for j in range(image.shape[1]):
            #        tp= 

        except:
            print("not there")




