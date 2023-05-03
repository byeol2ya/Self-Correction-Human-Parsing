# name = 'straight_pants_ABCDLKEJK.png'
# gt = torch.zeros((20,256,256), dtype=torch.float32)
# IsPants = (name.find('straight_pants') != -1)
# B, G, R = 0, 1, 2
# if IsPants is True:
#      gt[5] = 1 - cv.imread(name)[:,:,G]


# make original gt to fit with schp channels
import os
import numpy as np
# imoprt glob
import cv2

file_path = ('C:/Users/Choi Byeoli/Downloads/check4')
file_names = os.listdir(file_path)

for i in file_names:
    src = cv2.imread(file_path + '/' + i)


