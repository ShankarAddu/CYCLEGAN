# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 09:24:49 2018

@author: djj
"""

import os
 
import numpy as np
import tensorflow as tf
import cv2
 
def TestImageReader(file_list, step, size):
    file_length = len(file_list)
    line_idx = step % file_length
    test_line_content = file_list[line_idx]
    test_image_name, _ = os.path.splitext(os.path.basename(test_line_content))
    test_image = cv2.imread(test_line_content, 1)
    test_image_resize_t = cv2.resize(test_image, (size, size))
    test_image_resize = test_image_resize_t/127.5-1
    return test_image_name, test_image_resize
