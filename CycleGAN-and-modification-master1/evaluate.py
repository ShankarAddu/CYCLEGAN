# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 09:24:33 2018

@author: djj
"""

from __future__ import print_function
 
import argparse
from datetime import datetime
from random import shuffle
import os
import sys
import time
import math
import tensorflow as tf
import numpy as np
import glob
import cv2
 
from test_image_reader import *
from net import *
 
parser = argparse.ArgumentParser(description='')
 
parser.add_argument("--x_test_data_path", default='/underwater/', help="path of x test datas.")
parser.add_argument("--y_test_data_path", default='/underwater/', help="path of y test datas.")
parser.add_argument("--image_size", type=int, default=256, help="load image size")
parser.add_argument("--snapshots", default='./snapshots/',help="Path of Snapshots")
parser.add_argument("--out_dir_x", default='./test_output_x/',help="Output Folder")
parser.add_argument("--out_dir_y", default='./test_output_y/',help="Output Folder")
 
args = parser.parse_args()
 
def make_test_data_list(x_data_path, y_data_path):
    x_input_images = glob.glob(os.path.join(x_data_path, "*"))
    y_input_images = glob.glob(os.path.join(y_data_path, "*"))
    return x_input_images, y_input_images
 
def cv_inv_proc(img):
    img_rgb = (img + 1.) * 127.5
    return img_rgb.astype(np.float32) #bgr
 
def get_write_picture(x_image, y_image, fake_y, fake_x):
    x_image = cv_inv_proc(x_image)
    y_image = cv_inv_proc(y_image) #还原y域的图像
    fake_y = cv_inv_proc(fake_y[0]) #还原生成的y域的图像
    fake_x = cv_inv_proc(fake_x[0]) #还原生成的x域的图像
    x_output = np.concatenate((x_image, fake_y), axis=1) #得到x域的输入图像以及对应的生成的y域图像
    y_output = np.concatenate((y_image, fake_x), axis=1) #得到y域的输入图像以及对应的生成的x域图像
    return x_output, y_output
 
def main():
    if not os.path.exists(args.out_dir_x):
        os.makedirs(args.out_dir_x)
    if not os.path.exists(args.out_dir_y):
        os.makedirs(args.out_dir_y)
    if os.path.exists(args.out_dir_x):
       tf.compat.v1.disable_eager_execution()
        
 
    x_datalists, y_datalists = make_test_data_list(args.x_test_data_path, args.y_test_data_path)
    test_x_image = tf.compat.v1.placeholder(tf.compat.v1.float32,shape=[1, 256, 256, 3], name = 'test_x_image')
    test_y_image = tf.compat.v1.placeholder(tf.compat.v1.float32,shape=[1, 256, 256, 3], name = 'test_y_image')
 
    fake_y = generator(image=test_x_image, reuse=False, name='generator_x2y')
    fake_x = generator(image=test_y_image, reuse=False, name='generator_y2x')
 
    v1 = tf.compat.v1.placeholder(tf.float32, name="v1")
    v2 = tf.compat.v1.placeholder(tf.float32, name="v2")
    v3 = tf.compat.v1.math.multiply(v1, v2)
    vx = tf.compat.v1.Variable(10.0, name="vx")
    v4 = tf.compat.v1.add(v3, vx, name="v4")
    saver = tf.compat.v1.train.Saver([vx])
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(vx.assign(tf.compat.v1.add(vx, vx)))
    result = sess.run(v4, feed_dict={v1:12.0, v2:3.3})
    print(result)
    saver.save(sess, "./model_ex1")
    saver = tf.train.import_meta_graph("./model_ex1.meta")
    sess = tf.Session()
    saver.restore(sess, "./model_ex1")
    result = sess.run("v4:0", feed_dict={"v1:0": 12.0, "v2:0": 3.3})
    print(result)
 
    restore_var = [v for v in tf.compat.v1.global_variables() if 'generator' in v.name]
 
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    
    saver = tf.compat.v1.train.Saver(var_list=restore_var,max_to_keep=1)
    checkpoint = tf.compat.v1.train.latest_checkpoint(args.snapshots) #读取模型参数
    saver.restore(sess, save_path=checkpoint)
 
    total_step = len(x_datalists) if len(x_datalists) > len(y_datalists) else len(y_datalists)
    for step in range(total_step):
        test_ximage_name, test_ximage = TestImageReader(x_datalists, step, args.image_size)
        test_yimage_name, test_yimage = TestImageReader(y_datalists, step, args.image_size)
        batch_x_image = np.expand_dims(np.array(test_ximage).astype(np.float32), axis = 0)
        batch_y_image = np.expand_dims(np.array(test_yimage).astype(np.float32), axis = 0)
        feed_dict = { test_x_image : batch_x_image, test_y_image : batch_y_image}
        fake_y_value, fake_x_value = sess.run([fake_y, fake_x], feed_dict=feed_dict)
        x_write_image, y_write_image = get_write_picture(test_ximage, test_yimage, fake_y_value, fake_x_value)
        x_write_image_name = args.out_dir_x + "/"+ test_ximage_name + ".png"
        y_write_image_name = args.out_dir_y + "/"+ test_yimage_name + ".png"
        cv2.imwrite(x_write_image_name, x_write_image)
        cv2.imwrite(y_write_image_name, y_write_image)
        print('step {:d}'.format(step))
 
if __name__ == '__main__':
    main()
