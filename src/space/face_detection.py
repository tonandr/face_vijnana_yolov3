'''
MIT License

Copyright (c) 2019 Inwoo Chung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
Created on Apr 5, 2019

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import os
import glob
import argparse
import time
import platform

import numpy as np
import pandas as pd
import cv2 as cv
from skimage.io import imread, imsave

from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Lambda, ZeroPadding2D, LeakyReLU
from keras.layers.merge import add, concatenate
from keras.utils import multi_gpu_model
import keras.backend as K
from keras import optimizers
from keras.utils.data_utils import Sequence

from yolov3_detect import make_yolov3_model, BoundBox, do_nms_v2, WeightReader, draw_boxes_v2, draw_boxes_v3

#os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Constants.
DEBUG = True
MULTI_GPU = False
NUM_GPUS = 4
YOLO3_BASE_MODEL_LOAD_FLAG = True

RATIO_TH = 0.8

class FaceDetector(object):
    """Face detector to use yolov3."""
    # Constants.
    MODEL_PATH = 'face_detector.hd5'
    OUTPUT_FILE_NAME = 'solution.csv'
    EVALUATION_FILE_NAME = 'eval.csv'
    CELL_SIZE = 13

    class TrainingSequence(Sequence):
        """Training data set sequence."""
        
        def __init__(self, raw_data_path, hps, CELL_SIZE, cell_image_size):
            # Get ground truth.
            self.raw_data_path = raw_data_path
            self.hps = hps
            self.gt_df = pd.read_csv(os.path.join(self.raw_data_path, 'training.csv'))
            self.gt_df_g = self.gt_df.groupby('FILE')
            self.file_names = list(self.gt_df_g.groups.keys())
            self.batch_size = self.hps['batch_size'] 
            self.hps['step'] = len(self.file_names) // self.batch_size
            
            if len(self.file_names) % self.batch_size != 0:
                self.hps['step'] +=1
            
            self.CELL_SIZE = CELL_SIZE
            self.cell_image_size = cell_image_size
        
        def __len__(self):
            return self.hps['step']
        
        def __getitem__(self, index):
            images = []
            gt_tensors = []
            
            # Check the last index.
            if index == (self.hps['step'] - 1):
                for bi in range(index * self.batch_size, len(self.file_names)):
                    file_name = self.file_names[bi] 
                    #if DEBUG: print(file_name )
                    
                    df = self.gt_df_g.get_group(file_name)
                    df.index = range(df.shape[0])
                    
                    # Load an image.
                    image = cv.imread(os.path.join(self.raw_data_path, file_name))
                    image = image/255
                                            
                    r = image[:, :, 0].copy()
                    g = image[:, :, 1].copy()
                    b = image[:, :, 2].copy()
                    image[:, :, 0] = b
                    image[:, :, 1] = g
                    image[:, :, 2] = r 
                 
                    # Adjust the original image size into the normalized image size according to the ratio of width, height.
                    w = image.shape[1]
                    h = image.shape[0]
                    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
                                    
                    if w >= h:
                        w_p = self.hps['image_size']
                        h_p = int(h / w * self.hps['image_size'])
                        pad = self.hps['image_size'] - h_p
                        
                        if pad % 2 == 0:
                            pad_t = pad // 2
                            pad_b = pad // 2
                        else:
                            pad_t = pad // 2
                            pad_b = pad // 2 + 1
        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
                    else:
                        h_p = self.hps['image_size']
                        w_p = int(w / h * self.hps['image_size'])
                        pad = self.hps['image_size'] - w_p
                        
                        if pad % 2 == 0:
                            pad_l = pad // 2
                            pad_r = pad // 2
                        else:
                            pad_l = pad // 2
                            pad_r = pad // 2 + 1                
                        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?   
                        
                    # Create a ground truth bound box tensor (13x13x6).
                    gt_tensor = np.zeros(shape=(self.CELL_SIZE, self.CELL_SIZE, self.hps['num_filters']))
                    
                    for i in range(df.shape[0]):
                        # Check exception.
                        res = df.iloc[i, 3:] > 0
                        if res.all() == False:
                            continue
                    
                        # Calculate a target feature tensor according to the ratio of width, height.
                        # Calculate a transformed raw bound box.
                        #print(df.columns)
                        x1 = int(df.loc[i, 'FACE_X'])
                        y1 = int(df.loc[i, 'FACE_Y'])
                        x2 = x1 + int(df.loc[i, 'FACE_WIDTH']) - 1
                        y2 = y1 + int(df.loc[i, 'FACE_HEIGHT']) - 1
                        wb = x2 - x1 + 1
                        hb = y2 - y1 + 1
                        
                        if w >= h:
                            x1_p = int(x1 / w * self.hps['image_size'])
                            y1_p = int(y1 / w * self.hps['image_size']) + pad_t
                            x2_p = int(x2 / w * self.hps['image_size'])
                            y2_p = int(y2 / w * self.hps['image_size']) + pad_t
                        else:
                            x1_p = int(x1 / h * self.hps['image_size']) + pad_l
                            y1_p = int(y1 / h * self.hps['image_size'])
                            x2_p = int(x2 / h * self.hps['image_size']) + pad_l
                            y2_p = int(y2 / h * self.hps['image_size'])                   
                        
                        # Calculate a cell position.
                        xc_p = (x1_p + x2_p) // 2
                        yc_p = (y1_p + y2_p) // 2
                        cx = xc_p // self.cell_image_size
                        cy = yc_p // self.cell_image_size
                        
                        # Calculate a bound box's ratio coordinate.
                        bx_p = (xc_p - cx * self.cell_image_size) / self.cell_image_size
                        by_p = (yc_p - cy * self.cell_image_size) / self.cell_image_size
                        
                        if w >= h: 
                            bw_p = wb / w #?
                            bh_p = hb / w
                        else:
                            bw_p = wb / h
                            bh_p = hb / h
                        
                        # Assign a bound box's values into the tensor.
                        gt_tensor[cy, cx, 0] = 1.
                        gt_tensor[cy, cx, 1] = bx_p
                        gt_tensor[cy, cx, 2] = by_p
                        gt_tensor[cy, cx, 3] = bw_p
                        gt_tensor[cy, cx, 4] = bh_p
                        gt_tensor[cy, cx, 5] = 1.
                        
                    images.append(image)
                    gt_tensors.append(gt_tensor)            
            else:
                for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                    file_name = self.file_names[bi] 
                    #if DEBUG: print(file_name )
                    
                    df = self.gt_df_g.get_group(file_name)
                    df.index = range(df.shape[0])
                    
                    # Load an image.
                    image = cv.imread(os.path.join(self.raw_data_path, file_name))
                    image = image/255
                                            
                    r = image[:, :, 0].copy()
                    g = image[:, :, 1].copy()
                    b = image[:, :, 2].copy()
                    image[:, :, 0] = b
                    image[:, :, 1] = g
                    image[:, :, 2] = r 
                 
                    # Adjust the original image size into the normalized image size according to the ratio of width, height.
                    w = image.shape[1]
                    h = image.shape[0]
                    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
                                    
                    if w >= h:
                        w_p = self.hps['image_size']
                        h_p = int(h / w * self.hps['image_size'])
                        pad = self.hps['image_size'] - h_p
                        
                        if pad % 2 == 0:
                            pad_t = pad // 2
                            pad_b = pad // 2
                        else:
                            pad_t = pad // 2
                            pad_b = pad // 2 + 1
        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
                    else:
                        h_p = self.hps['image_size']
                        w_p = int(w / h * self.hps['image_size'])
                        pad = self.hps['image_size'] - w_p
                        
                        if pad % 2 == 0:
                            pad_l = pad // 2
                            pad_r = pad // 2
                        else:
                            pad_l = pad // 2
                            pad_r = pad // 2 + 1                
                        
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?   
                        
                    # Create a ground truth bound box tensor (13x13x6).
                    gt_tensor = np.zeros(shape=(self.CELL_SIZE, self.CELL_SIZE, self.hps['num_filters']))
                    
                    for i in range(df.shape[0]):
                        # Check exception.
                        res = df.iloc[i, 3:] > 0
                        if res.all() == False:
                            continue
                    
                        # Calculate a target feature tensor according to the ratio of width, height.
                        # Calculate a transformed raw bound box.
                        #print(df.columns)
                        x1 = int(df.loc[i, 'FACE_X'])
                        y1 = int(df.loc[i, 'FACE_Y'])
                        x2 = x1 + int(df.loc[i, 'FACE_WIDTH']) - 1
                        y2 = y1 + int(df.loc[i, 'FACE_HEIGHT']) - 1
                        wb = x2 - x1 + 1
                        hb = y2 - y1 + 1
                        
                        if w >= h:
                            x1_p = int(x1 / w * self.hps['image_size'])
                            y1_p = int(y1 / w * self.hps['image_size']) + pad_t
                            x2_p = int(x2 / w * self.hps['image_size'])
                            y2_p = int(y2 / w * self.hps['image_size']) + pad_t
                        else:
                            x1_p = int(x1 / h * self.hps['image_size']) + pad_l
                            y1_p = int(y1 / h * self.hps['image_size'])
                            x2_p = int(x2 / h * self.hps['image_size']) + pad_l
                            y2_p = int(y2 / h * self.hps['image_size'])                   
                        
                        # Calculate a cell position.
                        xc_p = (x1_p + x2_p) // 2
                        yc_p = (y1_p + y2_p) // 2
                        cx = xc_p // self.cell_image_size
                        cy = yc_p // self.cell_image_size
                        
                        # Calculate a bound box's ratio coordinate.
                        bx_p = (xc_p - cx * self.cell_image_size) / self.cell_image_size
                        by_p = (yc_p - cy * self.cell_image_size) / self.cell_image_size
                        
                        if w >= h: 
                            bw_p = wb / w #?
                            bh_p = hb / w
                        else:
                            bw_p = wb / h
                            bh_p = hb / h
                        
                        # Assign a bound box's values into the tensor.
                        gt_tensor[cy, cx, 0] = 1.
                        gt_tensor[cy, cx, 1] = bx_p
                        gt_tensor[cy, cx, 2] = by_p
                        gt_tensor[cy, cx, 3] = bw_p
                        gt_tensor[cy, cx, 4] = bh_p
                        gt_tensor[cy, cx, 5] = 1.
                        
                    images.append(image)
                    gt_tensors.append(gt_tensor)
                                                                         
            return ({'input': np.asarray(images)}, {'output': np.asarray(gt_tensors)})   

    def __init__(self, raw_data_path, hps, model_loading):
        """
        Parameters
        ----------
        raw_data_path : string
            Raw data path
        hps : dictionary
            Hyper-parameters
        model_loading : boolean 
            Face detection model loading flag
        """
        # Initialize.
        self.raw_data_path = raw_data_path
        self.hps = hps
        self.model_loading = model_loading
        self.cell_image_size = hps['image_size'] // self.CELL_SIZE # ?
        
        if model_loading: 
            if MULTI_GPU:
                self.model = load_model(self.MODEL_PATH)
                self.parallel_model = multi_gpu_model(self.model, gpus = NUM_GPUS)
                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay']) 
                self.parallel_model.compile(optimizer=opt, loss='mse') 
            else:
                self.model = load_model(self.MODEL_PATH)
        else:
            # Design the face detector model.
            # Input.
            input1 = Input(shape=(hps['image_size'], hps['image_size'], 3), name='input1')

            # Load yolov3 as the base model.
            base = self.YOLOV3Base 
            
            # Get 13x13x6 target features. #?
            x = base(input1) # Non-linear.
            output = Conv2D(filters=hps['num_filters']
                       , activation='linear'
                       , kernel_size=(3, 3)
                       , padding='same'
                       , name='output')(x)

            if MULTI_GPU:
                self.model = Model(inputs=[input1], outputs=[output])

                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay'])
                
                self.model.compile(optimizer=opt, loss='mse')
                self.model.summary()
                
                self.parallel_model = multi_gpu_model(self.model, gpus = NUM_GPUS)
                self.parallel_model.compile(optimizer=opt, loss='mse')
                self.parallel_model.summary()
                
            else:
                self.model = Model(inputs=[input1], outputs=[output])

                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay'])
                
                self.model.compile(optimizer=opt, loss='mse')
                self.model.summary()

    @property
    def YOLOV3Base(self):
        """Get yolov3 as a base model.
        
        Returns
        -------
        Model of Keras
            Partial yolo3 model from the input layer to the add_23 layer
        """
        if YOLO3_BASE_MODEL_LOAD_FLAG:
            base = load_model('yolov3_base.hd5')
            base.trainable = True
            return base

        yolov3 = make_yolov3_model()

        # Load the weights.
        weight_reader = WeightReader('yolov3.weights')
        weight_reader.load_weights(yolov3)
        
        # Make a base model.
        input1 = Input(shape=(self.hps['image_size'], self.hps['image_size'], 3))
        
        # 0 ~ 1.
        conv_layer = yolov3.get_layer('conv_' + str(0))
        x = ZeroPadding2D(1)(input1) #?               
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(0))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        conv_layer = yolov3.get_layer('conv_' + str(1))
        x = ZeroPadding2D(1)(x) #?               
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(1))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)
        skip = x
        
        # 2 ~ 3.
        for i in range(2, 4, 2):
            conv_layer = yolov3.get_layer('conv_' + str(i))
            
            if conv_layer.kernel_size[0] > 1:
                x = ZeroPadding2D(1)(x) #? 
              
            x = conv_layer(x)
            norm_layer = yolov3.get_layer('bnorm_' + str(i))
            x = norm_layer(x)
            x = LeakyReLU(alpha=0.1)(x)

            conv_layer = yolov3.get_layer('conv_' + str(i + 1))
            
            if conv_layer.kernel_size[0] > 1:
                x = ZeroPadding2D(1)(x) #? 
              
            x = conv_layer(x)
            norm_layer = yolov3.get_layer('bnorm_' + str(i + 1))
            x = norm_layer(x)
            x = LeakyReLU(alpha=0.1)(x)
            
            x = add([skip, x]) #?

        # 5.
        conv_layer = yolov3.get_layer('conv_' + str(5))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(5))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)
        skip = x
        
        # 6 ~ 10.
        for i in range(6, 10, 3):
            conv_layer = yolov3.get_layer('conv_' + str(i))
            
            if conv_layer.kernel_size[0] > 1:
                x = ZeroPadding2D(1)(x) #? 
              
            x = conv_layer(x)
            norm_layer = yolov3.get_layer('bnorm_' + str(i))
            x = norm_layer(x)
            x = LeakyReLU(alpha=0.1)(x)

            conv_layer = yolov3.get_layer('conv_' + str(i + 1))
            
            if conv_layer.kernel_size[0] > 1:
                x = ZeroPadding2D(1)(x) #? 
              
            x = conv_layer(x)
            norm_layer = yolov3.get_layer('bnorm_' + str(i + 1))
            x = norm_layer(x)
            x = LeakyReLU(alpha=0.1)(x)
            
            x = add([skip, x]) #?
            skip = x #?

        # 12.
        conv_layer = yolov3.get_layer('conv_' + str(12))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(12))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)
        skip = x

        # 13 ~ 35.
        for i in range(13, 35, 3):
            conv_layer = yolov3.get_layer('conv_' + str(i))
            
            if conv_layer.kernel_size[0] > 1:
                x = ZeroPadding2D(1)(x) #? 
              
            x = conv_layer(x)
            norm_layer = yolov3.get_layer('bnorm_' + str(i))
            x = norm_layer(x)
            x = LeakyReLU(alpha=0.1)(x)

            conv_layer = yolov3.get_layer('conv_' + str(i + 1))
            
            if conv_layer.kernel_size[0] > 1:
                x = ZeroPadding2D(1)(x) #? 
              
            x = conv_layer(x)
            norm_layer = yolov3.get_layer('bnorm_' + str(i + 1))
            x = norm_layer(x)
            x = LeakyReLU(alpha=0.1)(x)
            
            x = add([skip, x]) #?
            skip = x #?

        # 37.
        conv_layer = yolov3.get_layer('conv_' + str(37))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(37))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)
        skip = x

        # 38 ~ 60.
        for i in range(38, 60, 3):
            conv_layer = yolov3.get_layer('conv_' + str(i))
            
            if conv_layer.kernel_size[0] > 1:
                x = ZeroPadding2D(1)(x) #? 
              
            x = conv_layer(x)
            norm_layer = yolov3.get_layer('bnorm_' + str(i))
            x = norm_layer(x)
            x = LeakyReLU(alpha=0.1)(x)

            conv_layer = yolov3.get_layer('conv_' + str(i + 1))
            
            if conv_layer.kernel_size[0] > 1:
                x = ZeroPadding2D(1)(x) #? 
              
            x = conv_layer(x)
            norm_layer = yolov3.get_layer('bnorm_' + str(i + 1))
            x = norm_layer(x)
            x = LeakyReLU(alpha=0.1)(x)
            
            x = add([skip, x]) #?
            skip = x #?

        # 62.
        conv_layer = yolov3.get_layer('conv_' + str(62))
        
        if conv_layer.kernel_size[0] > 1:
            x = ZeroPadding2D(1)(x) #? 
          
        x = conv_layer(x)
        norm_layer = yolov3.get_layer('bnorm_' + str(62))
        x = norm_layer(x)
        x = LeakyReLU(alpha=0.1)(x)
        skip = x

        # 63 ~ 73.
        for i in range(63, 73, 3):
            conv_layer = yolov3.get_layer('conv_' + str(i))
            
            if conv_layer.kernel_size[0] > 1:
                x = ZeroPadding2D(1)(x) #? 
              
            x = conv_layer(x)
            norm_layer = yolov3.get_layer('bnorm_' + str(i))
            x = norm_layer(x)
            x = LeakyReLU(alpha=0.1)(x)

            conv_layer = yolov3.get_layer('conv_' + str(i + 1))
            
            if conv_layer.kernel_size[0] > 1:
                x = ZeroPadding2D(1)(x) #? 
              
            x = conv_layer(x)
            norm_layer = yolov3.get_layer('bnorm_' + str(i + 1))
            x = norm_layer(x)
            x = LeakyReLU(alpha=0.1)(x)
            
            x = add([skip, x]) #?
            skip = x #?
        
        output = x
        base = Model(inputs=[input], outputs=[output])
        base.trainable = True
        base.save('yolov3_base.hd5')
        
        return base

    def train(self):
        """Train face detector."""
        # Get the generator of training data.
        #tr_gen = self._get_training_generator()
        tr_gen = self.TrainingSequence(self.raw_data_path, self.hps, self.CELL_SIZE, self.cell_image_size)
        
        if MULTI_GPU:
            self.parallel_model.fit_generator(tr_gen
                      , steps_per_epoch=self.hps['step']                  
                      , epochs=self.hps['epochs']
                      , verbose=1
                      , max_queue_size=100
                      , workers=8
                      , use_multiprocessing=True)
        else:
            self.model.fit_generator(tr_gen
                      , steps_per_epoch=self.hps['step']                  
                      , epochs=self.hps['epochs']
                      , verbose=1
                      , max_queue_size=100
                      , workers=4
                      , use_multiprocessing=True)
        
        print('Save the model.')            
        self.model.save(self.MODEL_PATH)
    
    def evaluate(self, test_path, output_file_path):
        """Evaluate.
        
        Parameters
        ----------
        test_path : string
            Testing directory.
        output_file_path : string
            Output file path.
        """
        gt_df = pd.read_csv(os.path.join(self.raw_data_path, 'training.csv'))
        gt_df_g = gt_df.groupby('FILE')        
        file_names = glob.glob(os.path.join(test_path, '*.jpg'))
        ratios = []
                
        # Detect faces and save results.
        count1 = 1
        with open(output_file_path, 'w') as f:
            for file_name in file_names:          
                if DEBUG: print(count1, '/', len(file_names), file_name)
                count1 += 1
                
                # Load an image.
                image = cv.imread(os.path.join(test_path, file_name))
                image_o_size = (image.shape[0], image.shape[1])
                image_o = image.copy() 
                image = image/255
                                    
                r = image[:, :, 0].copy()
                g = image[:, :, 1].copy()
                b = image[:, :, 2].copy()
                image[:, :, 0] = b
                image[:, :, 1] = g
                image[:, :, 2] = r 
             
                # Adjust the original image size into the normalized image size according to the ratio of width, height.
                w = image.shape[1]
                h = image.shape[0]
                pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
                                
                if w >= h:
                    w_p = self.hps['image_size']
                    h_p = int(h / w * self.hps['image_size'])
                    pad = self.hps['image_size'] - h_p
                    
                    if pad % 2 == 0:
                        pad_t = pad // 2
                        pad_b = pad // 2
                    else:
                        pad_t = pad // 2
                        pad_b = pad // 2 + 1
    
                    image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                    image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
                else:
                    h_p = self.hps['image_size']
                    w_p = int(w / h * self.hps['image_size'])
                    pad = self.hps['image_size'] - w_p
                    
                    if pad % 2 == 0:
                        pad_l = pad // 2
                        pad_r = pad // 2
                    else:
                        pad_l = pad // 2
                        pad_r = pad // 2 + 1                
                    
                    image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                    image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
       
                image = image[np.newaxis, :]
                       
                # Detect faces.
                boxes = self.detect(image, image_o_size)
                
                # correct the sizes of the bounding boxes
                for box in boxes:
                    if w >= h:
                        box.xmin = np.min([box.xmin * w / self.hps['image_size'], w])
                        box.xmax = np.min([box.xmax * w / self.hps['image_size'], w])
                        box.ymin = np.min([np.max([box.ymin - pad_t, 0]) * w / self.hps['image_size'], h])
                        box.ymax = np.min([np.max([box.ymax - pad_t, 0]) * w / self.hps['image_size'], h])
                    else:
                        box.xmin = np.min([np.max([box.xmin - pad_l, 0]) * h / self.hps['image_size'], w])
                        box.xmax = np.min([np.max([box.xmax - pad_l, 0]) * h / self.hps['image_size'], w])
                        box.ymin = np.min([box.ymin * h / self.hps['image_size'], h])
                        box.ymax = np.min([box.ymax * h / self.hps['image_size'], h])

                    # Correct image ratio.
                    r_wh = (box.xmax - box.xmin) / (box.ymax - box.ymin)
                    r_hw = (box.ymax - box.ymin) / (box.xmax - box.xmin)
                    
                    if r_wh < RATIO_TH:
                        box.xmax = RATIO_TH * (box.ymax - box.ymin) + box.xmin
                    elif r_hw < RATIO_TH:
                        box.ymax = RATIO_TH * (box.xmax - box.xmin) + box.ymin
                    
                    # Scale ?
                    # TODO
                        
                count = 1
                
                for box in boxes:
                    if count > 60:
                        break
                    
                    f.write(file_name.split('/')[-1] + ',' + str(box.xmin) + ',' + str(box.ymin) + ',')
                    f.write(str(box.xmax - box.xmin) + ',' + str(box.ymax - box.ymin) + ',' + str(box.get_score()) + '\n')
                    count +=1

                # Check exception.
                if len(boxes) == 0:
                    continue

                # Draw bounding boxes of ground truth.
                if platform.system() == 'Windows':
                    file_new_name = file_name.split('\\')[-1]
                else:
                    file_new_name = file_name.split('/')[-1]
                
                df = gt_df_g.get_group(file_new_name)
                gt_boxes = []
                
                for i in range(df.shape[0]):
                    # Check exception.
                    res = df.iloc[i, 3:] > 0
                    if res.all() == False:
                        continue
                    
                    xmin = int(df.iloc[i, 3])
                    xmax = int(xmin + df.iloc[i, 5] - 1)
                    ymin = int(df.iloc[i, 4])
                    ymax = int(ymin + df.iloc[i, 6] - 1)                    
                    gt_box = BoundBox(xmin, ymin, xmax, ymax, objness=1., classes=[1.0])
                    gt_boxes.append(gt_box)
                    ratios.append((xmax - xmin) / (ymax - ymin))
                    
                image = draw_boxes_v3(image_o, gt_boxes, self.hps['face_conf_th']) 
                
                # Draw bounding boxes on the image using labels.
                image = draw_boxes_v2(image, boxes, self.hps['face_conf_th']) 
         
                # Write the image with bounding boxes to file.
                file_new_name = file_new_name[:-4] + '_detected' + file_new_name[-4:]
                
                print(file_new_name)
                imsave(os.path.join(test_path, 'results', file_new_name), (image).astype('uint8'))                
        
        ratios = pd.DataFrame({'ratio': ratios})
        ratios.to_csv('ratios.csv')
        
    def test(self, test_path, output_file_path):
        """Test.
        
        Parameters
        ----------
        test_path : string
            Testing directory.
        output_file_path : string
            Output file path.
        """
        file_names = glob.glob(os.path.join(test_path, '*.jpg'))
                
        # Detect faces and save results.
        count1 = 1
        with open(output_file_path, 'w') as f:
            for file_name in file_names:
                if DEBUG: print(count1, '/', len(file_names), file_name)
                count1 += 1
                
                # Load an image.
                image = cv.imread(os.path.join(test_path, file_name))
                image = image/255
                image_o = image.copy() 
                
                r = image[:, :, 0].copy()
                g = image[:, :, 1].copy()
                b = image[:, :, 2].copy()
                image[:, :, 0] = b
                image[:, :, 1] = g
                image[:, :, 2] = r 
             
                # Adjust the original image size into the normalized image size according to the ratio of width, height.
                w = image.shape[1]
                h = image.shape[0]
                pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
                                
                if w >= h:
                    w_p = self.hps['image_size']
                    h_p = int(h / w * self.hps['image_size'])
                    pad = self.hps['image_size'] - h_p
                    
                    if pad % 2 == 0:
                        pad_t = pad // 2
                        pad_b = pad // 2
                    else:
                        pad_t = pad // 2
                        pad_b = pad // 2 + 1
    
                    image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                    image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
                else:
                    h_p = self.hps['image_size']
                    w_p = int(w / h * self.hps['image_size'])
                    pad = self.hps['image_size'] - w_p
                    
                    if pad % 2 == 0:
                        pad_l = pad // 2
                        pad_r = pad // 2
                    else:
                        pad_l = pad // 2
                        pad_r = pad // 2 + 1                
                    
                    image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                    image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
       
                image = image[np.newaxis, :]
                       
                # Detect faces.
                boxes = self.detect(image)
                
                # Correct the sizes of the bounding boxes
                for box in boxes:
                    if w >= h:
                        box.xmin = np.min([box.xmin * w / self.hps['image_size'], w])
                        box.xmax = np.min([box.xmax * w / self.hps['image_size'], w])
                        box.ymin = np.min([np.max([box.ymin - pad_t, 0]) * w / self.hps['image_size'], h])
                        box.ymax = np.min([np.max([box.ymax - pad_t, 0]) * w / self.hps['image_size'], h])
                    else:
                        box.xmin = np.min([np.max([box.xmin - pad_l, 0]) * h / self.hps['image_size'], w])
                        box.xmax = np.min([np.max([box.xmax - pad_l, 0]) * h / self.hps['image_size'], w])
                        box.ymin = np.min([box.ymin * h / self.hps['image_size'], h])
                        box.ymax = np.min([box.ymax * h / self.hps['image_size'], h])
                    
                    # Correct image ratio.
                    r_wh = (box.xmax - box.xmin) / (box.ymax - box.ymin)
                    r_hw = (box.ymax - box.ymin) / (box.xmax - box.xmin)
                    
                    if r_wh < RATIO_TH:
                        box.xmax = RATIO_TH * (box.ymax - box.ymin) + box.xmin
                    elif r_hw < RATIO_TH:
                        box.ymax = RATIO_TH * (box.xmax - box.xmin) + box.ymin
                    
                    # Scale ?
                    # TODO
                    
                count = 1
                
                for box in boxes:
                    if count > 60:
                        break
                    
                    f.write(file_name.split('/')[-1] + ',' + str(box.xmin) + ',' + str(box.ymin) + ',')
                    f.write(str(box.xmax - box.xmin) + ',' + str(box.ymax - box.ymin) + ',' + str(box.get_score()) + '\n')
                    count +=1

                # Check exception.
                if len(boxes) == 0:
                    continue

                # Write the image with bounding boxes to file.
                '''
                if platform.system() == 'Windows':
                    file_new_name = file_name.split('\\')[-1]
                    file_new_name = file_name[:-4] + '_detected' + file_name[-4:]
                
                print(file_new_name)
                imsave(os.path.join(test_path, 'results', file_new_name), (image).astype('uint8'))
                '''
                                                                    
    def detect(self, image):
        """Detect faces.
        
        Parameters
        ----------
        images: 4d numpy array
            Images
                    
        Returns
        -------
        list
            Face candidate bound boxes
        """
        # Get face region candidates.
        face_cands = self.model.predict(image) # 1x13x13x6. #?
        face_cands = np.squeeze(face_cands)
        
        # Eliminate candidates less than the face confidence threshold.
        face_cand_bboxes = []
        face_cands[...,-1] = face_cands[...,0] * face_cands[...,-1] #?

        for i in range(face_cands.shape[0]):
            for j in range(face_cands.shape[1]):
                if face_cands[i, j, 0] > 0. and  face_cands[i, j, -1] >= self.hps['face_conf_th']:
                    # Calculate values of a bound box.
                    objectness = face_cands[i, j, 0] # > 0.
                    bx = np.max([face_cands[i, j, 1], 0.])
                    by = np.max([face_cands[i, j, 2], 0.])
                    bw = np.max([face_cands[i, j, 3], 0.])
                    bh = np.max([face_cands[i, j, 4], 0.])
                    score = face_cands[i, j, 5] # > 0.
                                                            
                    # Convert a raw ratio bound box into a bound box for practical image size scale.  
                    px = np.min([int(bx * self.cell_image_size), self.cell_image_size - 1]) + self.cell_image_size * j
                    py = np.min([int(by * self.cell_image_size), self.cell_image_size - 1]) + self.cell_image_size * i
                    pw = np.min([bw * self.hps['image_size'], self.hps['image_size']]) 
                    ph = np.min([bh * self.hps['image_size'], self.hps['image_size']])
                    
                    # Calculate xmin, ymin, xmax, ymax positions for the original image size.
                    xmin = np.max([px - int(pw / 2), 0])
                    ymin = np.max([py - int(ph / 2), 0])
                    xmax = np.min([px + int(pw / 2), self.hps['image_size'] - 1])
                    ymax = np.min([py + int(ph / 2), self.hps['image_size'] - 1])
                    
                    # Get a bound box.
                    face_cand_bbox = BoundBox(xmin, ymin, xmax, ymax, objness = objectness, classes = [score])
                    face_cand_bboxes.append(face_cand_bbox)
        
        # Check exception.
        if len(face_cand_bboxes) == 0:
            return face_cand_bboxes
        
        # Conduct non-max suppression.
        do_nms_v2(face_cand_bboxes, self.hps['nms_iou_th'])
        
        # Get high face score candidates > 0.
        face_cand_bboxes = [face_cand_bbox for face_cand_bbox in face_cand_bboxes if face_cand_bbox.get_score() > 0.]
        scores = [face_cand_bbox.get_score() for face_cand_bbox in face_cand_bboxes]
        sorted_index = np.argsort(scores)
        
        face_cand_bboxes = [face_cand_bboxes[sorted_index[i]] \
                       for i in range(self.hps['num_cands']) if i < len(scores)]
        
        return face_cand_bboxes
                 
def main(args):
    """Main.
    
    Parameters
    ----------
    args : argument type 
        Arguments
    """
    hps = {}
    hps['step'] = 1
            
    if args.mode == 'train':
        # Get arguments.
        raw_data_path = args.raw_data_path
      
        # hps.
        hps['image_size'] = int(args.image_size)    
        hps['num_filters'] = int(args.num_filters)
        hps['lr'] = float(args.lr)
        hps['beta_1'] = float(args.beta_1)
        hps['beta_2'] = float(args.beta_2)
        hps['decay'] = float(args.decay)
        hps['batch_size'] = int(args.batch_size)
        hps['epochs'] = int(args.epochs) 
        hps['face_conf_th'] = float(args.face_conf_th)
        hps['nms_iou_th'] = float(args.nms_iou_th)
        hps['num_cands'] = int(args.num_cands)
        
        model_loading = False if int(args.model_loading) == 0 else True        
        
        # Train.
        fd = FaceDetector(raw_data_path, hps, model_loading)
        
        ts = time.time()
        fd.train()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif args.mode == 'evaluate':
        # Get arguments.
        raw_data_path = args.raw_data_path
        output_file_path = args.output_file_path
      
        # hps.
        hps['image_size'] = int(args.image_size) 
        hps['num_filters'] = int(args.num_filters)
        hps['lr'] = float(args.lr)
        hps['beta_1'] = float(args.beta_1)
        hps['beta_2'] = float(args.beta_2)
        hps['decay'] = float(args.decay)
        hps['batch_size'] = int(args.batch_size)
        hps['epochs'] = int(args.epochs) 
        hps['face_conf_th'] = float(args.face_conf_th)
        hps['nms_iou_th'] = float(args.nms_iou_th)
        hps['num_cands'] = int(args.num_cands)
        
        model_loading = False if int(args.model_loading) == 0 else True        
        
        fd = FaceDetector(raw_data_path, hps, True)
        
        ts = time.time()
        fd.evaluate(raw_data_path, output_file_path)
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif args.mode == 'test':
        # Get arguments.
        raw_data_path = args.raw_data_path
        output_file_path = args.output_file_path
      
        # hps.
        hps['image_size'] = int(args.image_size) 
        hps['num_filters'] = int(args.num_filters)
        hps['lr'] = float(args.lr)
        hps['beta_1'] = float(args.beta_1)
        hps['beta_2'] = float(args.beta_2)
        hps['decay'] = float(args.decay)
        hps['batch_size'] = int(args.batch_size)
        hps['epochs'] = int(args.epochs) 
        hps['face_conf_th'] = float(args.face_conf_th)
        hps['nms_iou_th'] = float(args.nms_iou_th)
        hps['num_cands'] = int(args.num_cands)
        
        model_loading = False if int(args.model_loading) == 0 else True        
        
        # Test.
        fd = FaceDetector(raw_data_path, hps, True)
        
        ts = time.time()
        fd.test(raw_data_path, output_file_path)
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
        
if __name__ == '__main__':
    
    # Parse arguments.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode')
    parser.add_argument('--raw_data_path')
    parser.add_argument('--output_file_path')
    parser.add_argument('--image_size')
    parser.add_argument('--num_filters')
    parser.add_argument('--lr')
    parser.add_argument('--beta_1')
    parser.add_argument('--beta_2')
    parser.add_argument('--decay')
    parser.add_argument('--batch_size')
    parser.add_argument('--epochs')
    parser.add_argument('--face_conf_th')
    parser.add_argument('--nms_iou_th')
    parser.add_argument('--num_cands')
    parser.add_argument('--model_loading')
    args = parser.parse_args()
    
    main(args)
    pass