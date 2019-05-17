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
Created on Apr 9, 2019

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import os
import glob
import argparse
import time
import pickle
import platform
import shutil

import numpy as np
import pandas as pd
import cv2 as cv
from skimage.io import imread, imsave
from scipy.stats import entropy
from scipy.linalg import norm

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, ZeroPadding2D, LeakyReLU, Flatten, Concatenate
from keras.layers.merge import add
from keras.utils import multi_gpu_model
from keras.utils.data_utils import Sequence
import keras.backend as K
from keras import optimizers

from yolov3_detect import make_yolov3_model, BoundBox, do_nms_v2, WeightReader, draw_boxes_v2, draw_boxes_v3
from face_detection import FaceDetector
from random import shuffle

# Constants.
DEBUG = True
MULTI_GPU = True
NUM_GPUS = 4
YOLO3_BASE_MODEL_LOAD_FLAG = False
ALPHA = 0.2

def triplet_loss(y_true, y_pred):
    # Calculate the difference of both face features and judge a same person.
    x = y_pred
    return K.mean(K.maximum(K.sqrt(K.sum(K.pow(x[:, 0:64] - x[:, 64:128], 2.0), axis=-1)) \
                     - K.sqrt(K.sum(K.pow(x[:, 0:64] - x[:, 128:192], 2.0), axis=-1)) + ALPHA, 0.))
                         
def create_db_fri(raw_data_path, hps):
    """Create db for face re-identifier."""
    if not os.path.isdir(os.path.join('subject_faces')):
        os.mkdir(os.path.join('subject_faces'))
    else:
        shutil.rmtree(os.path.join('subject_faces'))
        os.mkdir(os.path.join(os.path.join('subject_faces')))    

    gt_df = pd.read_csv(os.path.join(raw_data_path, 'training', 'training.csv'))
    gt_df_g = gt_df.groupby('SUBJECT_ID')
    
    # Collect face region images and create db, by subject ids.
    db = pd.DataFrame(columns=['subject_id', 'face_file', 'w', 'h'])
    
    for k in gt_df_g.groups.keys():
        if k == -1: continue
        df = gt_df_g.get_group(k)
        
        for i in range(df.shape[0]):
            file_name = df.iloc[i, 1]
            
            # Load an image.
            image = cv.imread(os.path.join(raw_data_path, 'training', file_name))
            
            # Check exception.
            res = df.iloc[i, 3:] > 0
            if res.all() == False:
                continue
            
            r = image[:, :, 0].copy()
            g = image[:, :, 1].copy()
            b = image[:, :, 2].copy()
            image[:, :, 0] = b
            image[:, :, 1] = g
            image[:, :, 2] = r
                        
            # Crop a face region.
            l, t, r, b = (int(df.iloc[i, 3])
                , int(df.iloc[i, 4])
                , int((df.iloc[i, 3] + df.iloc[i, 5] - 1))
                , int((df.iloc[i, 4] + df.iloc[i, 6] - 1)))

                    
            image = image[(t - 1):(b - 1), (l - 1):(r - 1), :]
            
            # Adjust the original image size into the normalized image size according to the ratio of width, height.
            w = image.shape[1]
            h = image.shape[0]
            pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
                            
            if w >= h:
                w_p = hps['image_size']
                h_p = int(h / w * hps['image_size'])
                pad = hps['image_size'] - h_p
                
                if pad % 2 == 0:
                    pad_t = pad // 2
                    pad_b = pad // 2
                else:
                    pad_t = pad // 2
                    pad_b = pad // 2 + 1
                 
                image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
            else:
                h_p = hps['image_size']
                w_p = int(w / h * hps['image_size'])
                pad = hps['image_size'] - w_p
                
                if pad % 2 == 0:
                    pad_l = pad // 2
                    pad_r = pad // 2
                else:
                    pad_l = pad // 2
                    pad_r = pad // 2 + 1                
                
                image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?                
        
            # Write a face region image.
            face_file_name = file_name[:-4] + '_' + str(k) + '_' \
                + str(int(df.iloc[i, 3])) + '_' + str(int(df.iloc[i, 4])) + file_name[-4:]
                
            print('Save ' + face_file_name)
            imsave(os.path.join('subject_faces', face_file_name), (image).astype('uint8'))           
            
            # Add subject face information into db.
            db = pd.concat([db, pd.DataFrame({'subject_id': [k]
                                                  , 'face_file': [face_file_name]
                                                  , 'w': [w]
                                                  , 'h': [h]})])
    # Save db.
    db.to_csv('db.csv')
            
class FaceReIdentifier(object):
    """Face re-identifier to use yolov3."""
    # Constants.
    MODEL_PATH = 'face_reidentifier.h5'

    class TrainingSequence(Sequence):
        """Training data set sequence."""
        
        def __init__(self, raw_data_path, hps, load_flag = True):
            if load_flag:
                with open('img_triplet_pairs.pickle', 'rb') as f:
                    self.img_triplet_pairs = pickle.load(f)
                    self.img_triplet_pairs = self.img_triplet_pairs
                    
                # Create indexing data of positive and negative cases.
                self.raw_data_path = raw_data_path
                self.hps = hps
                self.db = pd.read_csv('db.csv')
                self.db = self.db.iloc[:, 1:]

                self.batch_size = self.hps['batch_size']
                self.hps['step'] = len(self.img_triplet_pairs) // self.batch_size
                
                if len(self.img_triplet_pairs) % self.batch_size != 0:
                    self.hps['step'] +=1    
            else:    
                # Create indexing data of positive and negative cases.
                self.raw_data_path = raw_data_path
                self.hps = hps
                self.db = pd.read_csv('db.csv')
                self.db = self.db.iloc[:, 1:]
                self.t_indexes = np.asarray(self.db.index)
                self.db_g = self.db.groupby('subject_id')
                
                self.img_triplet_pairs = []
                valid_indexes = self.t_indexes
                
                for i in self.db_g.groups.keys():                
                    df = self.db_g.get_group(i)
                    ex_indexes2 = np.asarray(df.index)
                    
                    ex_inv_idxes = []
                    for v in valid_indexes: 
                        if (ex_indexes2 == v).any():
                            ex_inv_idxes.append(False)
                        else:
                            ex_inv_idxes.append(True)
                    ex_inv_idxes = np.asarray(ex_inv_idxes)
                    valid_indexes2 = valid_indexes[ex_inv_idxes]   
                    
                    # Triplet sample pair.
                    for k in range(0, ex_indexes2.shape[0] - 1):
                        for l in range(k + 1, ex_indexes2.shape[0]):
                            self.img_triplet_pairs.append((ex_indexes2[k]
                                                   , ex_indexes2[l]
                                                   , np.random.choice(valid_indexes2, size=1)[0])) 
                
                self.batch_size = self.hps['batch_size']
                self.hps['step'] = len(self.img_triplet_pairs) // self.batch_size
                
                if len(self.img_triplet_pairs) % self.batch_size != 0:
                    self.hps['step'] +=1
                
                # Shuffle image pairs.
                shuffle(self.img_triplet_pairs)
                
                with open('img_triplet_pairs.pickle', 'wb') as f:
                    pickle.dump(self.img_triplet_pairs, f)
                
        def __len__(self):
            return self.hps['step']
        
        def __getitem__(self, index):
            images_a = []
            images_p = []
            images_n = []
            
            # Check the last index.
            if index == (self.hps['step'] - 1):
                for bi in range(index * self.batch_size, len(self.img_triplet_pairs)):
                    # Get the anchor and comparison images.
                    image_a = cv.imread(os.path.join(self.raw_data_path
                                                     , 'subject_faces'
                                                     , self.db.loc[self.img_triplet_pairs[bi][0], 'face_file']))
                    image_p = cv.imread(os.path.join(self.raw_data_path
                                                     , 'subject_faces'
                                                     , self.db.loc[self.img_triplet_pairs[bi][1], 'face_file']))
                    image_n = cv.imread(os.path.join(self.raw_data_path
                                                     , 'subject_faces'
                                                     , self.db.loc[self.img_triplet_pairs[bi][2], 'face_file']))
                    
                    images_a.append(image_a/255)
                    images_p.append(image_p/255)
                    images_n.append(image_n/255)
            
            else:
                for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                    # Get the anchor and comparison images.
                    image_a = cv.imread(os.path.join(self.raw_data_path
                                                     , 'subject_faces'
                                                     , self.db.loc[self.img_triplet_pairs[bi][0], 'face_file']))
                    image_p = cv.imread(os.path.join(self.raw_data_path
                                                     , 'subject_faces'
                                                     , self.db.loc[self.img_triplet_pairs[bi][1], 'face_file']))
                    image_n = cv.imread(os.path.join(self.raw_data_path
                                                     , 'subject_faces'
                                                     , self.db.loc[self.img_triplet_pairs[bi][2], 'face_file']))
                    
                    images_a.append(image_a/255)
                    images_p.append(image_p/255)
                    images_n.append(image_n/255)                 
                                                                                                                     
            return ({'input_a': np.asarray(images_a)
                     , 'input_p': np.asarray(images_p)
                     , 'input_n': np.asarray(images_n)}
                     , {'output': np.zeros(shape=(len(images_a), 192))}) 

    def __init__(self, raw_data_path, hps, model_loading):
        """
        Parameters
        ----------
        raw_data_path : string
            Raw data path
        hps : dictionary
            Hyper-parameters
        model_loading : boolean 
            Face re-identification model loading flag
        """
        # Initialize.
        self.raw_data_path = raw_data_path
        self.hps = hps
        self.model_loading = model_loading
        #trGen = self.TrainingSequence(self.raw_data_path, self.hps, load_flag=False)
        
        if model_loading: 
            if MULTI_GPU:
                self.model = load_model(self.MODEL_PATH, custom_objects={'triplet_loss': triplet_loss})
                self.parallel_model = multi_gpu_model(self.model, gpus = NUM_GPUS)
                
                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay']) 
                self.parallel_model.compile(optimizer=opt, loss=triplet_loss)
            else:
                self.model = load_model(self.MODEL_PATH, custom_objects={'triplet_loss': triplet_loss})
        else:
            # Design the face re-identification model.
            # Inputs.
            input_a = Input(shape=(self.hps['image_size'], self.hps['image_size'], 3), name='input_a')
            input_p = Input(shape=(self.hps['image_size'], self.hps['image_size'], 3), name='input_p')
            input_n = Input(shape=(self.hps['image_size'], self.hps['image_size'], 3), name='input_n')

            # Load yolov3 as the base model.
            base = self.YOLOV3Base
            base.name = 'base' 
            
            # Get triplet facial ids.
            xa = base(input_a) # Non-linear.
            xa = Flatten()(xa)
            
            c_dense_layer = Dense(self.hps['dense1'], activation='relu', name='dense1')
            l2_norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='l2_norm_layer')
            
            xa = c_dense_layer(xa)
            xa = l2_norm_layer(xa)
            
            '''
            for i in range(self.hps['num_dense1_layers']):
                xa = Dense(self.hps['dense1'], activation='linear', name='dense1_anchor_' + str(i))(xa)
            '''
            
            xp = base(input_p)
            xp = Flatten()(xp)
            xp = c_dense_layer(xp)
            xp = l2_norm_layer(xp)
            
            '''
            for i in range(self.hps['num_dense1_layers']):
                xp = Dense(self.hps['dense1'], activation='linear', name='dense1_pos_' + str(i))(xp)
            '''

            xn = base(input_n)
            xn = Flatten()(xn)
            xn = c_dense_layer(xn)
            xn = l2_norm_layer(xn)
            
            '''
            for i in range(self.hps['num_dense1_layers']):
                xn = Dense(self.hps['dense1'], activation='linear', name='dense1_neg_' + str(i))(xn)            
            '''
            
            output = Concatenate(name='output')([xa, xp, xn]) #?

            if MULTI_GPU:
                self.model = Model(inputs=[input_a, input_p, input_n], outputs=[output])
                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay'])
                
                self.model.compile(optimizer=opt, loss=triplet_loss)
                self.model.summary()
                
                self.parallel_model = multi_gpu_model(Model(inputs=[input_a, input_p, input_n], outputs=[output])
                                                   , gpus = NUM_GPUS)
                self.parallel_model.compile(optimizer=opt, loss=triplet_loss)
                self.parallel_model.summary()
            else:
                self.model = Model(inputs=[input_a, input_p, input_n], outputs=[output])
                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay'])
                
                self.model.compile(optimizer=opt, loss=triplet_loss)
                self.model.summary()

        # Create face detector.
        self.fd = FaceDetector(self.raw_data_path, self.hps, True)
        
        # Make fid extractor and face identifier.
        self._make_fid_extractor()

    def _make_fid_extractor(self):
        """Make facial id extractor."""
        # Design the face re-identification model.
        # Inputs.
        input1 = Input(shape=(self.hps['image_size'], self.hps['image_size'], 3), name='input1')
 
        # Load yolov3 as the base model.
        base = self.model.get_layer('base')
                
        # Get facial id.
        x = base(input1) # Non-linear.
        x = Flatten()(x)
        
        '''
        for i in range(self.hps['num_dense1_layers']):
            x = self.model.get_layer('dense1_anchor_' + str(i))(x)
        '''
        
        x = self.model.get_layer('dense1')(x)
        x = self.model.get_layer('l2_norm_layer')(x)
        
        facial_id = x
        self.fid_extractor = Model(inputs=[input1], outputs=[facial_id])
    
    @property
    def YOLOV3Base(self):
        """Get yolov3 as a base model.
        
        Returns
        -------
        Model of Keras
            Partial yolo3 model from the input layer to the add_23 layer
        """
        
        if YOLO3_BASE_MODEL_LOAD_FLAG:
            base = load_model('yolov3_base.h5')
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
        base.save('yolov3_base.h5')
        
        return base
        
    def train(self):
        """Train face detector."""
        trGen = self.TrainingSequence(self.raw_data_path, self.hps, load_flag=False)
        
        if MULTI_GPU:
            self.parallel_model.fit_generator(trGen
                          , steps_per_epoch=self.hps['step']                  
                          , epochs=self.hps['epochs']
                          , verbose=1
                          , max_queue_size=100
                          , workers=8
                          , use_multiprocessing=True)
        else:     
            self.model.fit_generator(trGen
                          , steps_per_epoch=self.hps['step']                  
                          , epochs=self.hps['epochs']
                          , verbose=1
                          , max_queue_size=10
                          , workers=4
                          , use_multiprocessing=True)

        print('Save the model.')            
        self.model.save(self.MODEL_PATH)
    
    def register_facial_ids(self):
        """Register facial ids."""
        db = pd.read_csv('db.csv')
        db = db.iloc[:, 1:]
        db_g = db.groupby('subject_id')

        db_facial_id = pd.DataFrame(columns=['subject_id', 'facial_id'])
        for subject_id in db_g.groups.keys():
            if subject_id == -1:
                continue
            
            # Get face images of a subject id.
            df = db_g.get_group(subject_id)
            images = []
            
            for ff in list(df.iloc[:, 1]):
                image = cv.imread(os.path.join('subject_faces', ff))
                r = image[:, :, 0].copy()
                g = image[:, :, 1].copy()
                b = image[:, :, 2].copy()
                image[:, :, 0] = b
                image[:, :, 1] = g
                image[:, :, 2] = r
                images.append(image/255)
            
            images = np.asarray(images)
            
            # Calculate facial ids and an averaged facial id of a subject id. Mean, Mode, Median?
            facial_ids = self.fid_extractor.predict(images)
            facial_id = np.asarray(pd.DataFrame(facial_ids).mean())
            
            db_facial_id = pd.concat([db_facial_id, pd.DataFrame({'subject_id': [subject_id]
                                                                  , 'facial_id': [facial_id]})])
        
        # Save db.
        db_facial_id.index = db_facial_id.subject_id
        db_facial_id = db_facial_id.to_dict()['facial_id']
        
        with open('db_facial_id.pobj', 'wb') as f:
            pickle.dump(db_facial_id, f)

    def evaluate(self, test_path, output_file_path):
        """Evaluate.
        
        Parameters
        ----------
        test_path : string
            Testing directory.
        output_file_path : string
            Output file path.
        """ 
        if not os.path.isdir(os.path.join(self.raw_data_path, 'results')):
            os.mkdir(os.path.join(self.raw_data_path, 'results'))
        else:
            shutil.rmtree(os.path.join(self.raw_data_path, 'results'))
            os.mkdir(os.path.join(self.raw_data_path, 'results'))
        
        gt_df = pd.read_csv(os.path.join(self.raw_data_path, 'training.csv'))
        gt_df_g = gt_df.groupby('FILE')        
        file_names = glob.glob(os.path.join(test_path, '*.jpg'))
        
        with open('db_facial_id.pobj', 'rb') as f:
            db_facial_id = pickle.load(f)
        
        # Get registered facial id data.
        subject_ids = list(db_facial_id.keys())
        facial_ids = []
        
        for subject_id in subject_ids:
            facial_ids.append(db_facial_id[subject_id])
            
        reg_facial_ids = np.asarray(facial_ids)
        
        # Detect faces, identify faces and save results.
        count1 = 1
        
        with open(output_file_path, 'w') as f:
            for file_name in file_names:
                if DEBUG: print(count1, '/', len(file_names), file_name)
                count1 += 1
                
                # Load an image.
                image = imread(os.path.join(test_path, file_name))
                image_o = image.copy()
                image = image/255 
             
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
                boxes = self.fd.detect(image)
                
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
                        
                count = 1
                
                for box in boxes:
                    if count > 60:
                        break
                    
                    # Search for id from registered facial ids.
                    # Crop a face region.
                    l, t, r, b = int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)
                    image = image_o[(t - 1):(b - 1), (l - 1):(r - 1), :]
                    image = image/255
                    
                    # Adjust the original image size into the normalized image size according to the ratio of width, height.
                    w = image.shape[1]
                    h = image.shape[0]
                    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0

                    # Check exception.
                    if w == 0 or h == 0:
                        continue
                                    
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
                    
                    # Create anchor facial ids.
                    anchor_facial_id = self.fid_extractor.predict(image[np.newaxis, ...])
                    anchor_facial_id = np.squeeze(anchor_facial_id)
                    
                    # Calculate similarity distances for each registered face ids.
                    sim_dists = []
                    for i in range(len(subject_ids)):
                        sim_dists.append(norm(anchor_facial_id - reg_facial_ids[i]))
                    sim_dists = np.asarray(sim_dists)
                    cand = np.argmin(sim_dists)

                    #if sim_dists[cand] > self.hps['sim_th']:
                    #    continue
                    
                    subject_id = subject_ids[cand]
                    box.subject_id = subject_id     
                    
                    if platform.system() == 'Windows':
                        f.write(file_name.split('\\')[-1] + ',' + str(subject_id) + ',' + str(box.xmin) + ',' + str(box.ymin) + ',')
                    else:
                        f.write(file_name.split('/')[-1] + ',' + str(subject_id) + ',' + str(box.xmin) + ',' + str(box.ymin) + ',')
                        
                    f.write(str(box.xmax - box.xmin) + ',' + str(box.ymax - box.ymin) + ',' + str(box.get_score()) + '\n')
                    count +=1

                #boxes = [box for box in boxes if box.subject_id != -1]

                # Draw bounding boxes of ground truth.
                if platform.system() == 'Windows':
                    file_new_name = file_name.split('\\')[-1]
                else:
                    file_new_name = file_name.split('/')[-1]
                
                try:
                    df = gt_df_g.get_group(file_new_name)
                except KeyError:
                    continue
                
                gt_boxes = []
                
                for i in range(df.shape[0]):
                    # Check exception.
                    res = df.iloc[i, 3:] > 0 #?
                    if res.all() == False or df.iloc[i, 2] == -1:
                        continue
                    
                    xmin = int(df.iloc[i, 3])
                    xmax = int(xmin + df.iloc[i, 5] - 1)
                    ymin = int(df.iloc[i, 4])
                    ymax = int(ymin + df.iloc[i, 6] - 1)                    
                    gt_box = BoundBox(xmin, ymin, xmax, ymax, objness=1., classes=[1.0], subject_id=df.iloc[i, 2])
                    gt_boxes.append(gt_box)
                 
                # Check exception.
                if len(gt_boxes) == 0 or len(boxes) == 0: #?
                    continue
                    
                image = draw_boxes_v3(image_o, gt_boxes, self.hps['face_conf_th'], color=(255, 0, 0)) 

                # Draw bounding boxes on the image using labels.
                image = draw_boxes_v3(image, boxes, self.hps['face_conf_th'], color=(0, 255, 0)) 
         
                # Write the image with bounding boxes to file.
                # Draw bounding boxes of ground truth.
                if platform.system() == 'Windows':
                    file_new_name = file_name.split('\\')[-1]
                else:
                    file_new_name = file_name.split('/')[-1]
                    
                file_new_name = file_new_name[:-4] + '_detected' + file_new_name[-4:]
                
                print(file_new_name)
                imsave(os.path.join(test_path, 'results', file_new_name), (image).astype('uint8'))
        
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
        with open('db_facial_id.pobj', 'rb') as f:
            db_facial_id = pickle.load(f)
        
        # Get registered facial id data.
        subject_ids = list(db_facial_id.keys())
        facial_ids = []
        
        for subject_id in subject_ids:
            facial_ids.append(db_facial_id[subject_id])
            
        reg_facial_ids = np.asarray(facial_ids)
        
        # Detect faces, identify faces and save results.
        count1 = 1
        with open(output_file_path, 'w') as f:
            for file_name in file_names:
                if DEBUG: print(count1, '/', len(file_names), file_name)
                count1 += 1
                
                # Load an image.
                image = imread(os.path.join(test_path, file_name))
                image_o = image.copy()
                image = image/255 
             
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
                boxes = self.fd.detect(image)
                
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
                        
                count = 1
                
                for box in boxes:
                    if count > 60:
                        break
                    
                    # Search for id from registered facial ids.
                    # Crop a face region.
                    l, t, r, b = int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)
                    image = image_o[(t - 1):(b - 1), (l - 1):(r - 1), :]
                    image = image/255
                    
                    # Adjust the original image size into the normalized image size according to the ratio of width, height.
                    w = image.shape[1]
                    h = image.shape[0]
                    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
                    
                    # Check exception.
                    if w == 0 or h == 0:
                        continue
                                    
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
                    
                    # Create anchor facial ids.
                    anchor_facial_id = self.fid_extractor.predict(image[np.newaxis, ...])
                    anchor_facial_id = np.squeeze(anchor_facial_id)
                    anchor_facial_ids = np.asarray([anchor_facial_id for _ in range(len(subject_ids))])
                    
                    # Calculate similarity distances for each registered face ids.
                    sim_dists = []
                    for i in range(len(subject_ids)):
                        sim_dists.append(norm(anchor_facial_ids[i] - reg_facial_ids[i]))
                    sim_dists = np.asarray(sim_dists)
                    cand = np.argmin(sim_dists)

                    if sim_dists[cand] > self.hps['sim_th']:
                        continue
                    
                    subject_id = subject_ids[cand]    
                    
                    if platform.system() == 'Windows':
                        f.write(file_name.split('\\')[-1] + ',' + str(subject_id) + ',' + str(box.xmin) + ',' + str(box.ymin) + ',')
                    else:
                        f.write(file_name.split('/')[-1] + ',' + str(subject_id) + ',' + str(box.xmin) + ',' + str(box.ymin) + ',')
                        
                    f.write(str(box.xmax - box.xmin) + ',' + str(box.ymax - box.ymin) + ',' + str(box.get_score()) + '\n')
                    count +=1

                # Check exception.
                if len(boxes) == 0:
                    continue

                '''
                # Draw bounding boxes on the image using labels.
                image = draw_boxes_v2(image_o, boxes, self.hps['face_conf_th']) 
         
                # Write the image with bounding boxes to file.
                # Draw bounding boxes of ground truth.
                if platform.system() == 'Windows':
                    file_new_name = file_name.split('\\')[-1]
                else:
                    file_new_name = file_name.split('/')[-1]
                    
                file_new_name = file_new_name[:-4] + '_detected' + file_new_name[-4:]
                
                print(file_new_name)
                imsave(os.path.join(test_path, 'results', file_new_name), (image).astype('uint8'))
                '''
                
def main(args):
    """Main.
    
    Parameters
    ----------
    args : argument type 
        Arguments
    """
    hps = {}
    hps['step'] = 1

    if args.mode == 'data':
        # Get arguments.
        raw_data_path = args.raw_data_path
      
        # hps.
        hps['image_size'] = int(args.image_size)    
        hps['face_conf_th'] = float(args.face_conf_th)
        hps['nms_iou_th'] = float(args.nms_iou_th)
        hps['num_cands'] = int(args.num_cands)
             
        # Create db.
        ts = time.time()
        create_db_fri(raw_data_path, hps)
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))            
    elif args.mode == 'train':
        # Get arguments.
        raw_data_path = args.raw_data_path
      
        # hps.
        hps['image_size'] = int(args.image_size)
        hps['num_dense1_layers'] = int(args.num_dense1_layers)
        hps['dense1'] = int(args.dense1)
        hps['num_dense2_layers'] = int(args.num_dense2_layers)
        hps['dense2'] = int(args.dense2)    
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
        fr = FaceReIdentifier(raw_data_path, hps, model_loading)
        
        ts = time.time()
        fr.train()
        fr.register_facial_ids()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif args.mode == 'evaluate':
        # Get arguments.
        raw_data_path = args.raw_data_path
        output_file_path = args.output_file_path
      
        # hps.
        hps['image_size'] = int(args.image_size)
        hps['num_dense1_layers'] = int(args.num_dense1_layers)
        hps['dense1'] = int(args.dense1)
        hps['num_dense2_layers'] = int(args.num_dense2_layers)
        hps['dense2'] = int(args.dense2)  
        hps['lr'] = float(args.lr)
        hps['beta_1'] = float(args.beta_1)
        hps['beta_2'] = float(args.beta_2)
        hps['decay'] = float(args.decay)
        hps['batch_size'] = int(args.batch_size)
        hps['epochs'] = int(args.epochs) 
        hps['face_conf_th'] = float(args.face_conf_th)
        hps['nms_iou_th'] = float(args.nms_iou_th)
        hps['num_cands'] = int(args.num_cands)
        hps['sim_th'] = float(args.sim_th)
        
        model_loading = False if int(args.model_loading) == 0 else True        
        
        # Test.
        fr = FaceReIdentifier(raw_data_path, hps, model_loading)
        
        ts = time.time()
        #fr.register_facial_ids()
        fr.evaluate(raw_data_path, output_file_path)
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif args.mode == 'test':
        # Get arguments.
        raw_data_path = args.raw_data_path
        output_file_path = args.output_file_path
      
        # hps.
        hps['image_size'] = int(args.image_size)
        hps['num_dense1_layers'] = int(args.num_dense1_layers)
        hps['dense1'] = int(args.dense1)
        hps['num_dense2_layers'] = int(args.num_dense2_layers)
        hps['dense2'] = int(args.dense2)  
        hps['lr'] = float(args.lr)
        hps['beta_1'] = float(args.beta_1)
        hps['beta_2'] = float(args.beta_2)
        hps['decay'] = float(args.decay)
        hps['batch_size'] = int(args.batch_size)
        hps['epochs'] = int(args.epochs) 
        hps['face_conf_th'] = float(args.face_conf_th)
        hps['nms_iou_th'] = float(args.nms_iou_th)
        hps['num_cands'] = int(args.num_cands)
        hps['sim_th'] = float(args.sim_th)
        
        model_loading = False if int(args.model_loading) == 0 else True        
        
        # Test.
        fr = FaceReIdentifier(raw_data_path, hps, model_loading)
        
        ts = time.time()
        #fr.register_facial_ids()
        fr.test(raw_data_path, output_file_path)
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
        
if __name__ == '__main__':
    
    # Parse arguments.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode')
    parser.add_argument('--raw_data_path')
    parser.add_argument('--output_file_path')
    parser.add_argument('--image_size')
    parser.add_argument('--num_dense1_layers')
    parser.add_argument('--dense1')
    parser.add_argument('--num_dense2_layers')
    parser.add_argument('--dense2')
    parser.add_argument('--lr')
    parser.add_argument('--beta_1')
    parser.add_argument('--beta_2')
    parser.add_argument('--decay')
    parser.add_argument('--batch_size')
    parser.add_argument('--epochs')
    parser.add_argument('--face_conf_th')
    parser.add_argument('--nms_iou_th')
    parser.add_argument('--num_cands')
    parser.add_argument('--sim_th')
    parser.add_argument('--model_loading')
    args = parser.parse_args()
    
    main(args)