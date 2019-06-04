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
from random import shuffle
import json

import numpy as np
import pandas as pd
import cv2 as cv
from skimage.io import imread, imsave
from scipy.linalg import norm
import h5py

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, ZeroPadding2D, LeakyReLU, Flatten, Concatenate
from keras.layers.merge import add
from keras.utils import multi_gpu_model
from keras.utils.data_utils import Sequence
import keras.backend as K
from keras import optimizers

from yolov3_detect import make_yolov3_model, BoundBox, WeightReader, draw_boxes_v3
from face_detection import FaceDetector

# Constants.
DEBUG = True

ALPHA = 0.2

def triplet_loss(y_true, y_pred):
    # Calculate the difference of both face features and judge a same person.
    x = y_pred
    return K.mean(K.maximum(K.sqrt(K.sum(K.pow(x[:, 0:64] - x[:, 64:128], 2.0), axis=-1)) \
                     - K.sqrt(K.sum(K.pow(x[:, 0:64] - x[:, 128:192], 2.0), axis=-1)) + ALPHA, 0.))
                         
def create_db_fi(conf):
    """Create db for face identifier."""
    conf = conf['fi_conf']
    
    raw_data_path = conf['raw_data_path']
    nn_arch = conf['nn_arch']
    
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
                w_p = nn_arch['image_size']
                h_p = int(h / w * nn_arch['image_size'])
                pad = nn_arch['image_size'] - h_p
                
                if pad % 2 == 0:
                    pad_t = pad // 2
                    pad_b = pad // 2
                else:
                    pad_t = pad // 2
                    pad_b = pad // 2 + 1
                 
                image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
            else:
                h_p = nn_arch['image_size']
                w_p = int(w / h * nn_arch['image_size'])
                pad = nn_arch['image_size'] - w_p
                
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
    db.to_csv('subject_image_db.csv')
            
class FaceIdentifier(object):
    """Face identifier to use yolov3."""
    
    # Constants.
    MODEL_PATH = 'face_identifier.h5'

    class TrainingSequence(Sequence):
        """Training data set sequence."""
        
        def __init__(self, raw_data_path, hps, nn_arch, load_flag=True):
            if load_flag:
                with open('img_triplet_pairs.pickle', 'rb') as f:
                    self.img_triplet_pairs = pickle.load(f)
                    self.img_triplet_pairs = self.img_triplet_pairs
                    
                # Create indexing data of positive and negative cases.
                self.raw_data_path = raw_data_path
                self.hps = hps
                self.nn_arch = nn_arch
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

    def __init__(self, conf):
        """
        Parameters
        ----------
        conf: dictionary
            Face detector configuration dictionary.
        """
        
        # Initialize.
        self.conf = conf['fi_conf']
        self.raw_data_path = self.conf['raw_data_path']
        self.hps = self.conf['hps']
        self.nn_arch = self.conf['nn_arch']
        self.model_loading = self.conf['model_loading']
                
        if self.model_loading: 
            if self.conf['multi_gpu']:
                self.model = load_model(self.MODEL_PATH, custom_objects={'triplet_loss': triplet_loss})
                self.parallel_model = multi_gpu_model(self.model, gpus=self.conf['num_gpus'])
                
                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay']) 
                self.parallel_model.compile(optimizer=opt, loss=triplet_loss)
            else:
                self.model = load_model(self.MODEL_PATH, custom_objects={'triplet_loss': triplet_loss})
        else:
            # Design the face identification model.
            # Inputs.
            input_a = Input(shape=(self.nn_arch['image_size'], self.nn_arch['image_size'], 3), name='input_a')
            input_p = Input(shape=(self.nn_arch['image_size'], self.nn_arch['image_size'], 3), name='input_p')
            input_n = Input(shape=(self.nn_arch['image_size'], self.nn_arch['image_size'], 3), name='input_n')

            # Load yolov3 as the base model.
            base = self.YOLOV3Base
            base.name = 'base' 
            
            # Get triplet facial ids.
            xa = base(input_a) # Non-linear.
            xa = Flatten()(xa)
            
            c_dense_layer = Dense(self.nn_arch['dense1_dim'], activation='relu', name='dense1')
            l2_norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='l2_norm_layer')
            
            xa = c_dense_layer(xa)
            xa = l2_norm_layer(xa)
                        
            xp = base(input_p)
            xp = Flatten()(xp)
            xp = c_dense_layer(xp)
            xp = l2_norm_layer(xp)
            
            xn = base(input_n)
            xn = Flatten()(xn)
            xn = c_dense_layer(xn)
            xn = l2_norm_layer(xn)
                        
            output = Concatenate(name='output')([xa, xp, xn]) #?

            if self.conf['multi_gpu']:
                self.model = Model(inputs=[input_a, input_p, input_n], outputs=[output])
                opt = optimizers.Adam(lr=self.hps['lr']
                                        , beta_1=self.hps['beta_1']
                                        , beta_2=self.hps['beta_2']
                                        , decay=self.hps['decay'])
                
                self.model.compile(optimizer=opt, loss=triplet_loss)
                self.model.summary()
                
                self.parallel_model = multi_gpu_model(Model(inputs=[input_a, input_p, input_n], outputs=[output])
                                                   , gpus=self.conf['num_gpus'])
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
        self.fd = FaceDetector(conf['fd_conf'])
        
        # Make fid extractor and face identifier.
        self._make_fid_extractor()

    def _make_fid_extractor(self):
        """Make facial id extractor."""
        # Design the face identification model.
        # Inputs.
        input1 = Input(shape=(self.nn_arch['image_size'], self.nn_arch['image_size'], 3), name='input1')
 
        # Load yolov3 as the base model.
        base = self.model.get_layer('base')
                
        # Get facial id.
        x = base(input1) # Non-linear.
        x = Flatten()(x)
                
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
        
        if self.conf['yolov3_base_model_load']:
            base = load_model('yolov3_base.h5')
            base.trainable = True
            return base
        
        yolov3 = make_yolov3_model()

        # Load the weights.
        weight_reader = WeightReader('yolov3.weights')
        weight_reader.load_weights(yolov3)
        
        # Make a base model.
        input1 = Input(shape=(self.nn_arch['image_size'], self.nn_arch['image_size'], 3))
        
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
        trGen = self.TrainingSequence(self.raw_data_path, self.hps, self.nn_arch, load_flag=False)
        
        if self.conf['multi_gpu']:
            self.parallel_model.fit_generator(trGen
                          , steps_per_epoch=self.hps['step'] #?                   
                          , epochs=self.hps['epochs']
                          , verbose=1
                          , max_queue_size=400
                          , workers=8
                          , use_multiprocessing=True)
        else:     
            self.model.fit_generator(trGen
                          , steps_per_epoch=self.hps['step']                  
                          , epochs=self.hps['epochs']
                          , verbose=1
                          , max_queue_size=100
                          , workers=4
                          , use_multiprocessing=True)

        print('Save the model.')            
        self.model.save(self.MODEL_PATH)
  
    def make_facial_ids_db(self):
        """Make facial ids database."""
        db = pd.read_csv('subject_image_db.csv')
        db = db.iloc[:, 1:]
        db_g = db.groupby('subject_id')

        with h5py.File('subject_facial_ids.h5', 'w') as f:
            for subject_id in db_g.groups.keys():
                if subject_id == -1:
                    continue
                
                # Get face images of a subject id.
                df = db_g.get_group(subject_id)
                images = []
                
                for ff in list(df.iloc[:, 1]):
                    image = imread(os.path.join('subject_faces', ff))
                    images.append(image/255)
                
                images = np.asarray(images)
                
                # Calculate facial ids and an averaged facial id of a subject id. Mean, Mode, Median?
                facial_ids = self.fid_extractor.predict(images)

                for k, ff in enumerate(list(df.iloc[:, 1])):
                    f[ff] = facial_ids[k]
                    f[ff].attrs['subject_id'] = subject_id

    def register_facial_ids(self):
        """Register facial ids."""
        db = pd.read_csv('subject_image_db.csv')
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
                image = imread(os.path.join('subject_faces', ff))
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
        
        with open('ref_facial_id_db.pickle', 'wb') as f:
            pickle.dump(db_facial_id, f)

    def evaluate(self):
        """Evaluate."""
        test_path = self.conf['test_path']
        output_file_path = self.conf['output_file_path']
        
        if not os.path.isdir(os.path.join(test_path, 'results_fi')):
            os.mkdir(os.path.join(test_path, 'results_fi'))
        else:
            shutil.rmtree(os.path.join(test_path, 'results_fi'))
            os.mkdir(os.path.join(test_path, 'results_fi'))
        
        gt_df = pd.read_csv(os.path.join(test_path, 'validation.csv'))
        gt_df_g = gt_df.groupby('FILE')        
        file_names = glob.glob(os.path.join(test_path, '*.jpg'))
        
        with open('ref_facial_id_db.pickle', 'rb') as f:
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
                    w_p = self.nn_arch['image_size']
                    h_p = int(h / w * self.nn_arch['image_size'])
                    pad = self.nn_arch['image_size'] - h_p
                    
                    if pad % 2 == 0:
                        pad_t = pad // 2
                        pad_b = pad // 2
                    else:
                        pad_t = pad // 2
                        pad_b = pad // 2 + 1
    
                    image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                    image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
                else:
                    h_p = self.nn_arch['image_size']
                    w_p = int(w / h * self.nn_arch['image_size'])
                    pad = self.nn_arch['image_size'] - w_p
                    
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
                        box.xmin = np.min([box.xmin * w / self.nn_arch['image_size'], w])
                        box.xmax = np.min([box.xmax * w / self.nn_arch['image_size'], w])
                        box.ymin = np.min([np.max([box.ymin - pad_t, 0]) * w / self.nn_arch['image_size'], h])
                        box.ymax = np.min([np.max([box.ymax - pad_t, 0]) * w / self.nn_arch['image_size'], h])
                    else:
                        box.xmin = np.min([np.max([box.xmin - pad_l, 0]) * h / self.nn_arch['image_size'], w])
                        box.xmax = np.min([np.max([box.xmax - pad_l, 0]) * h / self.nn_arch['image_size'], w])
                        box.ymin = np.min([box.ymin * h / self.nn_arch['image_size'], h])
                        box.ymax = np.min([box.ymax * h / self.nn_arch['image_size'], h])
                        
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
                        w_p = self.nn_arch['image_size']
                        h_p = int(h / w * self.nn_arch['image_size'])
                        pad = self.nn_arch['image_size'] - h_p
                        
                        if pad % 2 == 0:
                            pad_t = pad // 2
                            pad_b = pad // 2
                        else:
                            pad_t = pad // 2
                            pad_b = pad // 2 + 1
                         
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
                    else:
                        h_p = self.nn_arch['image_size']
                        w_p = int(w / h * self.nn_arch['image_size'])
                        pad = self.nn_arch['image_size'] - w_p
                        
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

                    if sim_dists[cand] > self.hps['sim_th']:
                        continue
                    
                    subject_id = subject_ids[cand]
                    box.subject_id = subject_id     
                    
                    if platform.system() == 'Windows':
                        f.write(file_name.split('\\')[-1] + ',' + str(subject_id) + ',' + str(box.xmin) + ',' + str(box.ymin) + ',')
                        print(file_name.split('\\')[-1] + ',' + str(subject_id) + ',' + str(box.xmin) + ',' + str(box.ymin) + ',', end=' ')
                    else:
                        f.write(file_name.split('/')[-1] + ',' + str(subject_id) + ',' + str(box.xmin) + ',' + str(box.ymin) + ',')
                        print(file_name.split('/')[-1] + ',' + str(subject_id) + ',' + str(box.xmin) + ',' + str(box.ymin) + ',', end=' ')
                        
                    f.write(str(box.xmax - box.xmin) + ',' + str(box.ymax - box.ymin) + ',' + str(box.get_score()) + '\n')
                    print(str(box.xmax - box.xmin) + ',' + str(box.ymax - box.ymin) + ',' + str(box.get_score()))
                    
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
                    if res.all() == False: #or df.iloc[i, 2] == -1:
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
                    
                image1 = draw_boxes_v3(image_o, gt_boxes, self.hps['face_conf_th'], color=(255, 0, 0)) 
                del image_o

                # Draw bounding boxes on the image using labels.
                image = draw_boxes_v3(image1, boxes, self.hps['face_conf_th'], color=(0, 255, 0)) 
                del image1
                
                # Write the image with bounding boxes to file.
                # Draw bounding boxes of ground truth.
                if platform.system() == 'Windows':
                    file_new_name = file_name.split('\\')[-1]
                else:
                    file_new_name = file_name.split('/')[-1]
                    
                file_new_name = file_new_name[:-4] + '_detected' + file_new_name[-4:]
                
                print(file_new_name)
                imsave(os.path.join(test_path, 'results_fi', file_new_name), (image).astype('uint8'))
        
    def test(self):
        """Test."""
        test_path = self.conf['test_path']
        output_file_path = self.conf['output_file_path']
              
        file_names = glob.glob(os.path.join(test_path, '*.jpg'))
        with open('ref_facial_id_db.pickle', 'rb') as f:
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
                    w_p = self.nn_arch['image_size']
                    h_p = int(h / w * self.nn_arch['image_size'])
                    pad = self.nn_arch['image_size'] - h_p
                    
                    if pad % 2 == 0:
                        pad_t = pad // 2
                        pad_b = pad // 2
                    else:
                        pad_t = pad // 2
                        pad_b = pad // 2 + 1
    
                    image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                    image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
                else:
                    h_p = self.nn_arch['image_size']
                    w_p = int(w / h * self.nn_arch['image_size'])
                    pad = self.nn_arch['image_size'] - w_p
                    
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
                        box.xmin = np.min([box.xmin * w / self.nn_arch['image_size'], w])
                        box.xmax = np.min([box.xmax * w / self.nn_arch['image_size'], w])
                        box.ymin = np.min([np.max([box.ymin - pad_t, 0]) * w / self.nn_arch['image_size'], h])
                        box.ymax = np.min([np.max([box.ymax - pad_t, 0]) * w / self.nn_arch['image_size'], h])
                    else:
                        box.xmin = np.min([np.max([box.xmin - pad_l, 0]) * h / self.nn_arch['image_size'], w])
                        box.xmax = np.min([np.max([box.xmax - pad_l, 0]) * h / self.nn_arch['image_size'], w])
                        box.ymin = np.min([box.ymin * h / self.nn_arch['image_size'], h])
                        box.ymax = np.min([box.ymax * h / self.nn_arch['image_size'], h])
                        
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
                        w_p = self.nn_arch['image_size']
                        h_p = int(h / w * self.nn_arch['image_size'])
                        pad = self.nn_arch['image_size'] - h_p
                        
                        if pad % 2 == 0:
                            pad_t = pad // 2
                            pad_b = pad // 2
                        else:
                            pad_t = pad // 2
                            pad_b = pad // 2 + 1
                                                   
                        image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                        image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
                    else:
                        h_p = self.nn_arch['image_size']
                        w_p = int(w / h * self.nn_arch['image_size'])
                        pad = self.nn_arch['image_size'] - w_p
                        
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
                
def main():
    """Main."""
    
    # Load configuration.
    if platform.system() == 'Windows':
        with open("face_vijnana_yolov3_win.json", 'r') as f:
            conf = json.load(f)   
    else:
        with open("face_vijnana_yolov3.json", 'r') as f:
            conf = json.load(f)   
                
    if conf['fi_conf']['mode'] == 'data':               
        # Create db.
        ts = time.time()
        create_db_fi(conf)
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))            
    elif conf['fi_conf']['mode'] == 'train':        
        # Train.
        fi = FaceIdentifier(conf)
        
        ts = time.time()
        fi.train()
        fi.make_facial_ids_db()
        fi.register_facial_ids()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif conf['fi_conf']['mode'] == 'evaluate':        
        # Test.
        fi = FaceIdentifier(conf)
        
        ts = time.time()
        #fi.register_facial_ids()
        fi.evaluate()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif conf['fi_conf']['mode'] == 'test':        
        # Test.
        fi = FaceIdentifier(conf)
        
        ts = time.time()
        #fi.register_facial_ids()
        fi.test()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif conf['fi_conf']['mode'] == 'fid_db':
        fi = FaceIdentifier(conf)
        
        ts = time.time()
        fi.make_facial_ids_db()
        #fi.register_facial_ids()
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))        
        
if __name__ == '__main__':
    main()