'''
Created on Apr 9, 2019

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import os
import glob
import argparse
import time
import pickle

import numpy as np
import pandas as pd
import cv2 as cv
from skimage.io import imread, imsave
from scipy.stats import entropy

from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Lambda, ZeroPadding2D, LeakyReLU, Flatten
from keras.layers.merge import add, concatenate
from keras.utils import multi_gpu_model
from keras.utils.data_utils import Sequence
import keras.backend as K
from keras import optimizers

from yolov3_detect import make_yolov3_model, BoundBox, do_nms_v2, WeightReader, draw_boxes_v2
from face_detection import FaceDetector

# Constants.
DEBUG = True
MULTI_GPU = False
NUM_GPUS = 4

def create_db_fri(raw_data_path, hps):
    """Create db for face re-identifier."""
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
            imsave(os.path.join(raw_data_path, 'subject_faces', face_file_name), (image).astype('uint8'))           
            
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
    MODEL_PATH = 'face_reidentifier.hd5'

    class TrainingSequence(Sequence):
        """Training data set sequence."""
        
        def __init__(self, raw_data_path, hps):
            # Create indexing data of positive and negative cases.
            self.raw_data_path = raw_data_path
            self.hps = hps
            self.db = pd.read_csv('db.csv')
            self.db = self.db.iloc[:, 1:]
            self.t_indexes = np.asarray(self.db.index)
            self.db_g = self.db.groupby('subject_id')
            self.img_pairs = []
            valid_indexes = self.t_indexes
            
            for i in self.db_g.groups.keys():                
                df = self.db_g.get_group(i)
                ex_indexes2 = np.asarray(df.index)
                
                # Positive sample pair.
                for k in range(0, ex_indexes2.shape[0] - 1):
                    for l in range(k + 1, ex_indexes2.shape[0]):
                        self.img_pairs.append((ex_indexes2[k], ex_indexes2[l], 1.)) 
                
                # Negative sample pair.
                ex_inv_idxes = []
                for v in ex_indexes2:
                    if (valid_indexes == v).any():
                        ex_inv_idxes.append(False)
                    else:
                        ex_inv_idxes.append(True)
                ex_inv_idxes = np.asarray(ex_inv_idxes)
                valid_indexes2 = valid_indexes[ex_inv_idxes]                
                
                for j in ex_indexes2:
                    self.img_pairs.append((j, np.random.choice(valid_indexes2, size=1)[0], 0.))
            
            self.steps = self.hps['step_per_epoch']
            self.batch_size = len(self.img_pairs) // self.steps
                
        def __len__(self):
            return self.steps
        
        def __getitem__(self, index):
            # Check the last index.
            # TODO
            
            images_a = []
            images_c = []
            gt_tensors = []
            
            for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                # Get the anchor and comparison images.
                image_a = cv.imread(os.path.join(self.raw_data_path
                                                 , 'subject_faces'
                                                 , self.db.loc[self.img_pairs[bi][0], 'face_file']))
                image_c = cv.imread(os.path.join(self.raw_data_path
                                                 , 'subject_faces'
                                                 , self.db.loc[self.img_pairs[bi][1], 'face_file']))
                
                images_a.append(image_a)
                images_c.append(image_c)
                    
                # Create a ground truth.
                gt_tensor = np.asarray([self.img_pair[bi, 2]])
                
                gt_tensors.append(gt_tensor)
                                                                         
            return ({'input_a': np.asarray(images_a), 'input_c': np.asarray(images_c)}
                    , {'output': np.asarray(gt_tensors)}) 

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
        
        if model_loading: 
            self.model = load_model(os.path.join(self.MODEL_PATH))
        else:
            # Design the face re-identification model.
            # Inputs.
            input_a = Input(shape=(self.hps['image_size'], self.hps['image_size'], 3), name='input_a')
            input_c = Input(shape=(self.hps['image_size'], self.hps['image_size'], 3), name='input_c')

            # Load yolov3 as the base model.
            base = self.YOLOV3Base 
            
            # Get both facial ids.
            xa = base(input_a) # Non-linear.
            xa = Flatten()(xa)
            
            for i in range(self.hps['num_dense1_layers']):
                xa = Dense(self.hps['dense1'], activation='linear', name='dense1_anchor_' + str(i))(xa)
            
            xc = base(input_c)
            xc = Flatten()(xc)
            
            for i in range(self.hps['num_dense1_layers']):
                xc = Dense(self.hps['dense1'], activation='linear', name='dense1_comp_' + str(i))(xc)           
            
            # Calculate the difference of both face features and judge a same person.
            xd = Lambda(lambda x: K.pow(x[0] - x[1], 2.)/(x[0] - x[1]), name='lambda1')([xa, xc]) #?
            
            for i in range(self.hps['num_dense2_layers']):
                xd = Dense(self.hps['dense2'], activation='relu', name='dense2_' + str(i))(xd)
            
            output = Dense(1, activation='sigmoid', name='output')(xd)

            if MULTI_GPU:
                self.model = multi_gpu_model(Model(inputs=[input_a, input_c], outputs=[output])
                                                   , gpus = NUM_GPUS)
            else:
                self.model = Model(inputs=[input_a, input_c], outputs=[output])

            opt = optimizers.Adam(lr=self.hps['lr']
                                    , beta_1=self.hps['beta_1']
                                    , beta_2=self.hps['beta_2']
                                    , decay=self.hps['decay'])
            
            self.model.compile(optimizer=opt, loss='binary_crossentropy')
            self.model.summary()

        # Create face detector.
        self.fd = FaceDetector(self.raw_data_path, self.hps, True)
        
        # Make fid extractor and face indentifier.
        self._make_fid_extractor()
        self._make_face_identifier()

    def _make_fid_extractor(self):
        """Make facial id extractor."""
        # Design the face re-identification model.
        # Inputs.
        input = Input(shape=(self.hps['image_size'], self.hps['image_size'], 3), name='input')

        # Load yolov3 as the base model.
        base = self.YOLOV3Base 
        
        # Get facial id.
        x = base(input) # Non-linear.
        x = Flatten()(x)
        
        for i in range(self.hps['num_dense1_layers']):
            x = self.model.get_layer('dense1_anchor_' + str(i))(x)
        
        facial_id = x
        self.fid_extractor = Model(inputs=[input], outputs=[facial_id])
    
    def _make_face_identifier(self):
        """Make face identifier."""
        facial_id_a = Input(shape=(self.hps['dense1'],))
        facial_id_c = Input(shape=(self.hps['dense1'],))
        
        # Calculate the difference of both face features and judge a same person.
        xd = self.model.get_layer('lambda1')([facial_id_a, facial_id_c])
        
        for i in range(self.hps['num_dense2_layers']):
            xd = self.model.get_layer('dense2_' + str(i))(xd)
        
        output = self.model.get_layer('output')(xd)
        self.face_identifier = Model(inputs=[facial_id_a, facial_id_c], outputs=[output])

    @property
    def YOLOV3Base(self):
        """Get yolov3 as a base model.
        
        Returns
        -------
        Model of Keras
            Partial yolo3 model from the input layer to the add_23 layer
        """
        yolov3 = make_yolov3_model()

        # Load the weights.
        weight_reader = WeightReader('yolov3.weights')
        weight_reader.load_weights(yolov3)
        
        # Make a base model.
        input = Input(shape=(self.hps['image_size'], self.hps['image_size'], 3))
        
        # 0 ~ 1.
        conv_layer = yolov3.get_layer('conv_' + str(0))
        x = ZeroPadding2D(1)(input) #?               
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
        return base
        
    def train(self):
        """Train face detector."""
        trGen = self.TrainingSequence(self.raw_data_path, self.hps)
            
        self.model.fit_generator(trGen
                      , steps_per_epoch=self.hps['step_per_epoch']                  
                      , epochs=self.hps['epochs']
                      , verbose=1
                      , max_queue_size=10
                      , workers=1
                      , use_multiprocessing=False)
    
    def register_facial_ids(self):
        """Register facial ids."""
        db = pd.read_csv('db.csv')
        db = self.db.iloc[:, 1:]
        db_g = self.db.groupby('subject_id')

        db_facial_id = pd.DataFrame(columns=['subject_id', 'facial_id'])
        for subject_id in db_g.groups.keys():
            if subject_id == -1:
                continue
            
            # Get face images of a subject id.
            df = db_g.get_group(subject_id)
            images = []
            
            for ff in list(df.iloc[:, 1]):
                image = cv.imread(os.path.join(self.raw_data_path, 'subject_faces', ff))
                images.append(image)
            
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
            pickle.dump(f, db_facial_id)
        
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
        with open(output_file_path, 'w') as f:
            for file_name in file_names:
                if DEBUG: print(file_name)
                
                # Load an image.
                image = cv.imread(os.path.join(test_path, file_name))
                image_o_size = (image.shape[0], image.shape[1])
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
                boxes = self.fd.detect(image, image_o_size)
                
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
                    
                    # Create anchor facial ids.
                    anchor_facial_id = self.fid_extractor(image[np.newaxis, ...])
                    anchor_facial_id = np.squeeze(anchor_facial_id)
                    anchor_facial_ids = np.asarray([anchor_facial_id for _ in range(len(subject_ids))])
                    
                    # Calculate consistency probabilities for each registered face ids.
                    cons_probs = self.face_identifier.predict([anchor_facial_ids, reg_facial_ids])
                    cons_probs = np.squeeze(cons_probs)
                    
                    # Calculate consistency probabilities' Shannon entropy.
                    e = entropy(cons_probs)
                    
                    # Check whether an anchor facial id is a registered facial id.
                    if e > self.hps['entropy_th']:
                        continue
                    
                    subject_id = subject_ids[np.argmax(cons_probs)]    
                    
                    f.write(file_name.split('/')[-1] + ',' + str(subject_id) + ',' + str(box.xmin) + ',' + str(box.ymin) + ',')
                    f.write(str(box.xmax - box.xmin) + ',' + str(box.ymax - box.ymin) + ',' + str(box.get_score()) + '\n')
                    count +=1

                # Check exception.
                if len(boxes) == 0:
                    continue

                # Draw bounding boxes on the image using labels.
                image = draw_boxes_v2(image_o, boxes, self.hps['face_conf_th']) 
         
                # Write the image with bounding boxes to file.
                print('Save ' + file_name[:-4] + '_detected' + file_name[-4:])
                imsave(file_name[:-4] + '_detected' + file_name[-4:], (image).astype('uint8'))
        
    def identify(self, images_a, images_c):
        """Identify faces.
        
        Parameters
        ----------
        images_a : 4d numpy array
            Anchor images
        images_c : 4d numpy array
            Comparison images
        
        Returns
        -------
        1d numpy array
            Identification results
        """
        idents = self.model.predict([images_a, images_c])
        idents = np.squeeze(idents) #?
        
        return idents

def main(args):
    """Main.
    
    Parameters
    ----------
    args : argument type 
        Arguments
    """
    hps = {}

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
        hps['step_per_epoch'] = int(args.step_per_epoch)
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
        hps['step_per_epoch'] = int(args.step_per_epoch)
        hps['epochs'] = int(args.epochs) 
        hps['face_conf_th'] = float(args.face_conf_th)
        hps['nms_iou_th'] = float(args.nms_iou_th)
        hps['num_cands'] = int(args.num_cands)
        hps['entropy_th'] = float(args.entropy_th)
        
        model_loading = False if int(args.model_loading) == 0 else True        
        
        # Test.
        fr = FaceReIdentifier(raw_data_path, hps, model_loading)
        
        ts = time.time()
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
    parser.add_argument('--step_per_epoch')
    parser.add_argument('--epochs')
    parser.add_argument('--face_conf_th')
    parser.add_argument('--nms_iou_th')
    parser.add_argument('--num_cands')
    parser.add_argument('--entropy_th')
    parser.add_argument('--model_loading')
    args = parser.parse_args()
    
    main(args)