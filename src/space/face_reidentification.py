'''
Created on Apr 9, 2019

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import os
import glob
import argparse
import time

import numpy as np
import pandas as pd
import cv2 as cv
from skimage.io import imread, imsave

from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Lambda, ZeroPadding2D, LeakyReLU
from keras.layers.merge import add, concatenate
from keras.utils import multi_gpu_model
from keras.utils.data_utils import Sequence
import keras.backend as K

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
            self.db_g = self.db.groupby('subject_id')
            self.img_pairs = []
            self.df_non_id = self.db_g.get_groups(-1)
            
            for i in self.db_g.groups.keys():
                if i == -1:
                    continue
                
                df = self.db_g.get_group(i)
                
                
                
        def __len__(self):
            return int(np.floor(len(self.file_names) / self.batch_size))
        
        def __getitem__(self, index):
            # Check the last index.
            # TODO
            
            images = []
            gt_tensors = []
            
            for bi in range(index * self.batch_size, (index + 1) * self.batch_size):
                file_name = self.file_names[bi] 
                #if DEBUG: print(file_name )
                
                df = self.gt_df_g.get_group(file_name)
                df.index = range(df.shape[0])
                
                # Load an image.
                image = cv.imread(os.path.join(self.raw_data_path, file_name))
                                        
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
            input_a = Input(shape=(hps['x_size', hps['y_size'], 3]))
            input_c = Input(shape=(hps['x_size', hps['y_size'], 3]))

            # Load yolov3 as the base model.
            base = self.YOLOV3Base 
            
            # Get both face features.
            xa = base(input_a) # Linear?
            xc = base(input_c)
            
            # Calculate the difference of both face features.
            xd = Lambda(lambda x: K.sqrt(K.sum(K.pow(x[0] - x[1], 2.))))([xa, xc]) #?
            output = Dense(1, activation='sigmoid')(xd)

            if MULTI_GPU:
                self.model = multi_gpu_model(Model(inputs=[input_a, input_c], outputs=[output])
                                                   , gpus = NUM_GPUS)
            else:
                self.model = Model(inputs=[input_a, input_c], outputs=[output])
            
            self.model.compile(optimizer='adam', loss='binary_crossentropy')

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
        
    def train(self, trGen):
        """Train face detector.
        
        Parameters
        ----------
        trGen : generator
            
            TODO
        """     
        self.model.fit_generator(trGen
                      , steps_per_epoch=self.hps['step_per_epoch']                  
                      , epochs=self.hps['epochs']
                      , verbose=1
                      , max_queue_size=10
                      , workers=1
                      , use_multiprocessing=False)
    
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
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif args.mode == 'test':
        # Get arguments.
        raw_data_path = args.raw_data_path
        output_file_path = args.output_file_path
      
        # hps.
        hps['image_size'] = int(args.image_size) 
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
    parser.add_argument('--lr')
    parser.add_argument('--beta_1')
    parser.add_argument('--beta_2')
    parser.add_argument('--decay')
    parser.add_argument('--step_per_epoch')
    parser.add_argument('--epochs')
    parser.add_argument('--face_conf_th')
    parser.add_argument('--nms_iou_th')
    parser.add_argument('--num_cands')
    parser.add_argument('--model_loading')
    args = parser.parse_args()
    
    main(args)