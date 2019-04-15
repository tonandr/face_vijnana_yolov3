'''
Created on Apr 5, 2019

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import os
import glob
import argparse
import time

import numpy as np
import pandas as pd
import cv2 as cv

from keras.models import Model, load_model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Input, Dense, Conv2D, Lambda
from keras.utils import multi_gpu_model
import keras.backend as K

from space.yolov3_detect import make_yolov3_model, BoundBox, do_nms, correct_yolo_boxes_v2, WeightReader

# Constants.
MULTI_GPU = False
NUM_GPUS = 4

class FaceDetector(object):
    """Face detector to use yolov3."""
    # Constants.
    MODEL_PATH = 'face_detector.hd5'
    OUTPUT_FILE_NAME = 'solution.csv'
    EVALUATION_FILE_NAME = 'eval.csv'
    NORM_IMAGE_SIZE = 416
    CELL_SIZE = 8
    TARGET_FEATURE_DIM = 6

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
        self.cell_image_size = self.NORM_IMAGE_SIZE // self.CELL_SIZE # ?
        
        if model_loading: 
            self.model = load_model(os.path.join(self.MODEL_PATH))
        else:
            # Design the face detector model.
            # Input.
            input = Input(shape=(hps['x_size'], hps['y_size'], 3), name='input')

            # Load yolov3 as the base model.
            base = self.YOLOV3Base 
            
            # Get 8x8x6 target features. #?
            x = base(input) # Linear?
            output = Conv2D(filters=hps['num_filters']
                       , activation='relu' #?
                       , kernel_size=(1, 1)
                       , padding='same'
                       , name='output')(x)

            if MULTI_GPU:
                self.model = multi_gpu_model(Model(inputs=[input], outputs=[output])
                                                   , gpus = NUM_GPUS)
            else:
                self.model = Model(inputs=[input], outputs=[output])
            
            self.model.compile(optimizer='adam', loss='mse') #?
            self.model.summary()

    @property
    def YOLOV3Base(self):
        """Get yolov3 as a base model.
        
        Returns
        -------
        Model of Keras
            Partial yolo3 model from the input layer to the leaky_74 layer
        """
        yolov3 = make_yolov3_model()
        
        # load the weights trained on COCO into the model.
        weight_reader = WeightReader('yolov3.weights')
        weight_reader.load_weights(yolov3)
        
        base = Model(inputs=[yolov3.input], outputs=[yolov3.get_layer('leaky_74')])
        return base
        
    def train(self):
        """Train face detector."""
        # Get the generator of training data.
        tr_gen = self._get_training_generator()
        
        self.model.fit_generator(tr_gen
                      , steps_per_epoch=self.hps['step_per_epoch']                  
                      , epochs=self.hps['epochs']
                      , verbose=1
                      , max_queue_size=10
                      , workers=1
                      , use_multiprocessing=False)
    
    def test(self, test_path):
        """Test.
        
        Parameters
        ----------
        test_path : string
            Testing directory. 
        """
        file_names = glob.glob(test_path, '*.jpg')
                
        # Detect faces and save results.
        with open(self.OUTPUT_FILE_NAME, 'w') as f:
            for file_name in file_names:
                # Load an image.
                image = cv.imread(os.path.join(test_path, file_name))       
                image = image[np.newaxis, :]
                image_size = [(image.shape[1], image.shape[2])]    
                    
                # Detect faces.
                boxes = self.detect(image, image_size)
                count = 1
                
                for box in boxes:
                    if count > 60:
                        break
                    
                    f.write(file_name + ',' + str(box.xmin) + ',' + str(box.ymin) + ',')
                    f.write(str(box.xmax - box.xmin) + ',' + str(box.ymax - box.ymin) + ',' + str(box.score) + '\n')
                    count +=1
                            
    def _get_testing_generator(self, test_path):
        """Get a testing data generator.
        
        Parameters
        ----------
        test_path : string
            Testing directory.
            
        Returns
        -------
        generator
            ({'input': image})
        """
        file_names = glob.glob(test_path, '*.jpg')
        
        for file_name in file_names:
            # Load an image.
            image = cv.imread(os.path.join(test_path, file_name))
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
                w_p = self.TARGET_FEATURE_DIM
                h_p = np.floor(h / w * self.TARGET_FEATURE_DIM)
                pad = self.TARGET_FEATURE_DIM - h_p
                
                if pad % 2 == 0:
                    pad_t = pad // 2
                    pad_b = pad // 2
                else:
                    pad_t = pad // 2
                    pad_b = pad // 2 + 1

                image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
            else:
                h_p = self.TARGET_FEATURE_DIM
                w_p = np.floor(w / h * self.TARGET_FEATURE_DIM)
                pad = self.TARGET_FEATURE_DIM - w_p
                
                if pad % 2 == 0:
                    pad_l = pad // 2
                    pad_r = pad // 2
                else:
                    pad_l = pad // 2
                    pad_r = pad // 2 + 1                
                
                image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?   
            
            image = image[np.newaxis, ...]                  
                                     
            yield ({'input': image})    
    
    def _get_training_generator(self):
        """Get a training data generator.
        
        Returns
        -------
        generator
            ({'input': image, 'output': gtTensor})
        """
        # Get ground truth.
        gt_df = pd.read_csv(os.path.join(self.rawDataPath, 'protocol_v2', 'training.csv'))
        gt_df_g = gt_df.groupby('FILE')
        file_names = list(gt_df_g.groups.keys())
        
        for file_name in file_names:
            df = gt_df_g.get_group(file_name)
            
            # Load an image.
            image = cv.imread(os.path.join(self.rawDataPath, 'training', file_name))
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
                w_p = self.TARGET_FEATURE_DIM
                h_p = np.floor(h / w * self.TARGET_FEATURE_DIM)
                pad = self.TARGET_FEATURE_DIM - h_p
                
                if pad % 2 == 0:
                    pad_t = pad // 2
                    pad_b = pad // 2
                else:
                    pad_t = pad // 2
                    pad_b = pad // 2 + 1

                image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                image = cv.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?  
            else:
                h_p = self.TARGET_FEATURE_DIM
                w_p = np.floor(w / h * self.TARGET_FEATURE_DIM)
                pad = self.TARGET_FEATURE_DIM - w_p
                
                if pad % 2 == 0:
                    pad_l = pad // 2
                    pad_r = pad // 2
                else:
                    pad_l = pad // 2
                    pad_r = pad // 2 + 1                
                
                image = cv.resize(image, (w_p, h_p), interpolation=cv.INTER_CUBIC)
                image = cv.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv.BORDER_CONSTANT, value=[0, 0, 0]) # 416x416?   
            
            image = image[np.newaxis, ...]                  

            # Create a ground truth bound box tensor (8x8x6).
            gt_tensor = np.zeros(shape=(1, self.CELL_Y_SIZE, self.CELL_X_SIZE, self.TARGET_FEATURE_DIM))
            
            for i in range(df.shape[0]):
                # Calculate a target feature tensor according to the ratio of width, height.
                # Calculate a transformed raw bound box.
                x1 = np.floor(df.loc[i, 'FACE_X'])
                y1 = np.floor(df.loc[i, 'FACE_Y'])
                x2 = x1 + np.floor(df.loc[i, 'FACE_WIDTH']) - 1
                y2 = y1 + np.floor(df.loc[i, 'FACE_HEIGHT']) - 1
                wb = x2 - x1 + 1
                hb = y2 - y1 + 1
                
                if w >= h:
                    x1_p = np.floor(x1 / w * self.NORM_IMAGE_SIZE)
                    y1_p = np.floor(y1 / w * self.NORM_IMAGE_SIZE) + pad_t
                    x2_p = np.floor(x2 / w * self.NORM_IMAGE_SIZE)
                    y2_p = np.floor(y2 / w * self.NORM_IMAGE_SIZE) + pad_t
                else:
                    x1_p = np.floor(x1 / h * self.NORM_IMAGE_SIZE) + pad_l
                    y1_p = np.floor(y1 / h * self.NORM_IMAGE_SIZE)
                    x2_p = np.floor(x2 / h * self.NORM_IMAGE_SIZE) + pad_l
                    y2_p = np.floor(y2 / h * self.NORM_IMAGE_SIZE)                   
                
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
                gt_tensor[0, cy, cx, 0] = 1.
                gt_tensor[0, cy, cx, 1] = bx_p
                gt_tensor[0, cy, cx, 2] = by_p
                gt_tensor[0, cy, cx, 3] = bw_p
                gt_tensor[0, cy, cx, 4] = bh_p
                gt_tensor[0, cy, cx, 5] = 1.
                                     
            yield ({'input': image, 'output': gt_tensor})
 
    def detect_with_generator(self, image_gen, image_sizes):
        """Detect faces with generator.
        
        Parameters
        ----------
        image_gen: image generator with 4d numpy array
            Images
        image_sizes: tuple list
            Image x, y size list
        
        Returns
        -------
        list
            Face candidate bounding boxs
        """
        # Get face region candidates.
        face_cands = self.model.predict_generator(image_gen
                                                  , steps=None
                                                  , max_queue_size=10
                                                  , workers=1
                                                  , use_multiprocessing=False
                                                  , verbose=1) # 8x8x6. #?
        
        # Eliminate candidates less than the face confidence threshold.
        cell_x_size = self.hps['x_size'] // face_cands.shape[1] # self.hps['x_size'] % face_cands.shape[0] must be zero.
        cell_y_size = self.hps['y_size'] // face_cands.shape[0]
        face_cand_bboxes = []
        face_cands[...,-1] = face_cands[...,0] * face_cands[...,-1] #?

        for i in range(face_cands.shape[0]):
            for j in range(face_cands.shape[1]):
                if face_cands[i, j, -1] >= self.hps['face_conf_th']:
                    # Calculate values of a bound box.
                    objectness = face_cands[i, j, 0]
                    bx = face_cands[i, j, 1]
                    by = face_cands[i, j, 2]
                    bw = face_cands[i, j, 3]
                    bh = face_cands[i, j, 4]
                    score = face_cands[i, j, 5]
                                                            
                    # Convert a raw ratio bound box into a bound box for practical image size scale.  
                    px = np.min([np.floor(bx * cell_x_size), cell_x_size - 1]) + cell_x_size * j
                    py = np.min([np.floor(by * cell_y_size), cell_y_size - 1]) + cell_y_size * i
                    pw = np.min([bw * self.hps['x_size'], self.hps['x_size']]) 
                    ph = np.min([bh * self.hps['y_size'], self.hps['y_size']])
                    
                    # Calculate xmin, ymin, xmax, ymax positions for the original image size.
                    xmin = np.max([px - np.floor(pw / 2), 0])
                    ymin = np.max([py - np.floor(ph / 2), 0])
                    xmax = np.max([px + np.floor(pw / 2), self.hps['x_size'] - 1])
                    ymax = np.max([py + np.floor(ph / 2), self.hps['y_size'] - 1])
                    
                    # Get a bound box.
                    face_cand_bbox = BoundBox(xmin, ymin, xmax, ymax, objectness, score)
                    face_cand_bboxes.append(face_cand_bbox)
        
        # Conduct non-max suppression.
        face_cand_bboxes = do_nms(face_cand_bboxes, self.hps['nms_iou_th'])
        
        # Get high face score candidates.
        scores = [face_cand_bbox.score for face_cand_bbox in face_cand_bboxes]
        sorted_index = np.argsort(scores)
        
        face_cand_bboxes = [face_cand_bbox[sorted_index[i]] \
                       for i in range(self.hps['num_cands']) if i < len(scores)]

        # correct the sizes of the bounding boxes
        correct_yolo_boxes_v2(face_cand_bboxes
                           , image_sizes
                           , self.NORM_IMAGE_SIZE
                           , self.NORM_IMAGE_SIZE)

        return face_cand_bboxes
        
    def detect(self, images, image_sizes):
        """Detect faces.
        
        Parameters
        ----------
        images: 4d numpy array
            Images
        image_sizes: tuple list
            Image x, y size list
                    
        Returns
        -------
        list
            Face candidate bounding boxs
        """
        # Get face region candidates.
        face_cands = self.model.predict(images) # 8x8x6. #?
        
        # Eliminate candidates less than the face confidence threshold.
        cell_x_size = self.hps['x_size'] // face_cands.shape[1] # self.hps['x_size'] % face_cands.shape[0] must be zero.
        cell_y_size = self.hps['y_size'] // face_cands.shape[0]
        face_cand_bboxes = []
        face_cands[...,-1] = face_cands[...,0] * face_cands[...,-1] #?

        for i in range(face_cands.shape[0]):
            for j in range(face_cands.shape[1]):
                if face_cands[i, j, -1] >= self.hps['face_conf_th']:
                    # Calculate values of a bound box.
                    objectness = face_cands[i, j, 0]
                    bx = face_cands[i, j, 1]
                    by = face_cands[i, j, 2]
                    bw = face_cands[i, j, 3]
                    bh = face_cands[i, j, 4]
                    score = face_cands[i, j, 5]
                                                            
                    # Convert a raw ratio bound box into a bound box for practical image size scale.  
                    px = np.min([np.floor(bx * cell_x_size), cell_x_size - 1]) + cell_x_size * j
                    py = np.min([np.floor(by * cell_y_size), cell_y_size - 1]) + cell_y_size * i
                    pw = np.min([bw * self.hps['x_size'], self.hps['x_size']]) 
                    ph = np.min([bh * self.hps['y_size'], self.hps['y_size']])
                    
                    # Calculate xmin, ymin, xmax, ymax positions for the original image size.
                    xmin = np.max([px - np.floor(pw / 2), 0])
                    ymin = np.max([py - np.floor(ph / 2), 0])
                    xmax = np.max([px + np.floor(pw / 2), self.hps['x_size'] - 1])
                    ymax = np.max([py + np.floor(ph / 2), self.hps['y_size'] - 1])
                    
                    # Get a bound box.
                    face_cand_bbox = BoundBox(xmin, ymin, xmax, ymax, objectness, score)
                    face_cand_bboxes.append(face_cand_bbox)
        
        # Conduct non-max suppression.
        face_cand_bboxes = do_nms(face_cand_bboxes, self.hps['nms_iou_th'])
        
        # Get high face score candidates.
        scores = [face_cand_bbox.score for face_cand_bbox in face_cand_bboxes]
        sorted_index = np.argsort(scores)
        
        face_cand_bboxes = [face_cand_bbox[sorted_index[i]] \
                       for i in range(self.hps['num_cands']) if i < len(scores)]
        
        # correct the sizes of the bounding boxes
        correct_yolo_boxes_v2(face_cand_bboxes
                           , image_sizes
                           , self.NORM_IMAGE_SIZE
                           , self.NORM_IMAGE_SIZE)

        return face_cand_bboxes
                 
def main(args):
    """Main.
    
    Parameters
    ----------
    args : argument type 
        Arguments
    """
    hps = {}
            
    if args.mode == 'train':
        # Get arguments.
        raw_data_path = args.raw_data_path
      
        # hps.
        hps['x_size'] = int(args.x_size)    
        hps['y_size'] = int(args.y_size)
        hps['num_filters'] = int(args.num_filters)
        hps['step_per_epoch'] = int(args.step_per_epoch)
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
    elif args.mode == 'test':
        # Get arguments.
        raw_data_path = args.raw_data_path
        test_path = os.path.join(raw_data_path, 'img')
      
        # hps.
        hps['x_size'] = int(args.x_size)    
        hps['y_size'] = int(args.y_size)
        hps['num_filters'] = int(args.num_filters)
        hps['step_per_epoch'] = int(args.step_per_epoch)
        hps['epochs'] = int(args.epochs) 
        hps['face_conf_th'] = float(args.face_conf_th)
        hps['nms_iou_th'] = float(args.nms_iou_th)
        hps['num_cands'] = int(args.num_cands)
        
        model_loading = False if int(args.model_loading) == 0 else True        
        
        # Test.
        fd = FaceDetector(raw_data_path, hps, model_loading)
        
        ts = time.time()
        fd.test(test_path)
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
        
if __name__ == '__main__':
    
    # Parse arguments.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode')
    parser.add_argument('--raw_data_path')
    parser.add_argument('--x_size')
    parser.add_argument('--y_size')
    parser.add_argument('--num_filters')
    parser.add_argument('--step_per_epoch')
    parser.add_argument('--epochs')
    parser.add_argument('--face_conf_th')
    parser.add_argument('--nms_iou_th')
    parser.add_argument('--num_cands')
    parser.add_argument('--model_loading')
    args = parser.parse_args()
    
    main(args)
    pass