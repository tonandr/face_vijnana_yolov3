'''
Created on Apr 5, 2019

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import numpy as np
import os

from keras.models import Model, load_model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Input, Dense, Conv2D, Lambda
from keras.utils import multi_gpu_model
import keras.backend as K

from .yolov3_detect import make_yolov3_model, BoundBox, bbox_iou, do_nms

# Constants.
MULTI_GPU = False
NUM_GPUS = 4

class FaceDetector(object):
    """Face detector to use yolov3."""
    # Constants.
    MODEL_PATH = 'face_detector.hd5'

    def __init__(self, hps, model_loading):
        """
        Parameters
        ----------
        hps : dictionary
            Hyper-parameters
        model_loading : boolean 
            Face detection model loading flag
        """
        # Initialize.
        self.hps = hps
        self.model_loading = model_loading
        
        if model_loading: 
            self.model = load_model(os.path.join(self.MODEL_PATH))
        else:
            # Design the face detector model.
            # Input.
            input = Input(shape=(hps['x_size', hps['y_size'], 3]))

            # Load yolov3 as the base model.
            base = self.YOLOV3Base 
            
            # Get 8x8x6 target features. #?
            x = base(input) # Linear?
            output = Conv2D(filters=hps['num_filters']
                       , activation='leakyrelu' # leaky relu?
                       , kernel_size=(1,1)
                       , padding='same')(x)

            if MULTI_GPU:
                self.model = multi_gpu_model(Model(inputs=[input], outputs=[output])
                                                   , gpus = NUM_GPUS)
            else:
                self.model = Model(inputs=[input], outputs=[output])
            
            self.model.compile(optimizer='adam', loss='mse')

    @property
    def YOLOV3Base(self):
        """Get yolov3 as a base model.
        
        Returns
        -------
        Model of Keras
            Partial yolo3 model from the input layer to the leaky_74 layer
        """
        yolov3 = make_yolov3_model()
        base = Model(inputs=[yolov3.input()], outputs=[yolov3.get_layer('leaky_74')])
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
     
    def detect(self, images):
        """Detect faces.
        
        Parameters
        ----------
        images: 4d numpy array
            Images
        
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
        
        return face_cand_bboxes
                 
if __name__ == '__main__':
    pass