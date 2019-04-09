'''
Created on Apr 9, 2019

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

class FaceReIdentifier(object):
    """Face re-identifier to use yolov3."""
    # Constants.
    MODEL_PATH = 'face_reidentifier.hd5'

    def __init__(self, hps, model_loading):
        """
        Parameters
        ----------
        hps : dictionary
            Hyper-parameters
        model_loading : boolean 
            Face re-identification model loading flag
        """
        # Initialize.
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