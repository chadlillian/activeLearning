import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf
import keras
import tensorflow_probability as tfp
tfd = tfp.distributions

from keras_tqdm import TQDMNotebookCallback

class ann:
    def __init__(self):#,nodesIn,nodesOut):
        self.epochs = 100
        self.verbose = 1

    def setInputsOutputs(self,nodesIn,nodesOut):
        trainSamples = 10
        inferSamples = 101

        modelVar = tf.keras.Sequential([
            tf.keras.layers.InputLayer(nodesIn),
            tf.keras.layers.Dense(10,activation='sigmoid'),
            tf.keras.layers.Dense(nodesOut,'sigmoid'),
            tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(nodesOut)),
            tfp.layers.MultivariateNormalTriL(nodesOut),
        ])

        modelSampleTrain = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(tf.transpose(x.sample(trainSamples),perm=[1,0,2]),axis=1)),
            #tf.keras.layers.Reshape([nodesOut,nodesOut])
        ])

        modelSampleInfer = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(tf.transpose(x.sample(inferSamples),perm=[1,0,2]),axis=1)),
            #tf.keras.layers.Reshape([nodesOut,nodesOut])
        ])

        modelSampleInferStd = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(tf.math.reduce_std(x.sample(inferSamples),axis=0),axis=1)),
        ])

        self.modelTrain = tf.keras.Model(inputs=modelVar.inputs,outputs=modelSampleTrain(modelVar.outputs[0]))
        self.modelInfer = tf.keras.Model(inputs=modelVar.inputs,outputs=modelSampleInfer(modelVar.outputs[0]))
        self.modelInferStd = tf.keras.Model(inputs=modelVar.inputs,outputs=modelSampleInferStd(modelVar.outputs[0]))

        self.modelTrain.compile(optimizer='Adam',loss='MSE',metrics=['accuracy'])
    
    def train(self,x,y):#,epochs,verbose=0):
        #cb = TQDMNotebookCallback()
        #cb.on_train_batch_begin = cb.on_batch_begin
        #cb.on_train_batch_end = cb.on_batch_end
        self.modelTrain.fit(x=x,y=y,epochs=self.epochs,verbose=self.verbose)
    
    def infer(self,x):
        ret = self.modelInfer.predict(x=x)
        return ret
    
    def var(self,x):
        ret = self.modelInferStd.predict(x=x)
        return ret

    def setEpochs(self,epochs):
        self.epochs = epochs

    def setVerbose(self,verbose):
        self.verbose = verbose
