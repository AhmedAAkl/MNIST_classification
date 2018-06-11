# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 21:01:04 2017

@author: A.Akl
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

model_dir="/mnist_learn/"

def cnn_model_fn(features,labels,mode):
        
    # input layer    
    input_layer = tf.reshape(features["x"],[-1,28,28,1])
    
    # -1 for the batch_size, to ensure that is a hyperparameter 
    # 28 for width , 28 for height , 1 for color channels which is monochrome (1)
    #convolutional layer #1
    
    conv1 = tf.layers.conv2d(
            inputs= input_layer,
            filters=32,
            kernel_size=[5,5],
            padding='same',
            activation=tf.nn.relu)
    
    # the output will be [ -1 , 28,28 , 32] the same width and height but, 32 feature map
    # bacause of the filters applied is 32
    # Pooling layer #1
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)
    
    # output will be reduced by 50% because of [2,2] maxpooling : [-1,14,14,32]    
    # Convolutional layer #2 and Pooling layer #2
    
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters = 64,
            kernel_size = [5,5],
            padding = 'same',
            activation= tf.nn.relu)
    
    # output will be [-1,14,14,64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)
    
    # after pooling will be [-1,7,7,64]
    
    # Dense Layer
    # we must flat our features
    
    pool2_flat = tf.reshape(pool2,[-1,7*7*64])
    
    dense = tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(inputs=dense,rate=0.4,training=mode == tf.estimator.ModeKeys.TRAIN)
        
    # Logits layer
    # last layer consists of 10 neurons one for each class 0-9
    
    logits = tf.layers.dense(inputs=dropout,units=10)
    
    predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits,axis = 1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits,name="softmax_tensor")                    
            }
                
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
            
    # Calculate the loss for both Train and EVal
    
    oneHot_lables = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=10)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=oneHot_lables,logits=logits)
    
    # Configure the taining op for Train Mode
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        training_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op = training_op)
    
    # Add evaluation metrics for EVAL mode
    
    eval_metric_op = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels,predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_op)

def main(unused_argv):
    
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels,dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.train.labels,dtype=np.int32)
    
    # create the estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir=model_dir)
    
    # set up logging for prediction
    
    tensors_to_log = {"probabilities":"softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=10)
                   
    # Train the model
        
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":train_data},
            y= train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True
            )
    
    mnist_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook]
            )

    # Evaluate the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":eval_data},
            y= eval_labels,
            num_epochs=1,
            shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
if __name__ == "__main__":
    tf.app.run()        
