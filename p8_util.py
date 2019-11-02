'''This file implements all utility functions involved in ADANET solution.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import numpy as np
import pandas as pd
import functools
import os
import shutil

import matplotlib.pyplot as plt

import tensorflow as tf
import adanet
from adanet.examples import simple_dnn

from six.moves import range
import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout

from sklearn import preprocessing

from adanet.ensemble import MixtureWeightType
from adanet.ensemble.weighted import ComplexityRegularizedEnsembler
from sklearn.model_selection import train_test_split

import nltk
import keras


import p5_util
import p8_util_config


import NNAdaNetBuilder
#import p8_util_config

RANDOM_SEED = 42
LOG_DIR = './tmp/models'

_NUM_LAYERS_KEY = "num_layers"
#FEATURES_KEY = 'images'
FEATURES_KEY = p8_util_config.FEATURES_KEY
IS_DEBUG = p8_util_config.IS_DEBUG


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_sample(X, y, ratio=-1) :
    if ratio == -1 :
        range_value = len(X)
    else :
        range_value = int(len(X)*ratio)
    return X[:range_value], y[:range_value]
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def clean_X_label(X, label) :
    
    ser_ = pd.Series(X)

    list_index = [i for i in ser_.index if len(ser_[i])==0]
    
    ser_.drop(list_index, inplace=True)
    list_to_clean_1 = ser_.tolist()
    
    print("Cleaned empty text = {}".format(len(list_index)))
    ser_ = pd.Series(label)
    ser_.drop(list_index, inplace=True)

    list_to_clean_2 = ser_.tolist()
    
    return list_to_clean_1, list_to_clean_2
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_adanet_estimator(subnetwork_generator):
    '''Build ADANET estimator.

    Input : 
        *   subnetwork_generator : subnetwork generator that generates a candidate 
            that is provided to ADANET weaklearner algorithm.
        *   nn_type : type of neural network candidate to be generated. It is used 
            to select mixture weights type : scalar, vector or matrix.
        *   feature_shape : shape of features.

    Output : 
        *   adanet_estimator : ADANET estimator instance of adanet.Estimator class.
    '''
    
    output_dir = './tmp/adanet'
    
    #---------------------------------------------------------------------------
    # Get parameters from configuration file.
    #---------------------------------------------------------------------------
    dict_adanet_config = p8_util_config.dict_adanet_config
    feature_shape = dict_adanet_config['adanet_feature_shape']
    
    dict_nn_layer_config = dict_adanet_config['adanet_nn_layer_config']
    nn_type = dict_nn_layer_config['nn_type']
    
    #---------------------------------------------------------------------------
    # Fixe mixture weights
    #---------------------------------------------------------------------------
    if 'RNN' == nn_type :
        mixture_weight_type=MixtureWeightType.VECTOR
    else :
        mixture_weight_type=MixtureWeightType.MATRIX

    ensembler = ComplexityRegularizedEnsembler(mixture_weight_type=mixture_weight_type\
                                            , adanet_lambda=p8_util_config.dict_adanet_config['adanet_lambda'])
    dataset_type = p8_util_config.DATASET_TYPE
    input_fn_param={'num_epochs':p8_util_config.NUM_EPOCHS,\
                    'batch_size':p8_util_config.BATCH_SIZE,\
                    'feature_shape': feature_shape,\
                    'dataset_type': dataset_type
                   }

    train_input_fn=input_fn_2("train", input_fn_param)
    
    model_name = build_model_name(nn_type)
    
    adanet_estimator_config, output_dir_log= make_config(model_name\
                                        , output_dir=output_dir\
                                        , is_restored=False)
                                        
    nb_class = p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_logit_dimension']
        
    feature_columns, loss_reduction, tf_head \
        = get_tf_head(feature_shape, nb_class, nn_type=nn_type, feature_shape=feature_shape)
    
    p8_util_config.dict_adanet_config['adanet_feature_columns'] = feature_columns    
    p8_util_config.dict_adanet_config['adanet_feature_shape'] = feature_shape
    
    adanet_estimator = adanet.Estimator(
        head=p8_util_config.dict_adanet_config['adanet_tf_head'],
        subnetwork_generator=subnetwork_generator,
        ensemblers        = [ensembler],

        max_iteration_steps=p8_util_config.ADANET_MAX_ITERATION_STEPS,

        evaluator=adanet.Evaluator(
            input_fn=train_input_fn,
            steps=None),
        config=  adanet_estimator_config)
    return adanet_estimator, output_dir_log
#-------------------------------------------------------------------------------
    

    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def create_nn_builder( output_dir, layer_num=None):
    '''Creates an NNAdaNetBuilder object from configuration defined into file 
    p8_util_config.
    This function may be used for Baseline when testing NN networks fixing 
    number of layers from p8_util_config.py file.
    Input :
        * param_feature_shape : shape of the features. This parameter is unknowned 
        from configuration prior to dataset read.
        * output_dir : Directory where logs are dumped.
        * layer_num : number of layers used to build (deep) neural network. 
            If  value is None, then number of layers value is picked from configration file 
            to instantiate NNAdaNetBuilder object. 
            
            If value is not None, then this means that NNAdaNetBuilder object is 
            instantiated from Adanet Algorithm. In this case, dictionaries in 
            configuration file are updated with this value.
    Output :
        * Object of type NNAdaNetBuilder
    
    '''
    param_feature_shape = p8_util_config.dict_adanet_config['adanet_feature_shape']
    if layer_num is None :
        if p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_type'] == 'CNN' :
            if p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['cnn_dense_layer_num'] is None :
                layer_num = p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['cnn_conv_layer_num']
            elif p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['cnn_conv_layer_num'] is None :
                layer_num = p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['cnn_dense_layer_num']
            else :
                pass            
        elif p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_type'] == 'RNN' :
            layer_num = p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['rnn_layer_num']
        elif p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_type'] == 'DNN' :
            layer_num = p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['dnn_layer_num']
    else :
        pass

    if IS_DEBUG is True :
        print("\n*** Number of convolutional layers= {}".format(layer_num))
    
    oNNAdaNetBuilder = NNAdaNetBuilder.NNAdaNetBuilder(p8_util_config.dict_adanet_config, num_layers=layer_num)
    oNNAdaNetBuilder.feature_shape = param_feature_shape
    
    # Create internaly log output dir and classsifier.
    oNNAdaNetBuilder.output_dir = output_dir

    #---------------------------------------------------------------------------
    # Dictionaries from configuration file are updated with this number of layers.
    #---------------------------------------------------------------------------
    if p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_type'] == 'CNN' :
        if p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['cnn_dense_layer_num'] is not None :
            p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['cnn_dense_layer_num'] = layer_num
        elif p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['cnn_conv_layer_num'] is not None :
            p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['cnn_conv_layer_num'] = layer_num
        else :
            pass
    elif p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_type'] == 'RNN' :
        p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['rnn_layer_num']=layer_num
    elif p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_type'] == 'DNN' :
        p8_util_config.dict_adanet_config['adanet_nn_layer_config']['nn_layer_config']['dnn_layer_num']=layer_num

    if IS_DEBUG is True :
        print("\nMax steps= {} / Number of EPOCH={}".format(p8_util_config.TRAIN_STEPS,p8_util_config.NUM_EPOCHS))
    return oNNAdaNetBuilder
        
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_tf_head(tuple_dimension, nClasses, nn_type='CNN', feature_shape=None) :
    '''Build a multi-class (tensorflow) head along with tensorflow features 
    columns and loss computation type.
    '''
    #FEATURES_KEY = feature_key
    list_dimension = [dimension for dimension in tuple_dimension]
    
    if feature_shape is None :
        if(len(tuple_dimension) > 2) :
            channel = tuple_dimension[2]
        else :
            channel = 1
        w_size = tuple_dimension[0]
        h_size = tuple_dimension[1]
        feature_shape = [w_size, h_size, channel]
    else :
        pass

    if IS_DEBUG is True :
        print("\n*** get_tf_head() : feature shape= {}".format(feature_shape))
        
    #---------------------------------------------------------------------------
    # Build numeric features colums.
    #---------------------------------------------------------------------------
    my_feature_columns = [tf.feature_column.numeric_column(FEATURES_KEY\
                                                , shape=feature_shape)]
    if IS_DEBUG is True :
        print("\n*** get_tf_head() : feature columns= {}".format(my_feature_columns))
    # Some `Estimators` use feature columns for understanding their input features.
    # We will average the losses in each mini-batch when computing gradients.

    #---------------------------------------------------------------------------
    # Select loss reduction type depending on neural network type.
    #---------------------------------------------------------------------------
    if nn_type == 'RNN' :
        loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE
    else :
        loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE
    
    # A `Head` instance defines the loss function and metrics for `Estimators`.
    # Tells Tensorfow how to compute loss function and metrics
    if nClasses > 1 :
        tf_head = tf.contrib.estimator.multi_class_head(nClasses\
                                          , loss_reduction=loss_reduction)
    else :
        tf_head = tf.contrib.estimator.logistic_regression_head(weight_column=None,
        loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE, name=None)

    return my_feature_columns, loss_reduction, tf_head
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#   
#-------------------------------------------------------------------------------
def custom_model_fn( features, labels, mode, params ): 
    '''This function implements training, evaluation and prediction for customized 
    estimator.
    It also implements the predictor model.
    It is designed in the context of a customized Estimator.

    This function is invoked form Estimator's train, predict and evaluate methods.
    Input :
        features : batch of features provided from input function.
        labels : batch labels provided from input function.
        mode : provided by input function, mode discriminate train, evaluation and prediction steps.
        params : parameters used in this function, passed to Estimator by higher level call.
    Output:
        tf.estimator.EstimatorSpec thagt will be passed to an Estimator.
        
    '''
    #-----------------------------------------------------------------------------
    # Get from parameters object that is used form Adanet to build NN sub-networks.
    #-----------------------------------------------------------------------------
    net_builder = params['net_builder']
    nn_type = net_builder._nn_type
    
    feature_shape = net_builder.feature_shape
    logits_dimension = net_builder.nb_class
    
    input_layer = tf.feature_column.input_layer(features=features\
    , feature_columns=net_builder.feature_columns)
    is_training = False

    with tf.name_scope(nn_type):
        if IS_DEBUG is True :
                print("\n*** custom_model_fn() : input_layer shape= {} / Labels shape= {}"\
                .format(input_layer.shape, labels.shape))

        if mode == tf.estimator.ModeKeys.TRAIN :
            is_training = True

        #-----------------------------------------------------------------------
        # Predictions are computed for a batch
        #-----------------------------------------------------------------------
        if nn_type == 'CNN' or  nn_type == 'CNNBase' :
            _, logits = net_builder._build_cnn_subnetwork(input_layer, features\
                                                        , logits_dimension\
                                                        , is_training)
        elif nn_type == 'RNN' : 
            rnn_cell_type = net_builder._dict_rnn_layer_config['rnn_cell_type']
            _, logits = net_builder._build_rnn_subnetwork(input_layer, features\
                                                        , logits_dimension\
                                                        , is_training\
                                                        , rnn_cell_type = rnn_cell_type)
        elif nn_type == 'DNN' :
            _, logits = net_builder._build_dnn_subnetwork(input_layer, features\
                                                        , logits_dimension\
                                                        , is_training)
        else : 
            print("\n*** ERROR : Network type= {} NOT YET SUPPORTED!".format(nn_type))
            return None
        # Returns the index from logits for which logits has the maximum value.
        
                
        if IS_DEBUG is True :
            print("\n*** custom_model_fn() : logits shape= {} / labels shape= {}"\
            .format(logits.shape, labels.shape))

        
        #-----------------------------------------------------------------------
        # Loss is computed
        #-----------------------------------------------------------------------
        if p8_util_config.IS_LABEL_ENCODED is True:
            labels = tf.one_hot(labels, depth=3)
            
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            tf.summary.scalar('Loss', loss)

        if mode == tf.estimator.ModeKeys.TRAIN :
            with tf.name_scope('Train'):
                #---------------------------------------------------------------
                # Gradient descent is computed 
                #---------------------------------------------------------------
                optimizer = net_builder.optimizer
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                
                #---------------------------------------------------------------
                # Accuracy is computed
                #---------------------------------------------------------------
                tf_label_arg_max = tf.argmax(labels,1)
                accuracy, accuracy_op = tf.metrics.accuracy(labels=tf_label_arg_max\
                            , predictions=tf.argmax(logits,1)\
                            , name=nn_type+'Train_accuracy')
                            
                tf.summary.scalar(nn_type+'Train_Accuracy', accuracy)
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        elif mode ==  tf.estimator.ModeKeys.EVAL :
            with tf.name_scope('Eval'):
                # Compute accuracy from tf metrics package. It compares true values (labels) against
                # predicted one (predicted_classes)
                accuracy, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(labels,1)\
                            , predictions=tf.argmax(logits,1)\
                            , name=nn_type+'Eval_accuracy')
                tf.summary.scalar(nn_type+'_Eval_accuracy', accuracy)
                metrics = {nn_type+'_Eval_accuracy': (accuracy, accuracy_op)}

                
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(logits),
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        else :
            print("\n*** ERROR : custom_model_fn() : mode= {} is unknwoned!".format(mode))
            pass
    return None
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#   
#-------------------------------------------------------------------------------
def load_data_mnist():
    '''Defaut path to where MNIST dataset is located (after download) :
    ~/.keras/datasets/mnist.npz
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #print(y_train.shape, y_train.min(), y_train.max())
    #---------------------------------------------------------------------------
    # Make data separation between valid and train data.
    #---------------------------------------------------------------------------
    x_valid_num = x_test.shape[0]//2
    part = x_train.shape[0]-x_valid_num

    x_valid = x_train[part:,:]
    y_valid = y_train[part:]
    x_train = x_train[:part,:]
    y_train = y_train[:part]
    
    return x_train, x_valid, x_test, y_train, y_valid, y_test
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def batch_coloredimage_serial_reshape(x_batch):
    '''Reshape a batch of images allowing an image to be processed as serial data 
    for feefing RNN network.
    
    Each image with initial shape = [w,h,c]
    into a batch of images, each image with retruned shape=[h*c,w]

    This will allow for any image to be serialized as  h*c raws of w columns.
    
    Each raw sized as h*c will feed a deployed cell from RNN network while each 
    cell from RNN will be deployed as w parts.


             +--------------+        +---+----------+
             | B plan       |        |   |          |
          +--------------+  |        | R |          |
          | G plan       |  |        |   |          |
       +--------------+  |  |  ==>   |___|          |
       | R plan       |  |--+        |   |          |
       |              |  |           | G |          |
       |              |--+           |   |          |
       |              |              |___|          |
       +--------------+              |   |          |
                                     |   |          |
                                     | B |          |
                                     |   |          |
                                     +---+----------+
                                       |          |
                                       |          |
                                       |          +---> CELL #W
                                       |
                                       +----> CELL #1                                                                              
    '''
    batch_size = x_batch.shape[0]
    
    w = x_batch.shape[1]
    h = x_batch.shape[2]
    c = x_batch.shape[3]

    #-----------------------------------------------------------------------
    # [batch_size, w, h*c]  --> [batch_size, w, h, c] 
    #-----------------------------------------------------------------------
    x_batch = x_batch.reshape(batch_size,w,-1)

    #-----------------------------------------------------------------------
    # [batch_size, w, h*c]  --> [batch_size, h*c, w ]
    #-----------------------------------------------------------------------
    x_batch = np.transpose(x_batch, [0,2,1])

    return x_batch
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def preprocess_dataset_jigsaw(X_train, X_test) :
    ''' Preprocessing stands for :
        1) Removing stopwords from any document from corpus
        2) Tokenize any document from corpus; each document will be a list of 
        tokens.
        3) Encode any document as Bag Of Words
        4) Fixe the same sequence length for any document.
    '''
    #---------------------------------------------------------------------------
    # Stopwords to be removed
    #---------------------------------------------------------------------------
    nltk.download('stopwords')  

    stop_words = nltk.corpus.stopwords.words('english')

    X_train = [[word for word in list_word if word not in stop_words ] \
    for list_word in X_train]

    X_test = [[word for word in list_word if word not in stop_words ] \
    for list_word in X_test]
    
    #---------------------------------------------------------------------------
    # Get max length padding from Series quantile; 
    # For getting max length, all corpus is used : X_test and X_train.
    #---------------------------------------------------------------------------
    X = X_train.copy()
    #X.extend(X_test)
    
    ser_train = pd.Series(X)
    ser_train_len = ser_train.apply(lambda x : len(x))
    max_length = int(ser_train_len.quantile(0.75))
    
    
    #---------------------------------------------------------------------------
    # Build tokenizer; this last one is fitted from both X_train and X_test 
    # documents.
    #---------------------------------------------------------------------------    
    keras_tokenizer = keras.preprocessing.text.Tokenizer()
    keras_tokenizer.fit_on_texts(X)
    vocab_size = len(keras_tokenizer.word_index) + 1

    #---------------------------------------------------------------------------
    # X_train is encoded in BOW and padded 
    #---------------------------------------------------------------------------
    X_train_encoded = keras_tokenizer.texts_to_sequences(X_train)

    #---------------------------------------------------------------------------
    # Documents from X_train are padded to a max length of max_length words
    #---------------------------------------------------------------------------
    X_train_encoded = keras.preprocessing.sequence.\
    pad_sequences(X_train_encoded, maxlen=max_length, padding='post')
    
    #---------------------------------------------------------------------------
    # X_test is encoded in BOW and padded as well as X_train.
    #---------------------------------------------------------------------------
    X_test_encoded = keras_tokenizer.texts_to_sequences(X_test)
    
    # pad documents to a max length of 4 words
    X_test_encoded = keras.preprocessing.sequence.\
    pad_sequences(X_test_encoded, maxlen=max_length, padding='post')

    
    return X_train_encoded, X_test_encoded
    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def load_dataset_jigsaw(sampling_ratio =None) :
    df_test = pd.read_csv('./data/test.csv.zip', compression='zip', header=0,\
     sep=',', quotechar='"')
    df_train = pd.read_csv('./data/train.csv.zip', compression='zip', header=0,\
     sep=',', quotechar='"')

    df_train['comment_text'] = df_train['comment_text'].apply(lambda x : x.lower())
    X_train, X_test, y_train, y_test = \
    train_test_split(df_train['comment_text'],\
    df_train['target'],test_size=0.33, random_state=42)

    print("Train dataset: X = {} Label= {}".format(X_train.shape, y_train.shape))
    print("Test dataset: X = {} Label= {}".format(X_test.shape, y_test.shape))
    if sampling_ratio is None :
        pass
    else :
        X_train, y_train  = get_sample(X_train, y_train, ratio=sampling_ratio)
        print("X_train, y_train shapes= {} {}".format(X_train.shape, y_train.shape))

        X_test, y_test  = get_sample(X_test, y_test, ratio=sampling_ratio)
        print("X_test, y_test shapes= {} {}".format(X_test.shape, y_test.shape))

    return X_train, y_train, X_test, y_test
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#    
#-------------------------------------------------------------------------------
def load_dataset(filename, dataset_type='P7', is_label_encoded=False) :
    '''Load dataset from file name given as function parameter.
    '''
    is_label_encoded = p8_util_config.IS_LABEL_ENCODED
    if dataset_type == 'P7' :
        (x_train,x_test, y_train, y_test) = p5_util.object_load(filename)
        
        number = x_train.shape[0]
        if p8_util_config.NN_TYPE == 'RNN' :
            x_train = batch_coloredimage_serial_reshape(x_train)
            x_test  = batch_coloredimage_serial_reshape(x_test)
        else :
            pass
        
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        
        if is_label_encoded :
            y_train=array_label_encode_from_index(y_train)
            y_test=array_label_encode_from_index(y_test)
            nClasses = max(len(np.unique(y_train)), len(np.unique(y_test)))
        else :
            nClasses = y_train.shape[1]
    elif dataset_type == 'MNIST' :
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data_mnist()
        nClasses = max(len(np.unique(y_train)), len(np.unique(y_test)))
        if False:
            y_train=array_label_encode_binary(y_train)
            y_test=array_label_encode_binary(y_test)
            y_valid=array_label_encode_binary(y_valid)
            nClasses = y_train.shape[1]
    elif dataset_type == 'JIGSAW' :
        nClasses = 0
        if False :
            sampling_ratio = p8_util_config.SAMPLING_RATIO
            X_train, y_train, X_test, y_test = \
            load_dataset_jigsaw(sampling_ratio=sampling_ratio)
            
            x_train, x_test = preprocess_dataset_jigsaw(X_train, X_test)    
        else :
            filename = './data/X_train_encoded.dump'
            x_train = p5_util.object_load(filename)

            filename = './data/X_test_encoded.dump'
            x_test = p5_util.object_load(filename)

            filename = './data/y_test.dump'
            y_test = p5_util.object_load(filename)
            if type(y_test) is list :
                y_test = np.array(y_test)

            filename = './data/y_train.dump'
            y_train = p5_util.object_load(filename)
            if type(y_train) is list :
                y_train = np.array(y_train)
    else :
        pass
    
    #w_size = x_train.shape[1]
    #h_size = x_train.shape[2]        

    tuple_dimension = (x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    
    #print("Dimensions= {}".format(tuple_dimension))
    #print("Number of classes= "+str(nClasses))
    if dataset_type == 'MNIST' :
        return x_train, x_test, y_train, y_test,  nClasses \
        ,tuple_dimension[0][1:]        

    return x_train, x_test, y_train, y_test, nClasses,tuple_dimension[0][1:]
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Estimator configuration.
#-------------------------------------------------------------------------------
def make_config(model_name, output_dir=None, is_restored=False):
    '''Reset output directory.
    Returns a TF configuration object for feeding Esmimator.
    '''

    if output_dir is None : 
        output_dir=LOG_DIR

    outdir = os.path.join(output_dir, model_name)
    
    if IS_DEBUG is True :
        print("\n*** make_config() : output dir= {}".format(outdir))
    
    if is_restored is False :
        shutil.rmtree(outdir, ignore_errors = True)
    else :
        pass    
    return tf.estimator.RunConfig(
        save_checkpoints_steps=10,
        save_summary_steps=10,
        tf_random_seed=RANDOM_SEED,
        model_dir=outdir,
        ),outdir
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def array_label_encode_from_index(y):
    '''Label encoder from index array. 
    '''
    array_label_encode = np.array([np.where(y[i]==1)[0][0] for i in range(0,len(y),1)])
    return array_label_encode
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def array_label_encode_binary(y):
    '''Label encoder from value into binary values.
    '''

    lb = preprocessing.LabelBinarizer()
    y_enc = lb.fit_transform(y)
    return y_enc
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def preprocess_text(text, label):
    """Preprocesses an image for an `Estimator`."""
    if IS_DEBUG is True :
        print("\n*** preprocess_text() : label shape= {}".format(label.shape))
    
    features = {p8_util_config.FEATURES_KEY: text}
    return features, label

def preprocess_color_image(image, label):
    """Preprocesses an image for an `Estimator`."""
    if IS_DEBUG is True :
        print("\n*** preprocess_color_image() : label shape= {}".format(label.shape))
    
    features = {FEATURES_KEY: image}
    return features, label

def preprocess_image(image, label):
    """Preprocesses an image for an `Estimator`."""
      
    if IS_DEBUG is True :
        print("\n*** preprocess_image() : image shape= {}".format(image.shape))

    features = {FEATURES_KEY: image}
    
    if IS_DEBUG is True :
        print("\n*** preprocess_image() : features= {} / label shape= {}".format(features, label.shape))
    return features, label
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def generator(images, labels):
  """Returns a generator that returns image-label pairs."""

  def _gen():
    for image, label in zip(images, labels):
      yield image, label

    if IS_DEBUG is True :
        print("\n*** generator() : labels shape= {} / label values= {}".format(labels.shape, labels[10]))

  return _gen
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_model_name(nn_type):
    '''Build estimator model name from type of neural network.
    '''    

    if 'RNN' == nn_type :
        dict_rnn_layer_config= p8_util_config.dict_rnn_layer_config
        rnn_cell_type = dict_rnn_layer_config['rnn_cell_type']
        model_name = rnn_cell_type
    elif  'CNN' == nn_type or 'CNNBase' == nn_type :
        dict_cnn_layer_config = p8_util_config.dict_cnn_layer_config
        if dict_cnn_layer_config['cnn_dense_layer_num'] is None :
            model_name = nn_type+'DENSE'
        elif dict_cnn_layer_config['cnn_conv_layer_num'] is None :
            model_name = nn_type+'CONV'            
        else :
            model_name = nn_type    
    else : 
        model_name = nn_type
    return model_name
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def input_fn_2(partition, input_fn_param) :
  '''Generates an input_fn for the Estimator.
  Estimator is in charge to train and evluate model, serialise weights and biases.
  
  input_fn pumps and feeds data into the estimator.
  Data is pumped either from a .dump file or from MNIST dataset.
  
  Input:
    *   partition : fixes train, evaluation or prediction operation.
    *   input_fn_param : contains all parameters for loading data.
  '''

  def _input_fn():
    num_epochs = input_fn_param['num_epochs']
    batch_size = input_fn_param['batch_size']
    feature_shape = input_fn_param['feature_shape']    
    dataset_type = input_fn_param['dataset_type']    
    
    
    #---------------------------------------------------------------------------
    # Select data source.
    #---------------------------------------------------------------------------
    if dataset_type == 'MNIST' :
        x_train, x_test, y_train, y_test, n_class, feature_shape \
        = load_dataset(None, dataset_type=dataset_type)
    elif dataset_type == 'P7' :
        filename_dataset='./data/arr_keras_X_y_train_test.dump'
        x_train, x_test, y_train, y_test, n_class, feature_shape \
        = load_dataset(filename_dataset)
    elif dataset_type == 'JIGSAW':
        filename = None
        x_train, x_test, y_train, y_test, nClasses,tuple_dimension = \
        load_dataset(filename, dataset_type=dataset_type)
        
    else:
        print("\n*** ERROR : Dataset type= {} Not supported!".format(dataset_type))
    if 1 == len(y_train.shape):
        label_shape = [1]
    else : 
        label_shape = [y_train.shape[1]]
     
    #---------------------------------------------------------------------------
    # Defining shapes with None as first value allows the generator to 
    # adapt itself when batch does not fit expected size.
    # Otherwise an error value may be raized such as 
    # ValueError: `generator` yielded an element of shape () where an element of shape (1,) was expected.
    #---------------------------------------------------------------------------
    if IS_DEBUG is True :
        print("\n*** input_fn() : feature_shape= {} / label_shape= {}"\
        .format(feature_shape, label_shape))
    
    #---------------------------------------------------------------------------
    # Building the dataset, tensorflow formated
    #---------------------------------------------------------------------------
    training=False        
    if partition == "train":
        training = True
        if p8_util_config.DATASET_TYPE == 'JIGSAW' :
            dataset = tf.data.Dataset.from_generator(generator(x_train, y_train), (tf.float32, tf.float32), (feature_shape, ()))
        else :
            dataset = tf.data.Dataset.from_generator(generator(x_train, y_train), (tf.float32, tf.int32), (feature_shape, ()))
        dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat(num_epochs)
    else:
        if p8_util_config.DATASET_TYPE == 'JIGSAW' :
            dataset = tf.data.Dataset.from_generator(generator(x_test, y_test), (tf.float32, tf.float32), (feature_shape, ()))
        else :
            dataset = tf.data.Dataset.from_generator(generator(x_test, y_test), (tf.float32, tf.int32), (feature_shape, ()))
        if IS_DEBUG is True :
            print("\n*** input_fn : TEST / feature_shape= {}".format(feature_shape))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    if p8_util_config.DATASET_TYPE == 'P7':
        dataset = dataset.map(preprocess_color_image).batch(batch_size)
    if p8_util_config.DATASET_TYPE == 'MNIST':
        dataset = dataset.map(preprocess_image).batch(batch_size)
    if p8_util_config.DATASET_TYPE == 'JIGSAW':
        dataset = dataset.map(preprocess_text).batch(batch_size)
    
    iterator = dataset.make_one_shot_iterator()

    if IS_DEBUG is True :
        print("\n***_input_fn() : Label shape ={}".format(label_shape))

    features, labels = iterator.get_next()
    
    return features, labels

  return _input_fn    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def input_fn(partition, x, y, input_fn_param):
  """Generate an input_fn for the Estimator."""

  def _input_fn():
    
    num_epochs = input_fn_param['num_epochs']
    batch_size = input_fn_param['batch_size']
    feature_shape = input_fn_param['feature_shape']
    
    if 1 == len(y.shape):
        label_shape = [1]
    else : 
        label_shape = [y.shape[1]]
    
    

    #---------------------------------------------------------------------------
    # Defining shapes with None as first value allows the generator to 
    # adapt itself when batch does not fit expected size.
    # Otherwise an error value may be raized such as 
    # ValueError: `generator` yielded an element of shape () where an element of shape (1,) was expected.
    #---------------------------------------------------------------------------
    
    if IS_DEBUG is True :
        print("\n*** input_fn() : feature_shape= {} / label_shape= {}"\
        .format(feature_shape, label_shape))
    
    training=False
    
    dataset = tf.data.Dataset.from_generator(generator(x, y), (tf.float32, tf.int32), (feature_shape, ()))
        
        
    if partition == "train":
        training = True
        dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat(num_epochs)
    else:
        if IS_DEBUG is True :
            print("\n*** input_fn : TEST / feature_shape= {}".format(feature_shape))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.map(preprocess_color_image).batch(batch_size)
    
    iterator = dataset.make_one_shot_iterator()

    if IS_DEBUG is True :
        print("\n***_input_fn() : Label shape ={}".format(label_shape))

    features, labels = iterator.get_next()
    
    return features, labels

  return _input_fn    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def train_and_evaluate(adanet_estimator):
    '''Apply ADANET train and evaluation.
    '''

    #---------------------------------------------------------------------
    # Input function parameters stay same as for train_input_fn
    #---------------------------------------------------------------------
    dataset_type = p8_util_config.DATASET_TYPE
    input_fn_param={'num_epochs':p8_util_config.NUM_EPOCHS,\
                    'batch_size':p8_util_config.BATCH_SIZE,\
                    'feature_shape': p8_util_config.dict_adanet_config['adanet_feature_shape'],\
                    'dataset_type':dataset_type
                   }

    train_input_fn=input_fn_2("train", input_fn_param)
    test_input_fn =input_fn_2("test",input_fn_param)

    train_spec=tf.estimator.TrainSpec(
            input_fn= train_input_fn,
            max_steps=p8_util_config.TRAIN_STEPS)

    eval_spec=tf.estimator.EvalSpec(
            input_fn= test_input_fn,
            steps=None,
            start_delay_secs=1,
            throttle_secs=1)

    results, _ = tf.estimator.train_and_evaluate(adanet_estimator, train_spec, eval_spec)
    return results, _
#-------------------------------------------------------------------------------


  

