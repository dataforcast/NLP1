#!/usr/bin/python3.6
#-*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import time
import string

import matplotlib.pyplot as plt


from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt


import keras
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten, Dense, Dropout, Embedding, Reshape, Input
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.layers.merge import concatenate
from keras.regularizers import l1
from keras.regularizers import l2


import p5_util
import p9_util_keras
import p9_util_config
import DataGenerator

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def keras_cnn_channel_build(dict_param_keras=dict(),\
                            filter_size = 3,\
                            stride_size = 2,\
                            pool_size = 1,\
                            pool_stride = None,\
                            nb_filter=8,\
                            conv_layer=1,\
                            dense_layer = 4,\
                            nb_dense_neuron = 16,\
                            list_channel=list(),\
                            list_filter_channel=list(),\
                            dense_layer_decrease_rate=1,\
                           ):
    input_shape = dict_param_keras['input_dim']
    dropout_rate = dict_param_keras['dropout_rate']
    nbClasses = dict_param_keras['nbClasses']
    lr = dict_param_keras['lr']
    is_batch_normalized = dict_param_keras['is_batch_normalized']


    #'batch_size': 500,
    #'nb_epoch': 15,
    #'verbose': 1,


    if 0 == len(list_channel) :
        print("\n***ERROR :  Channels count={}".format(len(list_channel)))
        return None

    if conv_layer < 1 :
        print("\n***ERROR : Number of convolutional layer must be greater or equal to 1")
        return None

    if dropout_rate >0. :
        is_dropout = True
    else :
        is_dropout = False

    #-------------------------------------------------------------------
    # Fixe pool_size and pool_stride hyper-parameters.
    #-------------------------------------------------------------------
    if pool_size >= 2 :
        if pool_stride is None :
            pool_stride = pool_size - 1
        else : 
            #-------------------------------------------------------------------
            # Use pool_stride given as function parameter.
            #-------------------------------------------------------------------
            pass
    else :
        if pool_stride is None :
            pool_stride = pool_size
        else :
            #-------------------------------------------------------------------
            # Use pool_stride given as function parameter.
            #-------------------------------------------------------------------
            pass        

    isRegression=True
    if nbClasses > 1 :
        isRegression=False

    if isRegression :
        loss='mean_squared_error'
        metrics=['mae']
        activation = 'linear'
    else :
        loss='categorical_crossentropy'
        #metrics=['accuracy']
        metrics=['categorical_accuracy']
        activation = 'softmax'

    if dropout_rate is not None :
        if dropout_rate > 0. :
            is_dropout = True
        else :
            pass
    else :
        pass

    #-------------------------------------------------------------------
    # Fixe regularization of cost function
    #-------------------------------------------------------------------
    regularizer = None
    regul = dict_param_keras['regul']
    if regul[0] is not None :
        value = regul[1]
        if regul[0] == 'l1' :
            regularizer = keras.regularizers.l1(value)
        elif regul[0] == 'l2' :
            regularizer = keras.regularizers.l2(value)
        else : 
            print("\n***WARNING : Keras regularizer : \'{}\' not supported!".format(regul[0]))
    else :
        pass
        
    list_tf_channel = list()
    list_input_layer = list()
    if conv_layer > 0:
        for channel_id in range(len(list_channel)) :

            #----------------------------------------------------------------
            # Input layers
            #----------------------------------------------------------------
            tf_input =  Input(shape=input_shape)
            #print(input_shape)
            list_input_layer.append(tf_input)

            #----------------------------------------------------------------
            # Convolutional layers
            #----------------------------------------------------------------
            if False :
                #---------------------------------------------------------------
                # Each channel is applied a filter size
                #---------------------------------------------------------------
                filter_size = list_channel[channel_id]
            else :
                #---------------------------------------------------------------
                # Same size of filter for any channel
                #---------------------------------------------------------------
                pass

            for i in range(conv_layer) :
                #print("Conv layer id= {}".format(i))
                if 0 == i: 
                    #----------------------------------------------------
                    # For first conv layer then input shape is feeded
                    #----------------------------------------------------
                    if is_batch_normalized :
                        tf_bn = BatchNormalization()(tf_input)
                        tf_conv = Conv1D(filters=nb_filter, \
                                         kernel_size=filter_size, \
                                         activation='relu',\
                                         strides=stride_size)(tf_input)
                    else : 
                        tf_conv = Conv1D(filters=nb_filter, \
                                         kernel_size=filter_size, \
                                         activation='relu',\
                                         strides=stride_size)(tf_input)
                    
                    if pool_size > 0 :
                        tf_conv = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(tf_conv)
                else :
                    if is_batch_normalized :
                        tf_bn = BatchNormalization()(tf_conv)
                        tf_conv = Conv1D(filters=nb_filter, \
                                         kernel_size=filter_size, \
                                         activation='relu',\
                                         strides=stride_size)(tf_bn)
                    else : 
                        tf_conv = Conv1D(filters=nb_filter, \
                                         kernel_size=filter_size, \
                                         activation='relu',\
                                         strides=stride_size)(tf_conv)
                    
                    if pool_size > 0 :
                        tf_conv = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(tf_conv)

            #----------------------------------------------------------------
            # Flatten layer for connecting dense layers
            #----------------------------------------------------------------
            tf_flat = Flatten()(tf_conv)
            list_tf_channel.append(tf_flat)

        #---------------------------------------------------------------------
        # When all channels have been built, layers are concatenated
        #---------------------------------------------------------------------
        if len(list_tf_channel) >1 :
            tf_merged_channel = concatenate(list_tf_channel)
        else :
            tf_merged_channel = list_tf_channel[0]

    else :
        # Non convolutional layer
        #----------------------------------------------------------------
        # Input layers
        #----------------------------------------------------------------
        tf_input =  Input(shape=input_shape)
        list_input_layer.append(tf_input)
        tf_merged_channel = tf_input
        

    #---------------------------------------------------------------------
    # Dense Layers
    #---------------------------------------------------------------------
    tf_dense = tf_merged_channel
    for i in range(dense_layer) :
        if is_batch_normalized :
            tf_bn = BatchNormalization()(tf_dense)
            tf_dense = Dense(nb_dense_neuron, activation='relu')(tf_bn)
        else :
            tf_dense = Dense(nb_dense_neuron, activation='relu')(tf_dense)
            
        if is_dropout :
            tf_dense = Dropout(dropout_rate)(tf_dense)
        else :
            pass
        nb_dense_neuron //= 2

    if is_batch_normalized :
        tf_dense = BatchNormalization()(tf_dense)    
    else : 
        pass
    tf_output = Dense(nbClasses, activation=activation, \
    activity_regularizer=regularizer)(tf_dense)

    #---------------------------------------------------------------------
    # Build model and do compile
    #---------------------------------------------------------------------
    model = Model(inputs=list_input_layer, outputs=tf_output)    
    sgd = keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=loss, optimizer=sgd, metrics=metrics)
    #plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------    

def keras_cnn_build(dict_param_keras=dict(),\
                    filter_size = 3,\
                    stride_size = 2,\
                    pool_size = 1,\
                    pool_stride = None,
                    nb_filter=8,\
                    conv_layer=1,\
                    dense_layer = 4,\
                    nb_dense_neuron = 16,\
                    list_channel=list(),\
                    list_filter_channel=list(),\
                    dense_layer_decrease_rate=1,\
                           ):
                    
    '''Builds a convolutional model using keras framework.
    
    Input : 
        * input_shape : size of input for input layer.
        * nbClasses : number of classes for classication model.
            When nbClasses value is 1 then estimator is turned into a 
            regression model. Otherwise, estimatir is turned into a 
            classifier model.
    '''
    
    isWordEmbedding = False
    
    input_shape = dict_param_keras['input_dim']
    nbClasses = dict_param_keras['nbClasses']
    dropout_rate = dict_param_keras['dropout_rate']
    lr = dict_param_keras['lr']
    
    if input_shape == (0,0) :
        print("\n***ERROR : invalid input_shape parameter : {}".format(input_shape))
        return None
    
    isRegression=True
    if nbClasses > 1 :
        isRegression=False

    #-------------------------------------------------------------------
    # Fixe metrics, activation, loss function
    #-------------------------------------------------------------------
    if isRegression :
        loss='mean_squared_error'
        metrics=['mse']
        activation = 'linear'
    else :
        if 2 == nbClasses  :
            loss = 'binary_crossentropy'
            metrics = ['binary_accuracy']
        else :
            loss='categorical_crossentropy'
            #metrics=['accuracy']
            metrics=['categorical_accuracy']
        
        activation = 'softmax'
    
    if dropout_rate is not None :
        if dropout_rate > 0. :
            is_dropout = True
        else :
            is_dropout = False
    else :
        pass

    #-------------------------------------------------------------------
    # Fixe regularization of cost function
    #-------------------------------------------------------------------
    regularizer = None
    regul = dict_param_keras['regul']
    if regul[0] is not None :
        value = regul[1]
        if regul[0] == 'l1' :
            regularizer = keras.regularizers.l1(value)
        elif regul[0] == 'l2' :
            regularizer = keras.regularizers.l2(value)
        else : 
            print("\n***WARNING : Keras regularizer : \'{}\' not supported!".format(regul[0]))
    else :
        pass

    #-------------------------------------------------------------------
    # Fixe pool_size and pool_stride hyper-parameters.
    #-------------------------------------------------------------------
    if pool_size >= 2 :
        if pool_stride is None :
            pool_stride = pool_size - 1
        else : 
            #-------------------------------------------------------------------
            # Use pool_stride given as function parameter.
            #-------------------------------------------------------------------
            pass
    else :
        if pool_stride is None :
            pool_stride = pool_size
        else :
            #-------------------------------------------------------------------
            # Use pool_stride given as function parameter.
            #-------------------------------------------------------------------
            pass        
    model = Sequential()
    
    if isWordEmbedding :
        #----------------------------------------------------------------
        # Embedding layer
        #----------------------------------------------------------------
        max_length = input_shape[0]
        embedding_dim = input_shape[1]
        model.add(keras.layers.Embedding(vocab_size, embedding_dim, \
        weights=[weight_matrix], input_length = max_length, trainable=False))
    else :
        #----------------------------------------------------------------
        # Convolutional layers
        #----------------------------------------------------------------
        if conv_layer > 0:

            for i in range(conv_layer) :
                if 0 == i :
                    #----------------------------------------------------
                    # For first conv layer then input shape is feeded
                    #----------------------------------------------------                
                    model.add(Conv1D(nb_filter, filter_size,padding='same', activation='relu', \
                    strides=stride_size, input_shape=input_shape) )
                else :
                    model.add(BatchNormalization())
                    model.add(Conv1D(nb_filter, filter_size,padding='same', \
                    strides=stride_size, activation='relu'))
                    if False:
                        if is_dropout :
                            model.add(Dropout(dropout_rate))
                    else :
                        pass

                if pool_size >0 :
                    model.add(MaxPooling1D(pool_size=pool_size, strides=pool_stride))

            #----------------------------------------------------------------
            # Flatten layer for connecting dense layers
            #----------------------------------------------------------------
            model.add(Flatten())
        else :
            pass    

    #---------------------------------------------------------------------------
    # Dense layers
    #---------------------------------------------------------------------------
    if conv_layer <= 0:
        #-----------------------------------------------------------------------
        # No convolutional layer : build input dense layer
        #-----------------------------------------------------------------------
        model.add(Dense(nb_dense_neuron, input_dim=input_shape[0]))
        
    for i in range(dense_layer) :
        #-----------------------------------------------------------------------
        # Dense layers are added
        #-----------------------------------------------------------------------
        model.add(BatchNormalization())
        model.add(Dense(nb_dense_neuron, activation='relu'))
        if is_dropout :
            model.add(Dropout(dropout_rate))

        #-----------------------------------------------------------------------
        # Number of neurons in next dense layer is decreased by a rate
        #-----------------------------------------------------------------------
        nb_dense_neuron //= dense_layer_decrease_rate

    model.add(BatchNormalization())
    model.add(Dense(nbClasses, activation=activation, activity_regularizer=regularizer))
    #if is_dropout :
    #    model.add(Dropout(dropout_rate))

    sgd = keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    
	#plot_model(model, show_shapes=True, to_file='keras_cnn_build.png')

    model.compile(loss=loss, optimizer=sgd, metrics=metrics)


    return model
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------    
#
#-------------------------------------------------------------------------------    
def keras_batch_normalization(keras_model, is_batch_normalized=True) :
    '''Applies batch normalization over keras layer depending on boolean 
    flag value is_batch_normalized.
    Returns a keras model with normalized batch layer.
    '''
    if is_batch_normalized :
        keras_model.add(BatchNormalization())
    else :
        pass
    return keras_model
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------    
#
#-------------------------------------------------------------------------------    
def keras_rnn_build(rnnCellName, cell_units, input_shape, \
                    isWordEmbedding=False, nbClasses=1, optimizer_name='SGD',\
                    rnn_layers=1, lr=1.e-3, is_batch_normalized=False, \
                    dense_layer_count=1,
                    dropout_rate = 0.0) :
    '''Builds a RNN model using keras framework.
    Input :
        *   rnnCellType : keras cell type from keras.layers package, 
        such as keras.layers.GRU, keras.layers.LSTM...
        *   cell_units : number of units per cell.
        *   input_shape : 
        *   isWordEmbedding : when True, then firt RNN layer uses embedding layer.
        *   isRegression : when True, then a loss function is used for regression.
            Linear activation is then applied to the output of the last layer.
            Otherwise, a loss function is used for classification.
            
        *   nbClasses : in case of isRegression is False, this is the number 
            of classes for loss function. Softmax activation is then applied to 
            the outputs of last layer.
        *   optimizer_name : gradient descent algorithm name.
        *   rnn_layers : number of stacked RNN layers.
        *   lr : learning rate for optimizer algorithm.
        *   is_batch_normalized : when True, batch normalization is applied over 
        each layer.
        *   dense_layer_count : number of dense layers 
    output : 
        * compiled RNN model.
    '''
    #---------------------------------------------------------------------------
    # Validate input parameters
    #---------------------------------------------------------------------------
    isRegression=True
    if nbClasses > 1 :
        isRegression=False
    
    if rnnCellName == 'GRU' :
        rnnCellType = keras.layers.GRU
    elif rnnCellName == 'LSTM' :
        rnnCellType = keras.layers.LSTM
    else :
        print("\n***ERROR : cell type = "+rnnCellName+" NOT SUPPORTED!")
        return None
        

    #---------------------------------------------------------------------------
    # Building RNN model
    #---------------------------------------------------------------------------
    model = keras.models.Sequential()
    
    #---------------------------------------------------------------------------
    # Input layer is built depending of embedding or not.
    #---------------------------------------------------------------------------
    if isWordEmbedding :
        model.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length))
        model.add(keras.layers.Dropout(0.2))
    else :
        model.add(rnnCellType(cell_units, input_shape=input_shape,return_sequences=True))
        #model.add(keras.layers.Dropout(0.2))
    
    
    #---------------------------------------------------------------------------
    # Add intermediates layers before last layer
    #---------------------------------------------------------------------------
    for layer_id in range(1,   rnn_layers) :
        model = keras_batch_normalization(model, is_batch_normalized)
        model.add(rnnCellType(cell_units, input_shape=input_shape,return_sequences=True))
        if dropout_rate > 0.0 :
            model.add(keras.layers.Dropout(dropout_rate))

    #---------------------------------------------------------------------------
    # Add last layer
    #---------------------------------------------------------------------------
    model = keras_batch_normalization(model, is_batch_normalized)
    model.add(rnnCellType(cell_units,return_sequences=False))
    #model.add(keras.layers.Dropout(0.2))


    if isRegression :
        metrics=['mae']
        loss='mean_squared_error'
        activation = 'linear'

    else :
        metrics=['acc']
        activation = 'softmax'
        if nbClasses > 2 :
            loss = 'categorical_crossentropy'
        else :
            loss = 'binary_crossentropy'

    #---------------------------------------------------------------------------
    # The number of nodes in dense layers are decreased until reaching the 
    # number of classes for the last dense layer.
    #---------------------------------------------------------------------------
    for layer_id in range(0,dense_layer_count):
        nb_nodes = dense_layer_count-layer_id
        model = keras_batch_normalization(model, is_batch_normalized)
        model.add(keras.layers.Dense(nb_nodes*nbClasses,activation=activation))
        if dropout_rate > 0.0 :
            model.add(keras.layers.Dropout(dropout_rate))
        
    if optimizer_name == 'SGD' :
        optimizer = keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, \
        nesterov=False)
    elif optimizer_name == 'Adagrad' :
        optimizer = keras.optimizers.Adagrad(lr=lr)
    else : 
        print("\n***ERROR : Optimizer name not supported : {}".format(optimizer_name))
        return None

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    print(model.summary())
    return model
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def keras_rnn_reshape(X) :
    '''Reshape X matrix in order to feed Keras RNN network.
    '''
    X_reshape = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    print(X_reshape.shape)
    return X_reshape
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_model_core_name(dict_param_keras_cnn) :
    '''Builds a core name from hyper-parameters given in function argument.
    A core name (or root name) is a generic name used to build another name.
    '''
    
    filter_size     = dict_param_keras_cnn['filter_size']
    stride_size     = dict_param_keras_cnn['stride_size']
    pool_size       = dict_param_keras_cnn['pool_size']
    nb_filter       = dict_param_keras_cnn['nb_filter']
    conv_layer      = dict_param_keras_cnn['conv_layer']
    nb_dense_neuron = dict_param_keras_cnn['nb_dense_neuron']
    dense_layer     = dict_param_keras_cnn['dense_layer']
    list_channel    = dict_param_keras_cnn['list_channel']
    dropout_rate    = dict_param_keras_cnn['dict_param_keras']['dropout_rate']
    regul           = dict_param_keras_cnn['dict_param_keras']['regul']
    str_channel = 'ch_'
    for channel in list_channel :
        str_channel +=str(channel)+"_"
    str_channel = str_channel[:-1]
    str_channel
    if regul[0] is not None :
        regul_name_value = 'regul_'+str(regul[0])+'_'+str(regul[1])
    else :
        regul_name_value = ''
    
    core_name = "_f"+str(filter_size)+"_s"+str(stride_size)+"_p"+str(pool_size)+"_nbf"+str(nb_filter)+"_cnv"+str(conv_layer)
    core_name += "_d"+str(dense_layer)+"_dor"+str(dropout_rate)+"_nbn"+str(nb_dense_neuron)+"_"+regul_name_value
    core_name +="_L"+str_channel
    return core_name
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def evaluate_model(dict_hyperparam, train_generator, valid_generator, \
        dict_param_keras_cnn=None, key_validation=None):
    '''Model evaluation given a list of hyper-parameters.
    
    Configuration parameters are updated with values from dict_hyperparam.
    Keras CNN model is built from these parameters and model is evaluated 
    with train_geenrator and valid_generator.
    
    MAE from valadation dataset is computed.    
    
    Input :
        *   dict_hyperparam : dictionary of hyper-parameters structured as 
            following : {hyper_parameter_name : list_of_hyerparameter_values}
        *   train_generator : data generator for train dataset.
        *   valid_generator : data generator for validation dataset.
        *   dict_param_keras_cnn : parameters for building CNN model. When 
            this value is None, then parameters are picked from file 
            p9_util_config.dict_param_keras_cnn.
        
        *   key_validation : name of the score measuring the model evaluation.
    
    Output :
        a dictionary structured as : {name_value : list_of_mae_results}
            
    '''
    #---------------------------------------------------------------
    # Update of generic paramaters depending of DataGenerator 
    #---------------------------------------------------------------
    if dict_param_keras_cnn is None :
        dict_param_keras_cnn = p9_util_config.dict_param_keras_cnn.copy()
    else :
        pass
        
    dict_param_keras = dict_param_keras_cnn['dict_param_keras'].copy()
    dict_param_keras['input_dim']          = train_generator.get_params()['keras_input_dim']
    dict_param_keras['batch_size']         = train_generator.get_params()['batch_size']
    dict_param_keras_cnn['dict_param_keras'] = dict_param_keras.copy()
    

    #---------------------------------------------------
    # Update parameters with dict_hyperparam
    #---------------------------------------------------
    dict_list_val_score = dict()
    for name_hyperparam, list_value in dict_hyperparam.items() :
        if name_hyperparam == 'extra' :
            continue
        else :
            pass
            
        if name_hyperparam == 'dict_param_keras' :
            #-------------------------------------------------------------------
            # Evaluation of values of generic Keras hyper-parameter 
            #-------------------------------------------------------------------
            llist_val_score = list()
            dict_param = dict_hyperparam['dict_param_keras']
            for name_hyperparam, list_value in dict_param.items() :

                for value in list_value :

                    #-----------------------------------------------------------
                    # Update hyper-parameter value
                    #-----------------------------------------------------------
                    dict_param_keras[name_hyperparam] = value
                    dict_param_keras_cnn['dict_param_keras'] = dict_param_keras.copy()
                    print("\nParameter : {} / value= {}".format(name_hyperparam, value))
                    if True :
                        dict_list_val_score = \
                        core_evaluate_keras_cnn_model(dict_param_keras_cnn, \
                        name_hyperparam, value, key_validation, dict_list_val_score, \
                        train_generator, valid_generator)
                    
                    else :
                        #-----------------------------------------------------------
                        # Build model with specific CNN parameters.
                        #-----------------------------------------------------------
                        model = p9_util_keras.keras_cnn_build(**dict_param_keras_cnn)

                        #-----------------------------------------------------------
                        # Evaluate model against parameter value
                        #-----------------------------------------------------------
                        epochs = dict_param_keras_cnn['dict_param_keras']['nb_epoch']
                        verbose = dict_param_keras_cnn['dict_param_keras']['verbose']
                        history = model.fit_generator(generator=train_generator,
                                                      validation_data=valid_generator,
                                                      use_multiprocessing=False,
                                                      workers=1, 
                                                      verbose=verbose, 
                                                      epochs=epochs)
                        #---------------------------------------------------------------
                        # Collect results in a dictionary
                        #---------------------------------------------------------------
                        if key_validation is None :
                            key_validation = 'val_'+str(model.metrics_names[1])
                            list_val_score =  history.history[key_validation]
                        else :
                            #list_val_score =  history.history['val_binary_accuracy']
                            pass
                        name_value = name_hyperparam+'_'+str(value)
                        dict_list_val_score[name_value] = list_val_score
                        
                        #---------------------------------------------------
                        # Save lastest results
                        #---------------------------------------------------
                        filename = rootfilename+str(name_hyperparam)+'.dill'
                        p5_util.object_dump(dict_list_val_score, filename)
                        
        else :        
            #-------------------------------------------------------------------
            # Fixe metrix name as dictionary key for recording scores
            #-------------------------------------------------------------------
            if key_validation is None :
                key_validation = 'val_'+str(model.metrics_names[1])
            else :
                #list_val_score =  history.history['val_binary_accuracy']
                pass

            #-------------------------------------------------------------------
            # Evaluation of values of specific Keras CNN hyper parameter
            #-------------------------------------------------------------------
            llist_val_score = list()

            #-------------------------------------------------------------------
            # Checking for extra parmeters
            #-------------------------------------------------------------------
            if 'extra' in dict_hyperparam.keys() :
                dict_extra_parameter = dict_hyperparam['extra']
            else :
                dict_extra_parameter = dict()
 
            #-------------------------------------------------------------------
            # Get parameter value from list_value.
            #------------------------------------------------------------------- 
            for value in list_value :
            
                #-------------------------------------------------------------------
                # Update hyper-parameter into configuration.
                #-------------------------------------------------------------------
                dict_param_keras_cnn[name_hyperparam]= value
                
                if 0 < len(dict_extra_parameter) :
                    #-----------------------------------------------------------
                    # Get extra parameters; process evalution along with 
                    # combination of hyper-parameter and extra-parameters.
                    #-----------------------------------------------------------
                    for extra_name, list_extra_value in dict_extra_parameter.items() :
                        if 0 < len(list_extra_value):
                            for extra_value in list_extra_value :
                                #-----------------------------------------------
                                # Update hyper-parameter along with extra 
                                # parameter
                                #-----------------------------------------------
                                dict_param_keras_cnn[extra_name]= extra_value
                                dict_list_val_score = core_evaluate_keras_cnn_model(dict_param_keras_cnn, name_hyperparam, value, key_validation,dict_list_val_score,train_generator, valid_generator\
                                ,extra_name=extra_name,extra_value = extra_value)
                                
                        else :
                            #---------------------------------------------------
                            # Extra parameter flag is there but list of values 
                            # is empty; process evaluation without extra-parameter
                            # to be updated into configuration. 
                            #---------------------------------------------------
                            dict_list_val_score = core_evaluate_keras_cnn_model(dict_param_keras_cnn, name_hyperparam, value, key_validation, dict_list_val_score,train_generator, valid_generator)

                else :
                    #-----------------------------------------------------------
                    # No extra parameter flag. 
                    # Evaluation process without extra-parameter
                    # to be updated into configuration. 
                    #-----------------------------------------------------------
                    dict_list_val_score = \
                    core_evaluate_keras_cnn_model(dict_param_keras_cnn, \
                    name_hyperparam, value, key_validation, dict_list_val_score, \
                    train_generator, valid_generator)

                        
    return dict_list_val_score
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def core_evaluate_keras_cnn_model(dict_param_keras_cnn, name, value, \
key_validation, dict_list_val_score,train_generator, valid_generator, \
    extra_name=None, extra_value=None) :
    '''Core evaluation of Keras models. Keras model is built and evaluated 
    against data generators and hyper-parameter value, all given into function 
    parameters.
    
    Inputs :
        *   dict_param_keras_cnn : hyper-parameters used for building Keras model  
        *   name : hyper-parameter name.
        *   value : hyper-parameter value model is evaluated against.
        *   key_validation : string, name of score used for model evaluation.
        *   dict_list_val_score : records all scores issued from model evaluation.
            Dictionary items are updated in this function.
        *   train_generator : data-generator source for model training steps.
        *   valid_generator : data-validator source for model validation steps.
        *   extra_name : extra hyper-parameter name from which value co-variates 
            with hyper-parameter value.
        *   extra_value : value of extra hyper-parameter name.

    Output:
        *   dictionary of scores that records all scores issued from model 
        evaluation.
    '''
    
    
    #---------------------------------------------------------------
    # Common root for filename where restults are stored.
    #---------------------------------------------------------------
    rootfilename = './data/dict_param_keras_cnn_'

    #---------------------------------------------------------------
    # Build model against parameters given as function argument.
    #---------------------------------------------------------------
    model = p9_util_keras.keras_cnn_build(**dict_param_keras_cnn)

    #---------------------------------------------------------------
    # Evaluate model against parameter value
    #---------------------------------------------------------------
    epochs = dict_param_keras_cnn['dict_param_keras']['nb_epoch']
    verbose = dict_param_keras_cnn['dict_param_keras']['verbose']

    if extra_name is not None :
        print("\nParameter : {} / value= {} Extra-parameter : {} / value = {}"\
        .format(name, value, extra_name, extra_value))
        name_value = name+'_'+str(value)+'_'+extra_name+'_'+str(extra_value)
        filename = rootfilename+str(name)+'_'+str(extra_name)+'.dill'

    else :
        print("\nParameter : {} / value= {}".format(name, value))
        name_value = name+'_'+str(value)
        filename = rootfilename+str(name)+'.dill'
    
        
    history = model.fit_generator(generator=train_generator,
                                  validation_data=valid_generator,
                                  use_multiprocessing=False,
                                  workers=1, 
                                  verbose=verbose, 
                                  epochs=epochs)

    #---------------------------------------------------------------
    # Collect results in a dictionary
    #---------------------------------------------------------------
    if False :
        if key_validation is None :
            list_val_score =  history.history['val_mean_absolute_error']
        else :
            list_val_score =  history.history['val_binary_accuracy']
    
    dict_list_val_score[name_value] = history.history[key_validation]

    #---------------------------------------------------
    # Save latest results
    #---------------------------------------------------
    p5_util.object_dump(dict_list_val_score, filename)
    
    return dict_list_val_score
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def display_evaluation(dict_list_val, filename=None, title=None, \
x_label=None, y_label=None):
    '''Display boxplots of results computed from evaluate_model.
    for each list of resutls, mean and standard deviation is computed.
    
    Input : 
        *   dict_list_val : value returned from evaluate_model function.
        *   filename : name of file in which display is dumped.
    
    Output : 
        None
    '''
    
    params = list(dict_list_val.keys())
    scores = list(dict_list_val.values())
        
    # summarize mean and standard deviation
    for i in range(len(scores)):
        m, s = np.mean(scores[i]), np.std(scores[i])
        if 0 < len(params) :
            print('Param=%s: %.3f%% (+/-%.3f)' % (params[i], m, s))
        else : 
            print('Score: %.3f%% (+/-%.3f)' % ( m, s))
            
    # boxplot of scores
    if y_label is not None :
        plt.ylabel(y_label, fontsize=16)
    if x_label is not None :
        plt.xlabel(x_label, fontsize=16)

    plt.boxplot(scores, labels=params)
    if title is not None :
        plt.title(title, fontsize=16)
    if filename is not None :
        plt.savefig('exp_cnn_standardize.png')
    else :
        pass

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def replace_dictkey(dict_val, new_key) :
    '''Replace key name from dictionary with the one 
    given as parameter. 
    '''
    dict_new = dict()
    for key, item in dict_val.items():
    
    
        #value = key.split('_')[-1]
        #name =new_key+str(value)
    
        list_new_name = [ch for ch in key if ch.isdigit()]
        new_name = str()
        for ch in list_new_name :
            new_name += ch+','
        new_name = new_name[:-1]
    
        dict_new[new_name] = item
        
    return dict_new
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_df_history_score_from_kfold(ddict_kfold_history, epochs_kfold) :
    '''Builds dataframe with metrics name as columns name and metrix values from 
    each step of k-fold computation.
    
    Input  : 
        * ddict_kfold_history : dictionary of scores issued from k-fold 
          evaluation
        * epochs_kfold : number of epochs per k-fold computation.
    Output :
        A dataframe with :
            * columns name as metrics for model evaluation
            * index as values for each computation of k-fold evaluation.
    '''
    #-------------------------------------------------------------
    # Build list of keys composing metrics names.
    #-------------------------------------------------------------
    list_metrix_scrore_key = list()
    
    for key in ddict_kfold_history[0].keys():
        list_metrix_scrore_key.append(key)
    list_metrix_scrore_key

    #-------------------------------------------------------------
    # Builduig dataframe that will host aggregated k-fold history scores
    # for all metrics.
    #-------------------------------------------------------------
    df_score = pd.DataFrame(columns=list_metrix_scrore_key, index \
    = range(len(ddict_kfold_history)*epochs_kfold), dtype=float)

    index = 0
    for batch_id, dict_history in ddict_kfold_history.items():
        for i_epochs in range(epochs_kfold):
            for metrix_scrore_key in list_metrix_scrore_key :
                df_score.loc[index][metrix_scrore_key] \
                = dict_history[metrix_scrore_key][i_epochs]
            index+=1
            
    return df_score
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_keras_history_from_df_score(df_history_score):
    '''Build a keras history object from dataframe issued from 
    build_df_history_score_from_kfold() function.
    
    Input : 
        * df_history_score dataframe
    Output :
        * Keras History object
    '''
    dict_history_score = dict()
    for key in df_history_score.columns :
        dict_history_score[key] = df_history_score[key]
    
    keras_history = keras.callbacks.History()
    keras_history.history = dict_history_score
    return keras_history
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def keras_model_evaluate_kfold(train_generator, batch_count = None) :
    '''Evaluate a Keras model using k-fold validations.
    A DataGenerator object for validation is built from train_gnerator. 
    At each step, a new batch is activated from DataGenerator objects.
    Model is trained with batchs issued from train_gnerator while a validation batch
    is extracted from DataGenerator object for validation.
    train_generator bacths and validation generator batch are a set of batchs, with no 
    intersection.
    For each new batch, this is set is changed.
    
    Results are saved into a file after each batch is processed.
    
    input : 
        * train_generator : DataGenerator object used to train model and to build a 
        DataGenerator object for validation.
        
        * batch_count : number of batchs to be processed. If none, this number 
        is picked from batchs into train_generator.
    Output :
        * dictionary of results structured as {batch_id : {{metric1:list_of_score}, 
                                                           {metric2:list_of_score}}}
    '''
    
    #---------------------------------------------------------------------------
    # Build Datageneragtor object for validation from train_generator.
    #---------------------------------------------------------------------------
    valid_generator = DataGenerator.DataGenerator(train_generator.dict_X, \
                                                  train_generator.dict_label, \
                                                  train_generator.nb_record,\
                              other=train_generator)

    #---------------------------------------------------------------------------
    # Fixe hyper-parameters
    #---------------------------------------------------------------------------
    dict_param_keras_cnn = p9_util_config.dict_param_keras_cnn.copy()
    dict_param_keras = p9_util_config.dict_param_keras.copy()

    dict_param_keras['input_dim']          = train_generator.get_params()['keras_input_dim']
    dict_param_keras['batch_size']         = train_generator.get_params()['batch_size']
    dict_param_keras_cnn['dict_param_keras'] = dict_param_keras.copy()

    print("")
    for key,item in dict_param_keras_cnn.items():
        if isinstance(key, dict) :
            print('')
            for k,i in key.items():
                print(k,i)
        print(key, item)


    #---------------------------------------------------------------------------
    # Build model with specific CNN parameters.
    #---------------------------------------------------------------------------
    model = p9_util_keras.keras_cnn_build(**dict_param_keras_cnn)


    #---------------------------------------------------------------------------
    # Build core name ued to save intermediate computations.
    #---------------------------------------------------------------------------
    core_name = p9_util_keras.build_model_core_name(dict_param_keras_cnn)

    
    #---------------------------------------------------------------------------
    # Fixe batch count
    #---------------------------------------------------------------------------
    if batch_count is None :
        batch_count = len(train_generator)

    #---------------------------------------------------------------------------
    # Verbose, epochs 
    #---------------------------------------------------------------------------
    verbose = dict_param_keras_cnn['dict_param_keras']['verbose']
    epochs = dict_param_keras_cnn['dict_param_keras']['nb_epoch']
    
    ddict_history_kfold = dict()
    for batch_index in range(batch_count) :
        #-----------------------------------------------------------------------
        # Fixe batch id that will be out of value in training of data issued
        # from train_gnerator.
        #-----------------------------------------------------------------------
        train_generator.index_oov = batch_index

        #-----------------------------------------------------------------------
        # Fixe batch id that will be used in validation of data issued
        # from valid_generator.
        #-----------------------------------------------------------------------
        valid_generator.index_oov = batch_index
        
        #-----------------------------------------------------------------------
        # Assign valid_generator as data for validation.
        #-----------------------------------------------------------------------
        valid_generator.is_training = False

        #-----------------------------------------------------------------------
        # Process batch for training/evaluation model in a k-fold manner.
        #-----------------------------------------------------------------------
        print("\nBatch ID= {}/{}".format(batch_index,batch_count))
        history = model.fit_generator(generator=train_generator,
                                      validation_data=valid_generator,
                                      use_multiprocessing=False,
                                      workers=1, 
                                      verbose=verbose, 
                                      epochs=epochs)

        #-----------------------------------------------------------------------
        # Store results for this batch in a dictionary 
        #-----------------------------------------------------------------------
        ddict_history_kfold[batch_index] = history.history.copy()
        
        #-----------------------------------------------------------------------
        # Save results from this batch abd all previous batchs.
        #-----------------------------------------------------------------------
        filename = "./data/ddict_history_kfold_"+core_name+"_"+str(batch_index)+".dill"
        p5_util.object_dump(ddict_history_kfold, filename, is_verbose=True)
        
        #-----------------------------------------------------------------------
        # Save model.
        #-----------------------------------------------------------------------
        filename = "./data/model_"+core_name+".h5"
        print("Save model : {}".format(filename))
        model.save(filename)
        
        
    return ddict_history_kfold
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def load_model(filename) :
    model = keras.models.load_model(filename)
    model.summary()
    return model
#-------------------------------------------------------------------------------        
