#!/usr/bin/python3.6
#-*- coding: utf-8 -*-


'''Utilities for benchmarking.
Benchmark model is the one issued from Kaggle Jigsaw kernel, located at :
https://www.kaggle.com/dborkan/benchmark-kernel

This kernel provides an implementation of functions for unintended biais 
evaluation.

Submitted model is the one that run for competition. It uses unintended biais 
functions from this package to evaluate its performances.

In the comments here-under, subgroups refer to identities. These terms will 
be used equivalently.

'''
import numpy as np
import pandas as pd


from sklearn import metrics
from sklearn import model_selection

import keras
from keras.utils import to_categorical

from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.layers.merge import concatenate
from keras.regularizers import l1
from keras.regularizers import l2


import p5_util
import p9_util
import p9_util_metrics
import p9_util_benchmark


TOXICITY_COLUMN = p9_util.COLUMN_NAME_TARGET
SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'
MAX_SEQUENCE_LENGTH = 250
EMBEDDINGS_PATH = './data/Glove/glove.6B.100d.txt'
EMBEDDINGS_DIMENSION = 100

LEARNING_RATE = 0.00005
NUM_EPOCHS = 10
BATCH_SIZE = 128

MAX_NUM_WORDS = 10000
DROPOUT_RATE = 0.3

'''Parameters used in order to train classifiers models. Models may be benchmark 
model or submission model.
Submission model is the one to be submitted in order to compete to Kaggle competition.
Benchmark model is the one provided by Kaggle with implementation of bias algorithm 
evaluation.
Both models use the ame configuration file format.
'''
dict_param_benchmark = {
#-------------------------------------------------------------------------------
# Root directory for storing artefacts related to benchmark model.
#-------------------------------------------------------------------------------
'root_directory' : './data/benchmark/',\
'format_file' : '.dill',\
'format_model':'.h5',\

#-------------------------------------------------------------------------------
# Generic name for benchmark dataset for validation to be backup or restored.
# This dataset will be used into validation step to compare benchmark model 
# with submitted model.
#-------------------------------------------------------------------------------
'root_filename_benchmark' : './data/benchmark/df_sample_benchmark_',\

#-------------------------------------------------------------------------------
# Generic name for benchmark model to be backup or restored.
#-------------------------------------------------------------------------------
'root_filename_model' : 'model_jigsaw_',\

#-------------------------------------------------------------------------------
# Number of samples used to validate the model
# This number is computed from n_sample_train and updated.
#-------------------------------------------------------------------------------
'n_sample' : None,\

#-------------------------------------------------------------------------------
# Number of samples used to train the model
# When this value is None, then the whole number of observations into train 
# dataset is used to train the model.
# Otherwise, a part of observtions from train dataset is used to train the model. 
#-------------------------------------------------------------------------------
'n_sample_train': None,\

#-------------------------------------------------------------------------------
# Model type : benchmark or submission
# Benchmark model is the one provided from Kaggle.
# Submission model is the one built by competitor.
# Expected values : submission, benchmark.
#-------------------------------------------------------------------------------
'model_type' : 'benchmark',\

#-------------------------------------------------------------------------------
# Reload a trained model when flag is fixed to True.
# Otherwise, build model.
#-------------------------------------------------------------------------------
'is_model_reloaded' : False   ,\

#-------------------------------------------------------------------------------
# Use Keras embedding layer in order to build CNN network
#-------------------------------------------------------------------------------
'is_embedding_layer' : True   ,\

#-------------------------------------------------------------------------------
# Reload train and test dataset.
# It is used to build a new model using same dataset and same transformer.
#-------------------------------------------------------------------------------
'is_dataset_reloaded' : True,\

#-------------------------------------------------------------------------------
# In case configuration is undefined or this flag is False, 
# then embeddings are loaded, embeddings matrix is built and tokenozer 
# is built.
# Otherwise, embeddings matrix is reloaded.
#-------------------------------------------------------------------------------
'is_embedding_reloaded' : True,\

#-------------------------------------------------------------------------------
# Threshold that fixe toxicity when training the model:
#  target < threshold : safe comment 
#  target >= threshold : toxic comment
#-------------------------------------------------------------------------------
'threshold':0.5,\

#-------------------------------------------------------------------------------
# Embedding dimension: 100 or 300 are expected values.
#-------------------------------------------------------------------------------
'embeddings_dimension':100,\

#-------------------------------------------------------------------------------
# Max validation accuracy achieved by model during training step.
# Used to save the best model.
#-------------------------------------------------------------------------------
'val_score_max':True,\

'epochs':10,\
}


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def _get_name_dimension(dict_param) :
    '''Returns an expression of embeddings dimension used to build a 
    file name.
    '''
    return str(dict_param['embeddings_dimension'])+'D_'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def _get_dict_param_benchmark(dict_param_benchmark) :
    '''Returns dictionary of parameters used for configuration in order to 
    build Keras models.
    
    When value given as function argument is None, then dictionary of parameters 
    defined in this file is returned. Otherwise, value given as function argument 
    is returned.
    
    Inputs :
        *   dict_param_benchmark : parameters structured as a dictionary.
    Output :
        *   Parameters structured as a dictionary.
    '''
    dict_param = dict()
    #---------------------------------------------------------------------------
    # Fixe dictionary of parameters, considering dict_param_benchmark given as 
    # parameter.
    #---------------------------------------------------------------------------
    if dict_param_benchmark is None :
        dict_param  = p9_util_benchmark.dict_param_benchmark.copy()
    else :
        dict_param  = dict_param_benchmark.copy()
    return dict_param
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def _get_n_sample_train_name(dict_param) :
    '''Returns an expression of number of observations in train dataset. 
    It is used in order to build a file name.
    
    Inputs :
        *   dict_param_benchmark : parameters structured as a dictionary.
    Output :
        *   Parameters structured as a dictionary.
    '''

    if dict_param['n_sample_train'] is None :
        name = 'FULL'
    else:
        name = str(dict_param['n_sample_train'])
    return name
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_filename_history(dict_param_benchmark=None) :
    '''Build a name from parameters that is used in order to reference 
    the file containing history of a Keras training process.

    Inputs :
        *   dict_param_benchmark : parameters structured as a dictionary. When 
        this value is None, then dictionary defined in this file is used.
    Output :
        *   File name used to store Keras history training.
    
    '''

    dict_param = _get_dict_param_benchmark(dict_param_benchmark)
    
    filename = dict_param['root_directory']
    filename += 'history_'
    filename += str(dict_param['model_type'])+'_'
    filename += str(dict_param['embeddings_dimension'])+'D_'
    filename += _get_n_sample_train_name(dict_param)+'_'
    filename += str(dict_param['format_file'])
    return filename
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_filename_embedding_matrix(dict_param_benchmark=None) :
    '''Build a name from parameters that is used in order to reference 
    the file containing embeddings matrix.

    Inputs :
        *   dict_param_benchmark : parameters structured as a dictionary. When 
        this value is None, then dictionary defined in this file is used.
    Output :
        *   File name used to store embeddings matrix.
    '''

    dict_param = _get_dict_param_benchmark(dict_param_benchmark)
    
    filename = dict_param['root_directory']
    filename += 'embedding_matrix_'
    filename += str(dict_param['embeddings_dimension'])+'D_'
    filename += _get_n_sample_train_name(dict_param)
    filename += str(dict_param['format_file'])
    return filename
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_filename_tokenizer(dict_param_benchmark=None) :
    '''Build a name from parameters that is used in order to reference 
    the file containing a tokenizer.

    Inputs :
        *   dict_param_benchmark : parameters structured as a dictionary. When 
        this value is None, then dictionary defined in this file is used.
    Output :
        *   File name used to store tokenizer.
    '''

    dict_param = _get_dict_param_benchmark(dict_param_benchmark)

    
    filename = dict_param['root_directory']
    filename += 'tokenizer_'
    if dict_param['n_sample_train'] is None :
        filename += 'FULL'
    else:
        filename += str(dict_param['n_sample_train'])
    
    filename += str(dict_param['format_file'])
    
    return filename
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_filename_param(dict_param_benchmark=None) :
    '''Build a name from parameters that is used in order to reference 
    the file containing this parameters.

    Inputs :
        *   dict_param_benchmark : parameters structured as a dictionary. When 
        this value is None, then dictionary defined in this file is used.
    Output :
        *   File name used to store configuration parameters.
    '''
    #---------------------------------------------------------------------------
    # Fixe dictionary of parameters, considering dict_param_benchmark given as 
    # parameter.
    #---------------------------------------------------------------------------
    dict_param = _get_dict_param_benchmark(dict_param_benchmark)
    filename_modelcore = _build_filename_modelcore(dict_param)
    filename = dict_param['root_directory']
    model_type = str(dict_param['model_type'])
    filename +='dict_param_'+model_type+'__'+filename_modelcore
    filename += dict_param['format_file']
    return filename
    
    
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_filename_benchmark(is_train=False, dict_param_benchmark=None) :
    '''Build a name from parameters that is used in order to reference 
    the file containing dataset used either train train a model or 
    used for validation.

    Inputs :
        *   is_train : when False, the name will be composed with the term 
        'valid', otherwise, the name will be compsoed with the term 'train'.
        
        *   dict_param_benchmark : parameters structured as a dictionary. When 
        this value is None, then dictionary defined in this file is used.
    Output :
        *   File name used to store a train or a validation dataset.
    '''

    dict_param = dict()
    #---------------------------------------------------------------------------
    # Fixe dictionary of parameters, considering dict_param_benchmark given as 
    # parameter.
    #---------------------------------------------------------------------------
    dict_param = _get_dict_param_benchmark(dict_param_benchmark)
        
    filename = dict_param['root_filename_benchmark']
        
    if is_train :
        filename += 'train_'
        #filename += _get_name_dimension(dict_param)
        if dict_param['n_sample_train'] is None :
            filename += 'FULL'
        else :
            filename += str(dict_param['n_sample_train'])
    else :
        filename += 'valid_'
        #filename += _get_name_dimension(dict_param)
        if dict_param['n_sample_train'] is None :
            filename +='FULL'
        else :
           filename += str(dict_param['n_sample'])

    filename += str(dict_param['format_file'])
    
    return filename
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def _build_filename_modelcore(dict_param) :
    '''Build a part of the name from parameters that is used in order to reference 
    the file containing a prediction model.

    Inputs :
        *   dict_param : parameters structured as a dictionary. 
    Output :
        *   The part of the file name for a prediction model.
    '''
    n_sample = dict_param['n_sample_train']
    
    filename = dict_param['root_filename_model']
    model_type = str(dict_param['model_type'])+'_'
    filename +=model_type
    if n_sample is None :
        filename += 'sampleFULL_'
    else :
        filename += 'sample'+str(dict_param['n_sample_train'])+'_'
    filename += 'threshold'+str(dict_param['threshold'])
    return filename
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_filename_model(dict_param_benchmark=None) :
    '''Build a name from parameters, name that is used in order to reference 
    the file containing a prediction model.

    Inputs :
        *   dict_param_benchmark : parameters structured as a dictionary. When 
        this value is None, then dictionary defined in this file is used.
    Output :
        *   File name used to store a prediction model.
    '''

    dict_param = _get_dict_param_benchmark(dict_param_benchmark)
    filename   = dict_param['root_directory'] 
    filename += _build_filename_modelcore(dict_param)+'_'
    if dict_param['val_score_max'] :
        filename +='best'
    else :
        pass
    #filename += 'valscoremax'+str(dict_param['val_score_max'])

    filename += str(dict_param['format_model'])
    
    return filename
#-------------------------------------------------------------------------------
    
def build_embeddings_matrix(tokenizer, dict_param_benchmark_=None):
    '''Build an embedding matrix from a tokenizer and configuration parameters.
    
    Embedding matrix has the following structure : 
    --> Rows are words or tokens issued from a corpus of texts.
    --> Columns are  vectors coefficients of embeddings.
    
    Embeddings matrix is saved over harddisk before its value to be returned.
    The embeddings matrix file name is buil from dictionary of configuration 
    parameters.
    
    Inputs : 
        * tokenizer : this is an operator that has been used in corpus 
        tokenization process. It contains the vocabulary issued from corpus 
        tokenization. Rows of resulting embedding matrix will be issued from 
        the tokenizer vocabulary.

        *   dict_param_benchmark : parameters structured as a dictionary. When 
        this value is None, then dictionary defined in this file is used.

    Output :
        * embeddings matrix the represents a digitalized corpus.
    '''

    if dict_param_benchmark_ is None :
        embeddings_dimension = EMBEDDINGS_DIMENSION
    else :
        embeddings_dimension = dict_param_benchmark_['embeddings_dimension']
    
    #---------------------------------------------------------------------------
    # Load embeddings
    #---------------------------------------------------------------------------
    if dict_param_benchmark_ is None or not dict_param_benchmark_['is_embedding_reloaded'] :
        #-----------------------------------------------------------------------
        # In case configuration is undefined or flag to reload embeddings matrix 
        # is False, then this last is built.
        #-----------------------------------------------------------------------
        print('Loading embeddings...')
        embeddings_index = {}
        with open(EMBEDDINGS_PATH) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        #---------------------------------------------------------------------------
        # Build embeddings
        #---------------------------------------------------------------------------
        print('Building embeddings...')
        embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,
                                     embeddings_dimension))
        num_words_in_embedding = 0
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                num_words_in_embedding += 1
                embedding_matrix[i] = embedding_vector
            else :
                #-------------------------------------------------------------------
                # words not found in embedding index will be all-zeros.
                #-------------------------------------------------------------------
                pass
        #-----------------------------------------------------------------------
        # Save embeddings matrix
        #-----------------------------------------------------------------------
        filename_embedding_matrix = p9_util_benchmark.build_filename_embedding_matrix(dict_param_benchmark=dict_param_benchmark_)
        p5_util.object_dump(embedding_matrix, filename_embedding_matrix, is_verbose=True)
    else :
        #-----------------------------------------------------------------------
        # Load embeddings matrix
        #-----------------------------------------------------------------------
        print('Loading embeddings matrix...')
        filename_embedding_matrix = p9_util_benchmark.build_filename_embedding_matrix(dict_param_benchmark=dict_param_benchmark_)
        embedding_matrix = p5_util.object_load(filename_embedding_matrix, is_verbose=True)


    return embedding_matrix
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_model(dict_param_benchmark, dict_param_keras_cnn):
    '''Build either submission model or benchmark model based on configuration 
    given as function parameter.
    
    Input : 
        *   dict_param_benchmark : parameters for both, submission and benchmark models.
        *   dict_param_keras_cnn : CNN hyper-parameters for building either 
        benchmak or submission model.

    Output : 
        *   the compiles CNN model 
        *   list of callbacks activated by Keras package in order to process 
        additional operations.          
        
    '''
    
    dict_param_keras = dict_param_keras_cnn['dict_param_keras']
    
    input_shape = dict_param_keras['input_dim']
    nbClasses = dict_param_keras['nbClasses']
    dropout_rate = dict_param_keras['dropout_rate']
    lr = dict_param_keras['lr']
    
    #----------------------------------------------------
    # Hard-coded parameters
    #----------------------------------------------------                
    is_batch_norm = dict_param_keras['is_batch_normalized']

    if dict_param_benchmark is None :
        model_type = 'submission'
    else : 
        model_type = dict_param_benchmark['model_type']

    if dict_param_benchmark['is_embedding_layer'] :
        filename_tokenizer = p9_util_benchmark.build_filename_tokenizer(dict_param_benchmark=dict_param_benchmark)
        tokenizer = p5_util.object_load(filename_tokenizer)
        
        sequence_input = Input(shape=(input_shape[0],), dtype='int32')
        embeddings_dimension = input_shape[1]
        
        embedding_matrix = build_embeddings_matrix(tokenizer, dict_param_benchmark_=dict_param_benchmark)
                
        embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                                    embeddings_dimension,
                                    weights=[embedding_matrix],
                                    input_length=input_shape[0],
                                    trainable=False)
        x = embedding_layer(sequence_input)
    else : 
        x = Input(shape=input_shape)
        sequence_input = x
                                
    if 'submission' ==  model_type :
        if is_batch_norm :
            x = BatchNormalization()(x)
        x = Conv1D(256, 2, activation='relu', padding='same')(x)
        print(type(x))
        x = MaxPooling1D(5, padding='same')(x)

        if is_batch_norm :
            x = BatchNormalization()(x)
        x = Conv1D(256, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(5, padding='same')(x)
        
        if is_batch_norm :
            x = BatchNormalization()(x)
        x = Conv1D(256, 4, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        if False :
            if is_batch_norm :
                x = BatchNormalization()(x)
            x = Conv1D(256, 5, activation='relu', padding='same')(x)
            x = MaxPooling1D(2, padding='same')(x)

            if is_batch_norm :
                x = BatchNormalization()(x)
            x = Conv1D(256, 6, strides=1, activation='relu', padding='same')(x)
            x = MaxPooling1D(2, padding='same')(x)
        
        x = Flatten()(x)
        if is_batch_norm :
            x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(2, activation='softmax')(x)
    elif 'benchmark' == model_type :
        x = Conv1D(128, 2, activation='relu', padding='same')(x)
        x = MaxPooling1D(5, padding='same')(x)
        
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(5, padding='same')(x)
        
        x = Conv1D(128, 4, activation='relu', padding='same')(x)
        x = MaxPooling1D(40, padding='same')(x)
        
        x = Flatten()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(2, activation='softmax')(x)
    else :
        print("\n*** ERROR : Unknown model type = {}".format(dict_param_benchmark['model_type']))


    #---------------------------------------------------------------------------
    # Build a compiled model from input and output layers.
    #---------------------------------------------------------------------------
    list_callback = list()
    if (sequence_input is not None) and (preds is not None) :
        model = Model(sequence_input, preds)
        model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(lr=lr),
                      metrics=['acc'])
        
        model.summary()
        if dict_param_benchmark['val_score_max'] :
            # checkpoint
            filepath=p9_util_benchmark.build_filename_model(dict_param_benchmark=dict_param_benchmark)
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            list_callback = [checkpoint]
        else :
            list_callback = None
    else :
        model = None
    return model,list_callback
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def train_model(train_generator, valid_generator, model,dict_param_keras_cnn, \
list_callback=None):
    '''Train a Keras model.
    When list_callback is not none, model is saved if model performances have 
    been improved from previous EPOCH.
    Model filename is included into callbacks.
    
    Inputs :
        *   train_generator : object used by Keras framework in order to pump train 
        dataset partitions from files on hardisk.
        
        *   valid_generator : object used by Keras framework in order to pump 
        validations daset partitions from files on hardisk.
        
        *   model : Kerass model to be trained.
        
        *   dict_param_keras_cnn : dictionary containing configuration parameters 
        used to train model : number of EPOCHS and verbocity.
        
        *   list_callback : list of callback objects containing functions to 
        be activated by Keras framework in the training process.
        
    Outputs : 
        *   the trained Keras model 
        *   history issued from the training process.
    '''
    # Train model.
    val_acc_max = 0.0
    print('\nTraining model...')
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
                                  callbacks=list_callback,
                                  epochs=epochs)
    

    return model, history
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_benchmark_model():

    # Read embeddings
    print('Read embeddings model...')
    embeddings_index = {}
    with open(EMBEDDINGS_PATH) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Build embeddings
    print('Building embeddings...')
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,
                                 EMBEDDINGS_DIMENSION))
    num_words_in_embedding = 0
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            num_words_in_embedding += 1
            embedding_matrix[i] = embedding_vector
        else :
            # words not found in embedding index will be all-zeros.
            pass

    # Create model layers.
    def get_convolutional_neural_net_layers():
        """Returns (input_layer, output_layer)"""
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                                    EMBEDDINGS_DIMENSION,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        x = embedding_layer(sequence_input)
        x = Conv1D(128, 2, activation='relu', padding='same')(x)
        x = MaxPooling1D(5, padding='same')(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(5, padding='same')(x)
        x = Conv1D(128, 4, activation='relu', padding='same')(x)
        x = MaxPooling1D(40, padding='same')(x)
        x = Flatten()(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(2, activation='softmax')(x)
        return sequence_input, preds

    
    
    # Build model.
    print('CNN model building...')
    input_layer, output_layer = get_convolutional_neural_net_layers()

    # Compile model.
    print('Model compilation...')
    input_layer, output_layer = get_convolutional_neural_net_layers()
    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=LEARNING_RATE),
                  metrics=['acc'])


    return model
#-------------------------------------------------------------------------------
    

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def train_benchmark_model(model, train_df, validate_df, tokenizer, \
dict_param_keras_cnn=None, list_callback=None):

    #---------------------------------------------------------------------------
    # Prepare data
    #---------------------------------------------------------------------------
    print('Data preparation...')
    train_text = pad_text(train_df[TEXT_COLUMN], tokenizer,dict_param_keras_cnn=dict_param_keras_cnn)
    train_labels = keras.utils.to_categorical(train_df[TOXICITY_COLUMN])
    
    validate_text = pad_text(validate_df[TEXT_COLUMN], tokenizer, dict_param_keras_cnn=dict_param_keras_cnn)
    validate_labels = keras.utils.to_categorical(validate_df[TOXICITY_COLUMN])
    
    #---------------------------------------------------------------------------
    # Configuration
    #---------------------------------------------------------------------------
    if dict_param_keras_cnn is None :
        batch_size = BATCH_SIZE
        num_epochs = NUM_EPOCHS
        verbose_level = 1    
    else:
        batch_size = dict_param_keras_cnn['dict_param_keras']['batch_size']
        num_epochs = dict_param_keras_cnn['dict_param_keras']['nb_epoch']
        verbose_level = dict_param_keras_cnn['dict_param_keras']['verbose']

    print("\nBatch size= {}".format(batch_size))

    #---------------------------------------------------------------------------
    # Train model.
    #---------------------------------------------------------------------------
    print('Model training...')
    history = model.fit(train_text,\
              train_labels,\
              batch_size=batch_size,\
              epochs=num_epochs,\
              validation_data=(validate_text, validate_labels),\
              verbose=verbose_level,\
              callbacks=list_callback)

    return model, history
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def pad_text(texts, tokenizer, dict_param_keras_cnn=None):
    if dict_param_keras_cnn is None :
        maxlen=MAX_SEQUENCE_LENGTH
    else :
        maxlen=dict_param_keras_cnn['dict_param_keras']['input_dim'][0]
        
    return keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=maxlen)
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# 
def convert_to_bool(df, col_name, threshold):
    '''Convert values from a given column into booleans, considering a threshold value.
    Values greater or equal threshold will be assigned to True, 
    while other values will be assigned to False.
    '''
    df[col_name] = np.where(df[col_name] >= threshold, True, False)
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def convert_dataframe_to_bool(df, dict_param_benchmark_=None, pred_column='my_model'):

    if dict_param_benchmark_ is None :
        threshold = THRESHOLD
    else :
        threshold = dict_param_benchmark_['threshold']

    bool_df = df.copy()
    for col in ['target'] + [pred_column] + p9_util_metrics.IDENTITY_COLUMNS:
        if col in bool_df.columns :
            convert_to_bool(bool_df, col, threshold)
        else :
            pass
    return bool_df
#-------------------------------------------------------------------------------
    
