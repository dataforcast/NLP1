#!/usr/bin/python3.6
#-*- coding: utf-8 -*-

#-------------------------------------------------------------------------------
# Configuration for DataPreparator
#-------------------------------------------------------------------------------
dict_param_preparator = {\
    'binary_threshold': -1,\
    'n_sample_per_class':-1,\
}
#-------------------------------------------------------------------------------
# Configuration for DataGenerator
#-------------------------------------------------------------------------------
dict_param_generator = {\
    #------------------------------------------------------------------------
    #data_type : type of dataset; should be either 'train' or 'valid' values.
    #    This value is used to create file names for partitions.
    #------------------------------------------------------------------------
    'data_type' : "valid",\
    
    'partition_size':100,\
    'is_dimension_mux' : False,\
    #------------------------------------------------------------------------
    # Select the dimension over which a text will be projected
    # When None, then all dimensions are taken into account.
    #------------------------------------------------------------------------
    'proj_dimension' : None,\
    'batch_size': 100,\
    'n_classes': 0,\
    'n_channels': 0,\
    'is_shuffle': False,\
    
    #---------------------------------------------------------------------------
    # keras_nb_channel it is used to build CNN networks with multiple 
    # channels for convolution network part.
    #---------------------------------------------------------------------------
    'keras_nb_channel':1,\
    'list_keras_channel':[channel for channel in range(0)],\

    #---------------------------------------------------------------------------
    # Input keras layer dimension.
    #---------------------------------------------------------------------------
    'keras_input_dim':(None, None),\
    
    #---------------------------------------------------------------------------
    # Binary threshold : threshold value above with, target values are fixed to 1
    #---------------------------------------------------------------------------
    'binary_threshold' : 0.0,\
}

#-------------------------------------------------------------------------------
# Configuration for Keras
#-------------------------------------------------------------------------------
dict_param_keras={
    #---------------------------------------------------------------------------
    # input_dim value is defined into build_cnn_datagenerator function.
    # It depends from DataPreparator_v2 object.
    #---------------------------------------------------------------------------
    'input_dim' : (250, 100),\
    'batch_size' : 500,\
    'nb_epoch' : 10,\
    'verbose' : 1,\
    'dropout_rate' : 0.3,\
    'regul':(None,None),\
    'is_batch_normalized' : True,\
    #'lr' : 5.e-5,\
    'lr' : 1.e-3,\
    'nbClasses':2,\
}


dict_param_keras_rnn={\
    #------------------------------------------------------------------------
    # common hyper-parameters for NN architectures.
    #------------------------------------------------------------------------
    'dict_param_keras' : dict_param_keras,\
    #------------------------------------------------------------------------
    # Specific hyper-parameters for RNN architecture.
    #------------------------------------------------------------------------
    'lstm_out' : 1,\
    'cell_units' : 8,\
    'rnn_layers' : 2,\
    'rnnCellType' : 'GRU',\
}


dict_param_keras_cnn={\
    #------------------------------------------------------------------------
    # common hyper-parameters for NN architectures.
    #------------------------------------------------------------------------
    'dict_param_keras' : dict_param_keras,\
    
    #------------------------------------------------------------------------
    # Specific hyper-parameters for RNN architecture.
    #------------------------------------------------------------------------
    'filter_size': 3,\
    
    #------------------------------------------------------------------------
    # Strides 2 : convolution window is shifted by 'stride_size' steps 
    #------------------------------------------------------------------------    
    'stride_size':1,\
    #------------------------------------------------------------------------
    # Invariance to localtranslation can be a useful property if we care more 
    # about whether some feature is present than exactly where it is.
    # Learned funtion is invariant to translations.
    #------------------------------------------------------------------------
    'pool_size':2,\
    'pool_stride' :2,\
    'nb_filter':256,\
    'conv_layer':4,\
    'nb_dense_neuron':128,\
    'dense_layer':1,\

    #------------------------------------------------------------------------
    # Number of layers is decreased for each new layer.
    #------------------------------------------------------------------------
    'dense_layer_decrease_rate' : 1,
    #'list_channel':list(),\
    'list_channel':dict_param_generator['list_keras_channel'],\
    'list_filter_channel':list(),\
}

