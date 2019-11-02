import os
import numpy as np
import tensorflow as tf
import keras
from tensorflow.contrib import rnn

import adanet

import p8_util
import p8_util_config

IS_DEBUG = p8_util_config.IS_DEBUG

#-------------------------------------------------------------------------------
# Estimator configuration.
#-------------------------------------------------------------------------------
class NNAdaNetBuilder(adanet.subnetwork.Builder) :
    '''Builds a NN subnetwork for AdaNet.'''

    def __init__(self, dict_adanet_config, num_layers, summary=None):
        """Initializes a `_DNNBuilder`.

        Args:
          dict_adanet_config : dictionary for configuration of AdaNet algo.
          num_layers: The number of hidden layers.


        Returns:
          An instance of `NNAdaNetBuilder`.
        """
        self._dict_adanet_config = dict_adanet_config.copy()
        self._summary = summary
        self._last_layer = None
        self._feature_shape =None# dict_adanet_config['adanet_feature_columns']
        self._learn_mixture_weights = dict_adanet_config['adanet_is_learn_mixture_weights']
        self._adanet_lambda = dict_adanet_config['adanet_lambda']
        self._output_dir = dict_adanet_config['adanet_output_dir']
        self._classifier_config = None
        #---------------------------------------------------
        # Hyper parameters for NN Builder
        #---------------------------------------------------
        dict_nn_layer_config= dict_adanet_config['adanet_nn_layer_config']
        
        self._nn_type    = dict_nn_layer_config['nn_type']
        self._optimizer  = dict_nn_layer_config['nn_optimizer']
        #self._layer_size = dict_nn_layer_config['nn_dense_unit_size']
        self._num_layers = num_layers
        
        self._dropout    = dict_nn_layer_config['nn_dropout_rate']
        self._seed       = dict_nn_layer_config['nn_seed']
        self._nb_class   = dict_nn_layer_config['nn_logit_dimension']
        
        # When value is None, then HE normal initializer is used as default. 
        self._layer_initializer_name = dict_nn_layer_config['nn_initializer_name']
        
        # Batch normalization activation
        self._is_nn_batch_norm = dict_nn_layer_config['nn_batch_norm']
        
        #---------------------------------------------------
        # Hyper parameters for CNN network
        #---------------------------------------------------
        if self._nn_type == 'CNN' or self._nn_type == 'CNNBase':
            # Strategy for growth convolutional layers or dense layers 
            # are fixed. 
            # If configuration holds None for dense layers, then dense layers 
            # will growth.
            
            dict_cnn_layer_config = dict_nn_layer_config['nn_layer_config']
            if False :
                if dict_cnn_layer_config['cnn_conv_layer_num'] is not None :
                    dict_cnn_layer_config['cnn_conv_layer_num'] = num_layers
                    #dict_cnn_layer_config['cnn_dense_layer_num'] = None
                elif dict_cnn_layer_config['cnn_dense_layer_num'] is not None :
                    dict_cnn_layer_config['cnn_dense_layer_num'] = num_layers
                    #dict_cnn_layer_config['cnn_conv_layer_num'] = None
            # Fixed CNN layers configuration 
            self._dict_cnn_layer_config=dict_cnn_layer_config.copy()

            self._cnn_seed = self._seed

            self._cnn_kernel_size = dict_cnn_layer_config['cnn_kernel_size']
            
            #-------------------------------------------------------------------  
            # Number of CNN CONVOLUTIONAL layers
            # In case of AdaNet, then number of layers is provided while 
            # constructor is invoked.
            # In case of baseline calibration, then number of conv. layers 
            # is extracted from a CNN dictionary and provided while invoking 
            # constructor.
            #-------------------------------------------------------------------
            self._cnn_dense_layer_num = dict_cnn_layer_config['cnn_dense_layer_num']            
            self._cnn_convlayer = dict_cnn_layer_config['cnn_conv_layer_num']            
            
            # Number of units in CNN dense layer.
            self._cnn_dense_unit_size = dict_cnn_layer_config['cnn_dense_unit_size']
            
            
            # Batch normaization activation
            self._is_cnn_batch_norm = self._is_nn_batch_norm
        elif self._nn_type == 'RNN' :
            dict_nn_layer_config['nn_layer_config']['rnn_layer_num'] = num_layers
            self._dict_rnn_layer_config = dict_nn_layer_config['nn_layer_config'].copy()
        elif self._nn_type == 'DNN' :
            dict_nn_layer_config['nn_layer_config']['dnn_layer_num'] = num_layers
            self._dict_dnn_layer_config = dict_nn_layer_config['nn_layer_config'].copy()
        else : 
            pass

        self._start_time = 0
        
        if IS_DEBUG is True :        
            print("\n*** NNAdaNetBuilder : NN Type={}".format(self._nn_type))
        

        
    #---------------------------------------------------------------------------
    #   Properties
    #---------------------------------------------------------------------------
    def _get_feature_shape(self) :
       return self._feature_shape
       
    def _set_feature_shape(self, feature_shape) :
    
        self._feature_shape = feature_shape
        nb_class = self._nb_class
        nn_type = self._nn_type
        
        feature_columns, loss_reduction, tf_head \
        = p8_util.get_tf_head(feature_shape, nb_class, nn_type=nn_type, feature_shape=feature_shape)

        self._feature_columns = feature_columns
        
    def _get_feature_columns(self) :
       return self._feature_columns
    def _set_feature_columns(self, feature_columns) :
        print('\n*** ERROR :  \`feature_columns\` is not assignable!')

    def _get_nb_class(self) :
       return self._nb_class
    def _set_nb_class(self, nb_class) :
        print("\n*** ERROR : assignement is forbidden for this parameter!")

    def _get_optimizer(self) :
       return self._optimizer
    def _set_optimizer(self, optimizer) :
        print("\n*** ERROR : assignement is forbidden for this parameter!")

    def _get_output_dir(self) :
       return self._output_dir

    def _set_output_dir(self, output_dir) :        
        self._output_dir = os.path.join(output_dir, self._nn_type)
        
        if False:
            if 'RNN' == self._nn_type :
                dict_nn_layer_config= self._dict_rnn_layer_config
                rnn_cell_type = dict_nn_layer_config['rnn_cell_type']
                model_name = rnn_cell_type
            elif  'CNN' == self._nn_type or 'CNNBase' == self._nn_type :
                if self._dict_cnn_layer_config['cnn_dense_layer_num'] is None :
                    model_name = self._nn_type+'DENSE'
                elif self._dict_cnn_layer_config['cnn_conv_layer_num'] is None :
                    model_name = self._nn_type+'CONV'            
                else :
                    model_name = self._nn_type    
            else : 
                model_name = self._nn_type

        model_name = p8_util.build_model_name(self._nn_type)
        self._classifier_config , self._output_dir_log= p8_util.make_config(model_name\
                                        , output_dir=self._output_dir\
                                        , is_restored=False)

    def _get_classifier_config(self) :
       return self._classifier_config
    def _set_classifier_config(self, set_classifier_config) :
        print("\n*** ERROR : assignement is forbidden for this parameter!")

    def _get_output_dir_log(self) :
       return self._output_dir_log
    def _set_output_dir_log(self, output_dir_log) :
        print("\n*** ERROR : assignement is forbidden for this parameter!")


    output_dir_log = property(_get_output_dir_log,_set_output_dir_log)
    output_dir = property(_get_output_dir,_set_output_dir)
    classifier_config = property(_get_classifier_config,_set_classifier_config)
    feature_shape = property(_get_feature_shape,_set_feature_shape)
    nb_class = property(_get_nb_class,_set_nb_class)
    optimizer = property(_get_optimizer, _set_optimizer)
    feature_columns = property(_get_feature_columns, _set_feature_columns)

    #----------------------------------------------------------------------------
    #
    #----------------------------------------------------------------------------
    def _show_cnn(self) :
        print('\n')
        print("CNN seed             : ............................ {}".format(self._cnn_seed))
        print("Conv. Kernel size    : ............................ {}".format(self._cnn_kernel_size))
        print("Conv. layers         : ............................ {}".format(self._dict_cnn_layer_config['cnn_conv_layer_num']))
        print("Dense layers         : ............................ {}".format(self._dict_cnn_layer_config['cnn_dense_layer_num']))
        print("Units in dense layers: ............................ {}".format(self._cnn_dense_unit_size))
        print("CNN bacth norm.      : ............................ {}".format(self._is_cnn_batch_norm))
        print("Features map size    : ............................ {}".format(self._dict_cnn_layer_config['feature_map_size']))
        print("Conv filters         : ............................ {}".format(self._dict_cnn_layer_config['cnn_filters']))
        print("Strides              : ............................ {}".format(self._dict_cnn_layer_config['cnn_strides']))
        print("Padding              : ............................ {}".format(self._dict_cnn_layer_config['cnn_padding_name']))
        print("Activation function  : ............................ {}".format(self._dict_cnn_layer_config['cnn_activation_name']))

    #----------------------------------------------------------------------------
    #
    #----------------------------------------------------------------------------
    def _show_rnn(self) :
        print('\n') 
        print("Cell type            : ............................ {}".format(self._dict_rnn_layer_config['rnn_cell_type']))
        print("Hidden units         : ............................ {}".format(self._dict_rnn_layer_config['rnn_hidden_units']))
        print("Stacked cells        : ............................ {}".format(self._dict_rnn_layer_config['rnn_layer_num']))
        print("Time steps           : ............................ {}".format(self._dict_rnn_layer_config['rnn_timesteps']))
    #----------------------------------------------------------------------------

    #----------------------------------------------------------------------------
    #
    #----------------------------------------------------------------------------
    def _show_dnn(self) :
        print('\n') 
        print("Number of layers     : ............................ {}".format(self._dict_dnn_layer_config['dnn_layer_num']))
        print("Hidden units         : ............................ {}".format(self._dict_dnn_layer_config['dnn_hidden_units']))
    #----------------------------------------------------------------------------
    
    #----------------------------------------------------------------------------
    #
    #----------------------------------------------------------------------------
    def show(self) :
        adanet_max_iteration_steps =  p8_util_config.ADANET_ITERATIONS
        adanet_global_steps = p8_util_config.TRAIN_STEPS
        adanet_iter_per_booting = p8_util_config.ADANET_MAX_ITERATION_STEPS
        print("\n")
        print("Global steps  : ................................... {}".format(adanet_global_steps))
        print("NN type              : ............................ {}".format(self._nn_type))
        print("Features shape       : ............................ {}".format(self._feature_shape))
        print("Adanet outputdir     : ............................ {}".format(self._output_dir))
        print("Adanet output log    : ............................ {}".format(self._output_dir_log))
        print("Adanet boosting iter.: ............................ {}".format(adanet_max_iteration_steps))
        print("Adanet iter per boost: ............................ {}".format(adanet_iter_per_booting))
        #print("Units in dense layer : ............................ {}".format(self._layer_size))
        print("Number of layers     : ............................ {}".format(self._num_layers))
        print("Dropout rate         : ............................ {}".format(self._dropout))
        print("Seed value           : ............................ {}".format(self._seed))
        print("Nb of classes (logit): ............................ {}".format(self._nb_class))
        print("Adanet regularization: ............................ {}".format(self._adanet_lambda))
        print("Weights initializer  : ............................ {}".format(self._layer_initializer_name))
        print("Batch normalization  : ............................ {}".format(self._is_nn_batch_norm))
        print("Learn mixture weights: ............................ {}".format(self._learn_mixture_weights))
        if self._nn_type == 'CNNBase' or self._nn_type == 'CNN' :
            self._show_cnn()
        elif self._nn_type == 'RNN':
            self._show_rnn()
        elif self._nn_type == 'DNN':
            self._show_dnn()
        else :
            pass
        print("\n\n")
    #----------------------------------------------------------------------------
        
    #----------------------------------------------------------------------------
    #
    #----------------------------------------------------------------------------
    def _build_dnn_subnetwork(self, input_layer, features, logits_dimension, is_training) :
    
        last_layer = input_layer

        dnn_layer_num = self._dict_dnn_layer_config['dnn_layer_num']
    
        if IS_DEBUG is True :
            print("**** *** _build_dnn_subnetwork : Layers= {}\n".format(dnn_layer_num))
        for i_ in range(dnn_layer_num):
            #print("\n**** *** _build_dnn_subnetwork : Layer= {} / Layers= {}".format(i_, self._num_layers))
            last_layer = tf.layers.dense(
                last_layer,
                units=self._dict_dnn_layer_config['dnn_hidden_units'],
                activation=tf.nn.relu,
                kernel_initializer=tf.glorot_uniform_initializer(seed=self._seed))
            if self._dropout > 0. :
                last_layer = tf.layers.dropout(
                    last_layer, rate=self._dropout, seed=self._seed, training=is_training)
        
        
        self._last_layer = last_layer
        if IS_DEBUG is True :
            print("\n*** _build_dnn_subnetwork : last layer= {}".format(self._last_layer))
        
        logits = tf.layers.dense(
            last_layer,
            units=logits_dimension,
            kernel_initializer=tf.glorot_uniform_initializer(seed=self._seed))
            
        
        
        
        
        return last_layer, logits
    #---------------------------------------------------------------------------
            
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _cnn_bacth_norm(self, layer, is_training) :
        if self._is_cnn_batch_norm is True :                      
            layer = tf.layers.batch_normalization(layer, training=is_training)
        return layer
    #---------------------------------------------------------------------------
        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _get_layer_initializer(self) :
        if self._layer_initializer_name == 'xavier' :
            layer_initializer = tf.contrib.layers.xavier_initializer
        elif self._layer_initializer_name == 'truncated_normal' :
            layer_initializer = tf.truncated_normal_initializer(stddev=0.01)
        else : 
            print("\*** WARN : default initializer defined : HE NORMAL")
            layer_initializer = tf.keras.initializers.he_normal
        return layer_initializer
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _get_conv_activation_fn(self, cnn_activation_name) :
        conv_activation_fn = None
        if cnn_activation_name == 'relu' :
            conv_activation_fn =tf.nn.relu
        else : 
            print("\n*** ERROR : activation function name= {} Unknown!".format(cnn_activation_name))
        return conv_activation_fn
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def nn_layertype_alternate(self) :
        conv_layer = 0
        dense_layer = 0
        if self._cnn_convlayer == 0 : 
            conv_layer = self._cnn_convlayer
            dense_layer = self._cnn_dense_layer_num
        else :
            if self._cnn_convlayer %2 == 0 : 
                conv_layer = self._cnn_convlayer
                dense_layer = 0
            else :
                conv_layer = 0
                dense_layer = self._cnn_dense_layer_num
        return conv_layer, dense_layer
    #---------------------------------------------------------------------------
        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _build_cnn_baseline_subnetwork(self, input_layer, features
    , logits_dimension, is_training) :
        
        last_layer = input_layer
        
        if IS_DEBUG is True :
            print("\n*** _build_cnn_baseline_subnetwork() : last_layer= {}".format(last_layer))
            print("\n*** _build_cnn_baseline_subnetwork() : features[images]= {}".format(features['images']))

        tuple_cnn_kernel_size = self._dict_cnn_layer_config['cnn_kernel_size']
        cnn_filters           = self._dict_cnn_layer_config['cnn_filters']
        cnn_strides           = self._dict_cnn_layer_config['cnn_strides']
        cnn_padding_name      = self._dict_cnn_layer_config['cnn_padding_name']
        cnn_activation_name   = self._dict_cnn_layer_config['cnn_activation_name']
        
        layer_initializer = self._get_layer_initializer()
        conv_activation_fn = self._get_conv_activation_fn(cnn_activation_name)
        
        #-----------------------------------------------------------------------                
        # Build alternatively conv and dense layers 
        #----------------------------------------------------------------------- 
        cnn_conv_layer  = 0   
        #conv_layer, dense_layer = self.nn_layertype_alternate()        
        
        
        
        #model = keras.models.Sequential(name='ConvAdanet')
        #-----------------------------------------------------------------------                
        # Convolutional Layers
        #-----------------------------------------------------------------------                        
        if self._dict_cnn_layer_config['cnn_conv_layer_num'] is None :
            range_layer = range(self._num_layers)
        else :
            range_layer = range(self._dict_cnn_layer_config['cnn_conv_layer_num'])
          
        last_layer =  features['images']
        last_layer = list(features.values())[0]
        print("\**** Input layer= {}".format(last_layer))
        for layer in range_layer :     
            last_layer = self._cnn_bacth_norm(last_layer, is_training)
            print("\**** Last layer= {} / {}".format(layer,last_layer))
            if True :
                last_layer = tf.layers.conv2d(last_layer
                                        , filters=cnn_filters
                                        , kernel_size=tuple_cnn_kernel_size
                                        , strides=cnn_strides
                                        , padding=cnn_padding_name
                                        , activation=conv_activation_fn
                                        , kernel_initializer=layer_initializer())
                #C = (I-F+2P)/S +1
                # 28-5+1 = 24
                pool_size = (2, 2)
                last_layer = tf.layers.max_pooling2d(inputs=last_layer, pool_size= pool_size , strides=2)
            else :
                last_layer = keras.layers.Conv2D(cnn_filters   
                , kernel_size=tuple_cnn_kernel_size
                , strides=cnn_strides
                , padding=cnn_padding_name
                , data_format=None
                , dilation_rate=(1, 1)
                , activation=conv_activation_fn
                , use_bias=True
                , kernel_initializer='glorot_uniform'
                , bias_initializer='zeros'
                , kernel_regularizer=None
                , bias_regularizer=None
                , activity_regularizer=None
                , kernel_constraint=None
                , bias_constraint=None)(last_layer)
                last_layer = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(last_layer)
    


        #-----------------------------------------------------------------------                
        # Flat Layer
        #-----------------------------------------------------------------------
        last_layer = tf.contrib.layers.flatten(last_layer)
                  
        #-----------------------------------------------------------------------                
        # Dense Layer(s) : when config is None, then they growth step by step.
        #-----------------------------------------------------------------------
        if self._dict_cnn_layer_config['cnn_dense_layer_num'] is None :
            range_layer = range(self._num_layers)
        else :
            range_layer = range(self._dict_cnn_layer_config['cnn_dense_layer_num'])
                
        for layer in range_layer :     
            last_layer = self._cnn_bacth_norm(last_layer, is_training)
            last_layer = tf.layers.dense( inputs=last_layer
                                        , units=self._cnn_dense_unit_size 
                                        , activation=conv_activation_fn
                                        , kernel_initializer=layer_initializer())
            if self._dropout > 0. :            
                last_layer = tf.layers.dropout(   inputs=last_layer
                                                , rate=self._dropout
                                                , training=is_training)

        #-----------------------------------------------------------------------                
        # Logits Layer
        #-----------------------------------------------------------------------                
        logits = tf.layers.dense( inputs=last_layer
                                , units=self._nb_class
                                , kernel_initializer=layer_initializer())

        return last_layer,logits
    #---------------------------------------------------------------------------
        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _build_rnn_subnetwork(self, input_layer, features\
    , logits_dimension, is_training,rnn_cell_type='RNN') :
        '''Builds RNN subnetwork depending given RNN type as parameter.
        
        Input :
            *   input_layer : tensor flow colums of features gained with 
            tf.feature_column.input_layer
            *   features : 
        '''
    
        last_layer = input_layer
        logits = None        

        layer_initializer = self._get_layer_initializer()
        num_hidden_units = self._dict_rnn_layer_config['rnn_hidden_units']
        
        # They are weights output 
        shape=[num_hidden_units, logits_dimension]
        weight = tf.get_variable('W', dtype=tf.float32, shape=shape\
        , initializer=layer_initializer)
        
        # They are bias output 
        bias = tf.get_variable('b',
                   dtype=tf.float32,
                   initializer=tf.constant(0., shape=[logits_dimension], dtype=tf.float32))
        
        timesteps = self._dict_rnn_layer_config['rnn_timesteps']

        raws = p8_util_config.ADANET_FEATURE_SHAPE[0]
        cols = p8_util_config.ADANET_FEATURE_SHAPE[1]
        #last_layer = tf.reshape(last_layer, [-1,raws*cols])   
        #list_layer = tf.unstack(last_layer, raws, 1)
        if True:
            if p8_util_config.DATASET_TYPE == 'P7' :
                last_layer = tf.reshape(last_layer, [-1,raws,cols])   
                list_layer = tf.unstack(last_layer, raws, 1)
            elif p8_util_config.DATASET_TYPE == 'MNIST' :
                last_layer = tf.reshape(last_layer, [-1,raws,cols])   
                list_layer = tf.unstack(last_layer, raws, 1)
            elif p8_util_config.DATASET_TYPE == 'JIGSAW' :
                last_layer = tf.reshape(last_layer, [-1,raws,cols])   
                list_layer = tf.unstack(last_layer, raws, 1)
            else :
                pass
        
        # Define a rnn cell with tensorflow
        if 'RNN' == rnn_cell_type :
            rnn_cell = tf.keras.layers.SimpleRNNCell(num_hidden_units)
        elif 'LSTM' == rnn_cell_type :
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden_units)
        elif 'GRU' == rnn_cell_type :
            rnn_cell = tf.keras.layers.GRUCell(num_hidden_units)
        elif 'SGRU' == rnn_cell_type :
            stacked_cell_number = self._dict_rnn_layer_config['rnn_layer_num']
            
            #-------------------------------------------------------------------
            # All cells to be stacked have same number of units
            #-------------------------------------------------------------------
            list_stacked_cell = [num_hidden_units for _ in range(0,stacked_cell_number)]
            
            #-------------------------------------------------------------------
            # Cells are stacked
            #-------------------------------------------------------------------
            list_rnn_cell = [tf.contrib.rnn.GRUCell(num_units=n) for n in list_stacked_cell]            
            rnn_cell = tf.contrib.rnn.MultiRNNCell(list_rnn_cell)
        elif 'SLSTM' == rnn_cell_type :
            stacked_cell_number = self._dict_rnn_layer_config['rnn_layer_num']
            
            #-------------------------------------------------------------------
            # All cells to be stacked have same number of units
            #-------------------------------------------------------------------
            list_stacked_cell = [num_hidden_units for _ in range(0,stacked_cell_number)]
            
            #-------------------------------------------------------------------
            # Cells are stacked
            #-------------------------------------------------------------------
            list_rnn_cell = [tf.contrib.rnn.LSTMCell(num_units=n) for n in list_stacked_cell]            
            rnn_cell = tf.contrib.rnn.MultiRNNCell(list_rnn_cell)
        elif 'SRNN' == rnn_cell_type :
            stacked_cell_number = self._dict_rnn_layer_config['rnn_layer_num']
            
            #-------------------------------------------------------------------
            # All cells to be stacked have same number of units
            #-------------------------------------------------------------------
            list_stacked_cell = [num_hidden_units for _ in range(0,stacked_cell_number)]
            
            #-------------------------------------------------------------------
            # Cells are stacked
            #-------------------------------------------------------------------
            list_rnn_cell = [tf.keras.layers.SimpleRNNCell(num_units=n) for n in list_stacked_cell]            
            rnn_cell = tf.contrib.rnn.MultiRNNCell(list_rnn_cell)
        else :
            print("\n*** ERROR : Recurrent Network type= {} NOT YET SUPPORTED!"\
            .format(rnn_cell_type))
            return None, None

        # Get cell output
        # If no initial_state is provided, dtype must be specified
        # If no initial cell state is provided, they will be initialized to zero
        output, last_layer = rnn.static_rnn(rnn_cell, list_layer, dtype=tf.float32)
        if IS_DEBUG is True :
            print("\n*** _build_rnn_subnetwork() : output[-1]= {} / Weight= {}"\
            .format(output[-1], weight))
        logits =tf.matmul(output[-1], weight) + bias

        # Linear activation, using rnn inner loop last output
        last_layer = tf.layers.dense(logits, units=num_hidden_units,
            kernel_initializer=tf.glorot_uniform_initializer(seed=self._seed))
        return last_layer,logits
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _build_cnn_subnetwork(self, input_layer, features\
    , logits_dimension, is_training) :
    
        if self._nn_type == 'CNNBase' or self._nn_type == 'CNN':
            return self._build_cnn_baseline_subnetwork(input_layer, features\
                                            , logits_dimension, is_training)
        else :
            pass
            
        features_key = list(features.keys())[0]
        w = features[features_key].shape[1]
        h = features[features_key].shape[2]
        c = features[features_key].shape[3]
        last_layer = input_layer
        
        layer_initializer = self._get_layer_initializer()
        
        if self._cnn_convlayer > 0 : 
            last_layer =  features['images']       
            for layer in range(self._cnn_convlayer) :
                last_layer = self._cnn_bacth_norm(last_layer, is_training)
                last_layer = tf.layers.conv2d(last_layer, filters=64,
                                      kernel_size=(3,3) , strides=1,
                                      padding='same', activation=tf.nn.relu,
                                      kernel_initializer=layer_initializer())
                last_layer = self._cnn_bacth_norm(last_layer, is_training)
                last_layer = tf.layers.conv2d(last_layer, filters=64,
                                      kernel_size=(3,3), strides=1,
                                      padding='same', activation=tf.nn.relu,
                                      kernel_initializer=layer_initializer())
                pool_size = (2, 2)
                last_layer = tf.layers.max_pooling2d(inputs=last_layer, pool_size= pool_size , strides=2)
                
            last_layer = tf.contrib.layers.flatten(last_layer)
            last_layer = self._cnn_bacth_norm(last_layer, is_training)
            
            last_layer = tf.layers.dense(inputs=last_layer
            , units=self._cnn_dense_unit_size
            , activation=tf.nn.relu, kernel_initializer=layer_initializer())
            
            if self._dropout > 0. :
                last_layer = tf.layers.dropout(inputs=last_layer
                , rate=self._dropout, training=is_training)



            # Process fixed CNN CONV. layers configuration 
            if self._dict_cnn_layer_config['cnn_conv_layer_num'] is not None :
                list_cnn_layer_filter = self._dict_cnn_layer_config['feature_map_size']
                for cnn_layer_filter in list_cnn_layer_filter :
                
                    
                    last_layer = self._cnn_bacth_norm(last_layer, is_training)
                    last_layer = tf.layers.conv2d(last_layer, filters=cnn_layer_filter,
                                          kernel_size=(3,3), strides=1,
                                          padding='same', activation=tf.nn.relu,
                                          kernel_initializer=layer_initializer())
                    pool_size = (2, 2)
                    last_layer = tf.layers.max_pooling2d(inputs=last_layer, pool_size= pool_size , strides=2)

        
            last_layer = tf.contrib.layers.flatten(last_layer)

            for _ in range(self._cnn_dense_layer_num) :
                last_layer = self._cnn_bacth_norm(last_layer, is_training)
                
                last_layer = tf.layers.dense(inputs=last_layer
                , units=self._cnn_dense_unit_size
                , activation=tf.nn.relu, kernel_initializer=layer_initializer())
                
                if self._dropout > 0. :
                    last_layer = tf.layers.dropout(inputs=last_layer
                    , rate=self._dropout, training=is_training)
                     
        
        # Logits Layer
        logits = tf.layers.dense(inputs=last_layer, units=self._nb_class
        ,kernel_initializer=layer_initializer())
        
        return last_layer,logits
   #----------------------------------------------------------------------------


    #----------------------------------------------------------------------------
    #
    #----------------------------------------------------------------------------
    def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
        """See `adanet.subnetwork.Builder`.
        This method is invoked from adanet.subnetwork.Builder.
        It builds a NN and returns a subnetwork candidate.
        Subnetwork candidate is built using last layer of NN subnetwork.
        
        Input : 
            * features : 
            * logits_dimension : 
        
        
        """
        
        #features = self._feature_columns
        if IS_DEBUG is True :
            print("\n\n*** build_subnetwork() : features= {}".format(features))
            print("\n\n*** build_subnetwork() : self._feature_columns= {}".format(self._feature_columns))
        
        input_layer \
        = tf.feature_column.input_layer(features=features\
                                    , feature_columns=self._feature_columns)
        
        if IS_DEBUG is True :
            print("\n\n*** build_subnetwork() : NN type= {} / Input layer shape= {}"\
            .format(self._nn_type, input_layer.shape))

        #-----------------------------------------------------------------------
        # Default complexity value
        #-----------------------------------------------------------------------
        
        if self._nn_type == 'DNN' :
            last_layer, logits \
            = self._build_dnn_subnetwork(input_layer, features\
            , logits_dimension, training)
            complexity = tf.sqrt(tf.to_float(self._num_layers))
            
        elif self._nn_type == 'CNN' or self._nn_type == 'CNNBase' :
            last_layer, logits \
            = self._build_cnn_baseline_subnetwork(input_layer, features\
            , logits_dimension, training)

            #-------------------------------------------------------------------
            # Approximate the Rademacher complexity of this subnetwork as the square-
            # root of its depth; depth includes Conv layers as well as dense layers.
            # Complexity is the sum of current growth layers and fixed layers.
            # Fixed layers may be convolutional or dense layers, depending 
            # configuration values from dictionary (None means layers are 
            # increased from Adanet weaklearner algorithm)
            #-------------------------------------------------------------------
            complexity = self._num_layers

            # Increase complexity with fixed convolutionals layers 
            if self._dict_cnn_layer_config['cnn_dense_layer_num'] is None :
                complexity += self._dict_cnn_layer_config['cnn_conv_layer_num']

            # Increase complexity with fixed denses layers 
            if self._dict_cnn_layer_config['cnn_conv_layer_num'] is None :
                complexity += self._dict_cnn_layer_config['cnn_dense_layer_num']
            
            # Convert number of added  layers to float and apply square root.
            complexity = tf.sqrt(tf.to_float(complexity))
            
        elif self._nn_type == 'RNN' :
            rnn_cell_type = self._dict_rnn_layer_config['rnn_cell_type']
            last_layer, logits = self._build_rnn_subnetwork(  input_layer\
                                                            , features\
                                                            , logits_dimension\
                                                            , training\
                                                            , rnn_cell_type = rnn_cell_type)
            complexity = tf.sqrt(tf.to_float(self._num_layers))
            if IS_DEBUG is True :
                print("\n***build_subnetwork() / RNN : logits shape= {}".format(logits.shape))
            
        else :
            print("\n*** ERROR : NN type={} no yet supported!".format(self._nn_type))
            return None
        

        if self._nn_type == 'CNN' or self._nn_type == 'CNNBase':
            persisted_tensors = {self._nn_type: tf.constant(self._num_layers)}
            print("\n***  build_subnetwork() : persisted_tensors= {}".format(persisted_tensors))
                
        else : 
            persisted_tensors = {self._nn_type: tf.constant(self._num_layers)}
            
        with tf.name_scope(""):
            if complexity is not None :
                summary.scalar("Complexity", complexity)
            summary.scalar("Layers", tf.constant(self._num_layers))
            if self._nn_type == 'CNN' or self._nn_type == 'CNNBase':
                if self._dict_cnn_layer_config['cnn_conv_layer_num'] is None :
                    summary.scalar("Conv_layers", tf.constant(self._num_layers))
                if self._dict_cnn_layer_config['cnn_dense_layer_num'] is None :
                    summary.scalar("Dense_layers", tf.constant(self._num_layers))
                else : 
                    summary.scalar("Layers", tf.constant(self._num_layers))
                    
        
        #print("\n*** persisted_tensors= {}".format(persisted_tensors))
        if IS_DEBUG is True :
            print("\n*** build_subnetwork() : last_layer= {}".format(last_layer))
        return adanet.Subnetwork(
            last_layer=last_layer,
            logits=logits,
            complexity=complexity,
            shared=persisted_tensors)

    def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
        """This method is invoked from method 
        _EnsembleBuilder._build_ensemble_spec  defined in adanet package 
        file ensemble.py.
        It is used to train subnetwork candidates.
        
        See `adanet.subnetwork.Builder`."""

        # NOTE: The `adanet.Estimator` increments the global step.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return self._optimizer.minimize(loss=loss, var_list=var_list)

    def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
        """ This method is invoked from method 
        _EnsembleBuilder._build_ensemble_spec  defined in adanet package 
        file ensemble.py.
        
        See `adanet.subnetwork.Builder`."""

        if not self._learn_mixture_weights:
            return tf.no_op("mixture_weights_train_op")

        # NOTE: The `adanet.Estimator` increments the global step.
        return self._optimizer.minimize(loss=loss, var_list=var_list)
    #----------------------------------------------------------------------------

    @property
    def name(self):
        """See `adanet.subnetwork.Builder`."""
        num_layers = 0
        if self._nn_type == 'CNN' or self._nn_type == 'CNNBase':
            if False :
                if self._dict_cnn_layer_config['cnn_conv_layer_num'] is None :
                    num_layers = self._cnn_convlayer
                else :
                    num_layers = self._num_layers
            num_layers = self._num_layers
        else : 
            num_layers = self._num_layers

        if True :
            if num_layers == 0:
                # No hidden layers is a linear model.
                return "{}_linear".format(self._nn_type)
            else : 
                if 'RNN' == self._nn_type :
                    rnn_cell_type = self._dict_rnn_layer_config['rnn_cell_type']
                    if rnn_cell_type == self._nn_type :
                        nn_type = self._nn_type
                    else :    
                        nn_type = self._nn_type+'_'+str(rnn_cell_type)
                else :
                    nn_type = self._nn_type
                return "{}_layer_{}".format(nn_type, num_layers)
        else :                 
            return "{}_layer_{}".format(self._nn_type, num_layers)

#-------------------------------------------------------------------------------

