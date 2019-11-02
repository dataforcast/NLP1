'''This is a global configuration file for parameters involved into ADANET 
networks.
'''
import tensorflow as tf

IS_DEBUG = True

#-------------------------------------------------------------------------------
# Classification pb / Regression pb
#-------------------------------------------------------------------------------
IS_CLASSIFIER = False
if IS_CLASSIFIER is True :
    IS_REGRESSOR = False
else :
    IS_REGRESSOR = True
        

#-------------------------------------------------------------------------------
# Dataset source
#-------------------------------------------------------------------------------
#DATASET_TYPE='P7'
#DATASET_TYPE='MNIST'
DATASET_TYPE='JIGSAW'

#-------------------------------------------------------------------------------
# Dataset configuration
#-------------------------------------------------------------------------------
IS_LABEL_ENCODED = True # Labels are re-encoded from one hot encoding to 
                        # classes values. 
                        # Apply with Classification pb only
 
#-------------------------------------------------------------------------------
# They are 414 data 
# Batch size is 138
# Then there is 414/138 = 3 steps for covering all data.
# Then 1  EPOCH contains 3 steps.
# If NUM EPOCH is 20, then data will be processed 20 times, each time within 3 steps.
#-------------------------------------------------------------------------------

# For ADANET, this is the number of iterations that will take place.
TRAIN_STEPS = 10


BATCH_SIZE = 138#138//4

#NUM_EPOCHS = 20
# This sequence is oriented to baseline, running
# over NUM_EPOCHS number considering TRAIN_STEPS

#-------------------------------------------------------------------------------
MINI_BATCH_SIZE=138
DATA_SIZE=414
MINI_BATCH_NUMBER = DATA_SIZE//MINI_BATCH_SIZE
NUM_EPOCHS = TRAIN_STEPS//MINI_BATCH_NUMBER
#-------------------------------------------------------------------------------

LEARNING_RATE = 1.e-3


NN_TYPE = 'DNN'
NN_TYPE = 'CNN'
NN_TYPE = 'CNNBase'
NN_TYPE = 'RNN'
RNN_CELL_TYPE = 'RNN'
RNN_CELL_TYPE =  'LSTM'
RNN_CELL_TYPE = 'SLSTM'
RNN_CELL_TYPE =  'GRU'
RNN_CELL_TYPE = 'SGRU'


NN_TYPE = 'RNN'
RNN_CELL_TYPE = 'SGRU'

DNN_HIDDEN_UNITS=128
DNN_NUM_LAYERS = 6

IS_BATCH_NORM = True
DROPOUT_RATE = 0.0

#-------------------------------------------------------------------------------
# When CONV_NUM_LAYERS value is None, then conv. layers will growth with number 
# of layers provided from NNGenerator.
# Otherwise, CNN is built at each Adanet iteration with same number of conv. 
# layers.
# For CNN baseline, value has to be >0 and NN type to be fixed to CNNBase.
#-------------------------------------------------------------------------------
#CNN_CONV_LAYER_NUM  = None
CNN_CONV_LAYER_NUM  = 3

#-------------------------------------------------------------------------------

CNN_KERNEL_SIZE=(5,5)
CNN_FILTERS = 32
CNN_STRIDES =1
CNN_PADDING_NAME ='same'
CNN_ACTIVATION_NAME = 'relu'
CNN_DENSE_UNIT_SIZE = 128
#NN_NUM_LAYERS   = DENSE_NUM_LAYERS+CONV_NUM_LAYERS
#-------------------------------------------------------------------------------
# When None, then dense layer will growth with number of layers 
# provided from NNGenerator
# Otherwise, CNN is built at each Adanet iteration with same number of dense 
# layers.
#-------------------------------------------------------------------------------
#CNN_DENSE_LAYER_NUM = 2
CNN_DENSE_LAYER_NUM = None
#-------------------------------------------------------------------------------


# The random seed to use.
RANDOM_SEED = 42
SEED = RANDOM_SEED

INITIALIZER_NAME = 'xavier'
#-------------------------------------------------------------------------------
# In case of DNN network
#-------------------------------------------------------------------------------
if NN_TYPE == 'DNN' :
    # To be checked
    OPTIMIZER=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)


#-------------------------------------------------------------------------------
# In case of CNN network
#-------------------------------------------------------------------------------
if NN_TYPE == 'CNN' or NN_TYPE == 'CNNBase':
    OPTIMIZER=tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)

#-------------------------------------------------------------------------------
# RNN Network
#-------------------------------------------------------------------------------
RNN_ACTIVATION_NAME = None
RNN_HIDDEN_UNITS = 128
RNN_NUM_LAYERS = 1

if NN_TYPE == 'RNN' :
    OPTIMIZER=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    INITIALIZER_NAME = 'truncated_normal'

#-------------------------------------------------------------------------------
# Adanet hyper-parameters
#-------------------------------------------------------------------------------
ADANET_FEATURE_SHAPE = (224,224*3)
if DATASET_TYPE == 'MNIST':
    ADANET_FEATURE_SHAPE = (28,28)
    FEATURES_KEY = 'images'
    NB_CLASS = 10
    RNN_TIMESTEPS = 28
elif DATASET_TYPE == 'JIGSAW':
    ADANET_FEATURE_SHAPE = (1,37)
    FEATURES_KEY = 'texts'
    NB_CLASS = 1
    RNN_TIMESTEPS = 37 
    SAMPLING_RATIO = 0.01
elif DATASET_TYPE == 'P7' :
    ADANET_FEATURE_SHAPE = (224,224*3)
    FEATURES_KEY = 'images'
    NB_CLASS = 3
    RNN_TIMESTEPS = 224 
else :
    pass


ADANET_OUTPUT_DIR='./tmp/adanet'
ADANET_INITIAL_NUM_LAYERS = 0
ADANET_NN_CANDIDATE = 2
ADANET_LAMBDA = 1.e-5#0.005
ADANET_ITERATIONS = 40  #@param {type:"integer"}
ADANET_IS_LEARN_MIXTURE_WEIGHTS = True
if NN_TYPE == 'RNN' :
    ADANET_INITIAL_NUM_LAYERS = 1
    #---------------------------------------------------------------------------
    # When True, then mixture weights are learned from cost function 
    # optimization, with or without L1 regularization.
    # Otherwise, mixture weights are averaged then are assigned the same value 
    # for any subnetwork forming ensemble.
    #---------------------------------------------------------------------------
    ADANET_IS_LEARN_MIXTURE_WEIGHTS = True
#-------------------------------------------------------------------------------
# Every TRAIN_STEPS then a new candidate will be generated
# Max number of ADANET iterations is 30//3 = 3
# Then AdaNet will build 3 subnetwoks :
# Step 1 : 1 Dense layer (Linear)
# Step 2 : 1 Conv layer + 1 Dense layer 
# Step 3 : 2 Conv layers + 1 dense layer
#-------------------------------------------------------------------------------
# There will be TRAIN_STEPS global steps within ADANET_ITERATIONS
# This mean Adanet will test ADANET_ITERATIONS couple of subnetworks.
# This is the Adanet train step per adanet iterations for learning mixture 
#   weights
ADANET_MAX_ITERATION_STEPS=TRAIN_STEPS//ADANET_ITERATIONS

#-------------------------------------------------------------------------------
# These attributes will be updated after NNAdaNetBuilder object is created.
#-------------------------------------------------------------------------------
feature_columns = None
tf_head = None

#---------------------------------------------------
# Hyper parameters for RNN network
#---------------------------------------------------
dict_rnn_layer_config={ 'rnn_layer_num':RNN_NUM_LAYERS
                       ,'rnn_hidden_units' : RNN_HIDDEN_UNITS
                       ,'rnn_timesteps' : RNN_TIMESTEPS
                       ,'rnn_cell_type' : RNN_CELL_TYPE
                    }


#---------------------------------------------------
# Hyper parameters for CNN network
#---------------------------------------------------
dict_cnn_layer_config={ 'feature_map_size':[64,]
                      ,'cnn_kernel_size':CNN_KERNEL_SIZE
                      ,'cnn_conv_layer_num':CNN_CONV_LAYER_NUM
                      ,'cnn_filters' : CNN_FILTERS
                      ,'cnn_strides' :  CNN_STRIDES
                      ,'cnn_padding_name' : CNN_PADDING_NAME
                      ,'cnn_activation_name' : CNN_ACTIVATION_NAME
                      ,'cnn_dense_layer_num' : CNN_DENSE_LAYER_NUM
                      ,'cnn_dense_unit_size' : CNN_DENSE_UNIT_SIZE
                      }
#---------------------------------------------------
# Hyper parameters for DNN network
#---------------------------------------------------
dict_dnn_layer_config={ 'dnn_layer_num':DNN_NUM_LAYERS
                       ,'dnn_hidden_units' : DNN_HIDDEN_UNITS
                    }

#---------------------------------------------------
# Hyper parameters for NN Builder
#---------------------------------------------------
dict_nn_layer_config = {  'nn_type':NN_TYPE
                        , 'nn_dropout_rate': DROPOUT_RATE
                        , 'nn_batch_norm' : IS_BATCH_NORM
                        , 'nn_logit_dimension':NB_CLASS
                        , 'nn_optimizer':OPTIMIZER
                        , 'nn_seed' : SEED
                        , 'nn_initializer_name' : INITIALIZER_NAME
                        , 'nn_layer_config':dict_cnn_layer_config
                       }
if NN_TYPE == 'RNN' :
    dict_nn_layer_config['nn_layer_config'] = dict_rnn_layer_config
if NN_TYPE == 'DNN' :    
    dict_nn_layer_config['nn_layer_config'] = dict_dnn_layer_config

#---------------------------------------------------
# AdaNet configuration
#---------------------------------------------------
dict_adanet_config = {  'adanet_feature_columns': feature_columns
                      , 'adanet_feature_shape' : ADANET_FEATURE_SHAPE
                      , 'adanet_tf_head' : tf_head
                      , 'adanet_lambda' : ADANET_LAMBDA
                      , 'adanet_is_learn_mixture_weights': ADANET_IS_LEARN_MIXTURE_WEIGHTS
                      , 'adanet_initial_num_layers': ADANET_INITIAL_NUM_LAYERS
                      , 'adanet_num_layers': None
                      , 'adanet_nn_candidate' : ADANET_NN_CANDIDATE
                      , 'adanet_lambda' : ADANET_LAMBDA
                      , 'adanet_output_dir':ADANET_OUTPUT_DIR
                      , 'adanet_nn_layer_config' : dict_nn_layer_config
                      }

