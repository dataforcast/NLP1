import time
import tensorflow as tf
import p8_util
import p8_util_config

class BaselineEstimator():
    '''This class implements a TensorFlow Estimator for baseline. 
    '''
    #-------------------------------------------------------------------------------
    #   
    #-------------------------------------------------------------------------------
    def __init__(self, params, IS_DEBUG=False):
        self._features = None
        self._is_debug = IS_DEBUG
        self._net_builder = params['net_builder']
        self._n_class = None
        classifier_config = None
        if self._net_builder is not None :
            if self._net_builder._nb_class is not None :
                self._n_class = self._net_builder._nb_class
            else :
                pass 
            classifier_config = self._net_builder.classifier_config
        else :
            pass
        self._train_spec = None
        self._eval_spec = None
        
        self._estimator = tf.estimator.Estimator(model_fn=self._baseline_model_fn\
                                                  , params=params\
                                                  , config=classifier_config)
    #-------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------
    #   
    #-------------------------------------------------------------------------------
    def train_and_evaluate(self):
        start_time = time.time()
        
        input_fn_param={'num_epochs':p8_util_config.NUM_EPOCHS,\
                'batch_size': p8_util_config.BATCH_SIZE,\
                'feature_shape': p8_util_config.ADANET_FEATURE_SHAPE,\
                'dataset_type': p8_util_config.DATASET_TYPE,\
               }
               
        self._train_spec=tf.estimator.TrainSpec(
                input_fn=p8_util.input_fn_2("train", input_fn_param),\
                max_steps=p8_util_config.TRAIN_STEPS)

        self._eval_spec=tf.estimator.EvalSpec(
                input_fn=p8_util.input_fn_2("test",  input_fn_param),\
                steps=None,\
                throttle_secs=1)    
        
        results, _ = tf.estimator.train_and_evaluate(self._estimator\
                                                 , train_spec=self._train_spec\
                                                 , eval_spec=self._eval_spec)
        
        end_time = time.time()
        return results, start_time, end_time
    #-------------------------------------------------------------------------------


    #-------------------------------------------------------------------------------
    #   
    #-------------------------------------------------------------------------------
    def train_and_evaluate_deprecated(self):
        start_time = time.time()

        input_fn_param={'num_epochs':p8_util_config.NUM_EPOCHS,\
                'batch_size':p8_util_config.BATCH_SIZE,\
                'feature_shape': self._feature_shape,\
               }
               
        self._train_spec=tf.estimator.TrainSpec(
                input_fn=p8_util.input_fn("train", self._x_train, self._y_train, input_fn_param),\
                max_steps=p8_util_config.TRAIN_STEPS)

        self._eval_spec=tf.estimator.EvalSpec(
                input_fn=p8_util.input_fn("test", self._x_test, self._y_test, input_fn_param),\
                steps=None,\
                throttle_secs=1)    
        
        results, _ = tf.estimator.train_and_evaluate(self._estimator, train_spec=self._train_spec\
                                                     , eval_spec=self._eval_spec)
        
        end_time = time.time()
        return results, start_time, end_time
    #-------------------------------------------------------------------------------
        
    #-------------------------------------------------------------------------------
    #   
    #-------------------------------------------------------------------------------
    def _baseline_model_fn( self, features, labels, mode, params ): 
        '''This function implements training, evaluation and prediction.
        It also implements the predictor model.
        It is designed in the context of a customized Estimator.

        This function is invoked from Estimator's train, predict and evaluate methods.
        
        Input : 
            *   features : batch of features provided from input function.
            *   labels : batch labels provided from input function.
            *   mode : provided by input function, mode discriminate train, evaluation and prediction steps.
            *   params : parameters used in this function, passed to Estimator by higher level call.

        '''
        self._features = features
        #-----------------------------------------------------------------------------
        # Get from parameters object that is used from Adanet to build NN sub-networks.
        #-----------------------------------------------------------------------------
        net_builder = params['net_builder']
        nn_type = net_builder._nn_type

        feature_shape = net_builder.feature_shape
        logits_dimension = net_builder.nb_class

        input_layer = tf.feature_column.input_layer(features=features\
        , feature_columns=net_builder.feature_columns)
        is_training = False

        with tf.name_scope(nn_type):
            if self._is_debug is True :
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


            if self._is_debug is True :
                print("\n*** custom_model_fn() : logits shape= {} / labels shape= {}"\
                .format(logits.shape, labels.shape))


            #-----------------------------------------------------------------------
            # Loss is computed
            #-----------------------------------------------------------------------
            if p8_util_config.IS_CLASSIFIER is True: 
                #---------------------------------------------------------------
                # Classification issue
                #---------------------------------------------------------------
                if p8_util_config.IS_LABEL_ENCODED is True:
                    labels = tf.one_hot(labels, depth=self._n_class)
            else :
                #---------------------------------------------------------------
                # Regression issue; labels values are not processed
                #---------------------------------------------------------------
                pass
            with tf.name_scope('Loss'):
                if p8_util_config.IS_CLASSIFIER is True: 
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
                else :
                    loss , update_op= tf.metrics.mean_squared_error(labels,logits)
                tf.summary.scalar('Loss', loss)

            #-----------------------------------------------------------------------
            # Train processing or evaluation processing
            #-----------------------------------------------------------------------
            if mode == tf.estimator.ModeKeys.TRAIN :
                with tf.name_scope('Train'):
                    #---------------------------------------------------------------
                    # Gradient descent is computed 
                    #---------------------------------------------------------------
                    optimizer = net_builder.optimizer
                    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                    
                    if p8_util_config.IS_CLASSIFIER is True: 
                        #---------------------------------------------------------------
                        # Accuracy is computed
                        #---------------------------------------------------------------
                        tf_label_arg_max = tf.argmax(labels,1)
                        accuracy, accuracy_op = tf.metrics.accuracy(labels=tf_label_arg_max\
                                    , predictions=tf.argmax(logits,1)\
                                    , name=nn_type+'Train_accuracy')

                        tf.summary.scalar(nn_type+'Train_Accuracy', accuracy)
                        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
                    elif p8_util_config.IS_REGRESSOR :
                    else :
                        print("\n***ERROR : undefined model type; should be either classfier or regressor type!")
                        return None
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


