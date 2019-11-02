#!/usr/bin/python3.6
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR

import p5_util
import p3_util_plot
import p9_util
import p9_util_config

import DataGenerator
#import DataPreparator
import DataPreparator_v2
import test_datapreparator

#-------------------------------------------------------------------------------
# Test function enabling classification or regression.
#-------------------------------------------------------------------------------
def test_DataPreparator(X_train_, y_train_, X_valid_, y_valid_, \
                         is_keras_vectorizer=False, \
                         is_spacy_data_preparation=True,\
                         dataPreparator_train=None,\
                         dataPreparator_valid=None,
                         model_learning=LinearSVC) :
    '''This function allows to test DataPreparator quality against LinearSVC 
    for classification model or LinearSVR for regression model.
    DataPreparator transforms dataset using selectivaly 2 libraries : Keras 
    or Spacy.
    DataPreparator quality mat be tested against these 2 libraries.
    
    Input : 
        *   X_train_ : inputs for learning, array type.
        *   y_train_ : dataset targets for learning, array type.
        *   X_valid_ : dataset inputs for validation, array type.
        *   y_valid_ : dataset targets for validation, array type.
        *   is_keras_vectorizer : option for dataset preparation with Keras librairie.
        *   is_spacy_data_preparation : option for dataset preparation with Spacy librairie.
        *   dataPreparator_train : when None, then DataPreparator object is built 
            using X_train_, y_train_ as training dataset. Otherwise, training 
            dataset is extracted from it into X_train_, y_train_.
        *   dataPreparator_valid : when None, then DataPreparator object is built 
            using X_valid_, y_valid_ as validation dataset. Otherwise, validation  
            dataset is extracted from it into X_valid_, y_valid_.
        *   model_learning : default learning model. May be LinearSVC for 
            classification or LinearSVR for regression.
    Output :
        *   Dataframe matching with training dataset.   
    '''
    if dataPreparator_train is None :
        self = DataPreparator.DataPreparator(min_doc_len=2)

        self._is_spacy_data_preparation = is_spacy_data_preparation    
        self.is_keras_vectorizer = is_keras_vectorizer
        self.is_spacy_vectorizer = not self.is_keras_vectorizer

        #----------------------------------------------------------------------
        # Tokenization
        #----------------------------------------------------------------------
        X_,y_ = self.fit_transform_step(X_train_, y_train_)
        if False :
            print(X_)
            print("")
            print(y_)
            print("")
    else :
        self = dataPreparator_train
        X_,y_ = self.transform(None, None)
        is_spacy_data_preparation = self._is_spacy_data_preparation
        is_keras_vectorizer = self.is_keras_vectorizer
        is_spacy_vectorizer = not is_keras_vectorizer

    if self.is_keras_vectorizer :
        lib_name = "Keras"
    else :
        lib_name = "Spacy"




    #----------------------------------------------------------------------
    # Transformation of validation dataset
    #----------------------------------------------------------------------
    if dataPreparator_valid is None :    
        dataPreparator_valid = DataPreparator.DataPreparator(min_doc_len=2)
        dataPreparator_valid.is_keras_vectorizer = self.is_keras_vectorizer
        dataPreparator_valid.fit(X_valid_, y_valid_)
        X_valid_t_,y_valid_t_ = dataPreparator_valid.fit_transform_step(X_valid_,y_valid_)
    else : 
        X_valid_t_,y_valid_t_ = dataPreparator_valid.transform(None,None)

    #print(X_valid_t)
    #print("")
    if model_learning is None :
        return None
    else :
        pass
    #----------------------------------------------------------------------
    # Classification with a linear SVM model
    #----------------------------------------------------------------------
    if model_learning == LinearSVR :
        score_name = "R2 Score"
    else :
        score_name = "Accuracy"
        y_ = self.vectorValue2BinaryvectorLabel()
        y_valid_t_ = dataPreparator_valid.vectorValue2BinaryvectorLabel()
        if y_valid_t_ is None or y_ is None :
            return None
        else :
            pass
        
    model = model_learning()
    model.fit(X_,y_)

    y_pred = model.predict(X_valid_t_)
    #print(y_valid_t)
    #print("")
    #print(y_pred)

    print(" ")
    score = model.score(X_valid_t_, y_valid_t_)
    print("Vectorization : "+lib_name+" / Scapy data preparation: {}".format(is_spacy_data_preparation))
    print(score_name+"= {}".format(round(score,3)))
    

    #from sklearn.metrics import accuracy_score 
    #score = round(accuracy_score(y_valid_t, y_pred),2)
    #print ("Keras = {} / Accuracy: {}".format(self.is_keras_vectorizer,score))
    print(" ")
    return self.df_data
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Following function is deprecated.
#-------------------------------------------------------------------------------
def build_generators(train_datapreparator, \
                     valid_datapreparator, \
                     partition_size,\
                     dict_param,\
                     data_column_name='vector' ) :
    
    #---------------------------------------------------------------------------
    # Retrieve data from DataPreparator dataframes.
    #---------------------------------------------------------------------------
    if data_column_name in train_datapreparator.df_data.columns :
        X_train = np.array(train_datapreparator.df_data[data_column_name].tolist())
    else :
        print("\n***ERROR : column name \'{}\' out of train dataframe !".\
        format(data_column_name))
        return None, None
    
    if data_column_name in valid_datapreparator.df_data.columns :
        X_test  = np.array(valid_datapreparator.df_data[data_column_name].tolist())
    else :
        print("\n***ERROR : column name {} out of validation dataframe !".\
        format(data_column_name))
        return None, None

    y_train = np.array(train_datapreparator.df_data.target.tolist())
    y_test  = np.array(valid_datapreparator.df_data.target.tolist())
    
    print(X_train.shape)
    
    #---------------------------------------------------------------------------
    # Update dictionary of parameters
    #---------------------------------------------------------------------------
    dict_param['dim'] = (X_train.shape[1], dict_param['dim'][1])

    #---------------------------------------------------------------------------
    # Make partitions
    #---------------------------------------------------------------------------
    dict_train_partition, dict_train_label = p9_util.make_partition(X_train, \
                                                y_train, partition_size,\
                                                data_type="train", \
                                                data_format='ndarray' )

    if (dict_train_partition is None) or (dict_train_label is None) :
        print("\n*** ERROR : build_generators() : building partitions for train dataset failed!")
        return None, None

    dict_test_partition, dict_test_label = p9_util.make_partition(X_test, \
                                                  y_test,partition_size,\
                                                  data_type="test", \
                                                  data_format='ndarray' )
    if (dict_test_partition is None) or (dict_test_label is None) :
        print("\n*** ERROR : build_generators() : building partitions for valid dataset failed!")
        return None, None

    #---------------------------------------------------------------------------
    # Build data generators
    #---------------------------------------------------------------------------
    len_train = X_train.shape[0]
    train_generator = DataGenerator.DataGenerator(dict_train_partition, \
                                                  dict_train_label, \
                                                  partition_size, \
                                                  len_train,**dict_param)
    len_test = X_test.shape[0]
    test_generator = DataGenerator.DataGenerator(dict_test_partition, \
                                                 dict_test_label, \
                                                 partition_size,\
                                                 len_test, **dict_param)
    return train_generator, test_generator
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_generator(dataPreparator, \
                     dict_param_generator,\
                     data_column_name='vector') :
                     
    '''Builds DataGenerator object from DataPreparator object.
    DataGenerator object allows to feed Keras neural network estimators 
    by pumping data recorded in a set of files named partitions.

    This allows to save RAM memory thanks to the use of hard disk.
    
    DataGenerator object contains all informations required in order to 
    access data into partitions files.
    
    
    Input : 
        * dataPreparator : object containing digitalized dataset along with 
        operators used for digitalization.
        * dict_param_generator : dictionary of parameters used to construct 
        DataGenerator.
        * data_column_name : name of the column from dataframe into 
        dataPreparator containing digitalized dataset.
    Output :
        * DataGenerator object
    '''
    
    data_type = dict_param_generator['data_type']

    #---------------------------------------------------------------------------
    # Consistency of inputs are checked
    #---------------------------------------------------------------------------
    if data_type is None :
        print("\n*** ERROR : build_generator() : unknown data type!")
        return None
    
    if (data_type != "train") and (data_type != "valid"):
        print("\n*** ERROR : build_generator() : Unknown data_type= {}! Supported data_type : train or valid".format(data_type))
        return None
        
    if (dataPreparator.list_df_data_file is None) or (0 == len(dataPreparator.list_df_data_file)) :

        #-----------------------------------------------------------------------
        # All data is stored into dataPreparator dataframe.
        # Retrieve data from DataPreparator dataframes.
        #-----------------------------------------------------------------------
        if data_column_name in dataPreparator.df_data.columns :
            X = np.array(dataPreparator.df_data[data_column_name].tolist()) 
            y = np.array(dataPreparator.df_data.target.tolist())
        else :
            print("\n***ERROR : column name \'{}\' out of train dataframe !".\
            format(data_column_name))
            return None, None
        
        #-----------------------------------------------------------------------
        # Make partitions
        #-----------------------------------------------------------------------
        print(y[:10])
        partition_size = dict_param_generator['partition_size']
        dict_partition, dict_label = p9_util.make_partition(X, \
                                                y, \
                                                partition_size,\
                                                data_type=data_type, \
                                                data_format='ndarray' )

        if (dict_partition is None) or (dict_label is None) :
            print("\n*** ERROR : build_generators() : building partitions for data_type= {} dataset failed!".format(data_type))
            return None

        #---------------------------------------------------------------------------
        # Total number of records
        #---------------------------------------------------------------------------
        len_dataset = X.shape[0]

    else : 
        #-----------------------------------------------------------------------
        # All data are stored into files on harddisk.
        # dataPreparator handle the name of those files.
        # Files are read and partitions are built for each of thses files.
        #-----------------------------------------------------------------------
        dict_partition = dict()
        dict_label = dict()
        partition_size = dict_param_generator['partition_size']
        print("\n*** Partition size = {}".format(partition_size))
        len_dataset = 0
        start_row = 0
        for df_data_file in dataPreparator.list_df_data_file :
            df_data = p5_util.object_load(df_data_file, is_verbose=True)
            if data_column_name in df_data.columns :
                X = np.array(df_data[data_column_name].tolist()) 
                end_row = start_row + X.shape[0]
                y = np.array(df_data.target.tolist())
                #y = np.array(df_data.target.tolist())[start_row:end_row]
                #y = np.array(dataPreparator.df_data.target.tolist())[start_row:end_row]

                start_row = end_row
                len_dataset += X.shape[0]
            else :
                print("\n***ERROR : file name= {} : column name \'{}\' out of train dataframe !".\
                format(df_data_file,data_column_name))
                return None, None

            #-------------------------------------------------------------------
            # Make partitions; dict_partition and dict_label are updated in each
            # function call for making partitions
            #-------------------------------------------------------------------
            dict_partition, dict_label = p9_util.make_partition(X, \
                                                    y, \
                                                    partition_size,\
                                                    data_type=data_type, \
                                                    data_format='ndarray',\
                                                    dict_partition = dict_partition,\
                                                    dict_label = dict_label,\
                                                    is_debug=False)
            
    #---------------------------------------------------------------------------
    # Build data generators
    #---------------------------------------------------------------------------
    dataGenerator = DataGenerator.DataGenerator(dict_partition, \
                                                dict_label, \
                                                len_dataset,\
                                                **dict_param_generator)
        
    #---------------------------------------------------------------------------
    # Save DataGenerator
    #---------------------------------------------------------------------------
    filename = "./data/"+str(data_type)+"_generator.dill"
    p5_util.object_dump(dataGenerator, filename)
    return dataGenerator
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def dataprepatator_subsequence_process(dict_param_sequence, dataPreparator_v2) :
    ''' This functions allows to concatenate dataframes stored into multiple files.
    This is due to the fact data has been process by bulk in order to save memory 
    resources.
    All files are read and concatenated into a single dataframe.
    
    Input :
        * dict_param_sequence : parameters of the data-prepatation sequence which 
        also contain parameters to process sub-steps in a step.
        The current step in which this sub-step has to be processed is contained 
        in these parameters.
        
        * dataPreparator_v2 : DataPreparator_v2 object in which df_data 
        attributes will be updated.

        
    
    Output :
        * dataframe containing all data splitted into files.
        
    '''
    df_data = None

    #----------------------------------------------------------------------
    # Extract parameters for sub-sequences related to this step.
    #----------------------------------------------------------------------
    step = dict_param_sequence['step']
    dict_step = dict_param_sequence['dict_param_step'][step]
    dict_param_subsequence = dict_step['dict_param_subsequence']
    n_sample = get_n_sample(dict_param_sequence)

    #---------------------------------------------------------------------------
    # Extract start and send of the sub-steps.
    # They are in dict_param_subsequence dictionary.
    #---------------------------------------------------------------------------
    start_substep = dict_param_subsequence['start_substep']
    end_substep   = dict_param_subsequence['end_substep']

    if 2 == step :
        bulk_row = dict_step['bulk_row']
        #---------------------------------------------------------------------------
        # Extract parameters that configure this sequence of sub-steps
        #---------------------------------------------------------------------------
        dict_param_step = dict_param_subsequence['dict_param_step']
        
        for substep in range(start_substep,end_substep+1) :
            if 1 == substep :
                total_row = dataPreparator_v2.total_row     
                #-------------------------------------------------------------------
                # In this sub-step, files are read form hardisk and aggregated 
                # as a dataframe.
                # 
                # Extract parameters that configure this sub-step.
                #-------------------------------------------------------------------
                fixed_count_file = dict_param_step[substep]['fixed_count_file']

                if 0 == total_row :
                    return None
                else :
                    pass

                if fixed_count_file > 0 :
                    count_file = fixed_count_file
                    tail = 0
                else :        
                    count_file = total_row//bulk_row
                    tail = total_row%bulk_row

                print("Step {} : sub-step: {} : dataframe concatenation of {} files".format(step, substep,count_file))
                root_filename = "./data/df_"+str(dict_param_sequence['data_type'])+"_"+str(n_sample)+"_step"+str(step)
                df_data = pd.DataFrame()
                is_intermediate = False
                for i in range(count_file) :
                    is_intermediate = True
                    filename = root_filename+"_"+str(i)+".dill"
                    df = p5_util.object_load(filename, is_verbose=False)
                    df_data = pd.concat([df_data,df])
                    print("Step {} : sub-step: {} : process status : {}/{}".\
                    format(step, substep, i+1,count_file), end='\r')

                if tail >0 :
                    if is_intermediate :
                        i+=1
                    else :
                        i=0
                    filename = root_filename+"_"+str(i)+".dill"
                    df_data = pd.concat([df_data,df])
            else :
                print("\n*** ERROR : Step : {} / sub-step={} not yet supported".format(step, substep))
                df_data = None        
            #-----------------------------------------------------------------------
            # Drop unused columns from df_data in order to save memory
            #-----------------------------------------------------------------------
            if 'vector' in df_data.columns : 
                del(df_data['vector'])
            
            if 'tokens' in df_data.columns : 
                del(df_data['tokens'])
                
            if 'counting' in df_data.columns : 
                del(df_data['counting'])
        
            #-----------------------------------------------------------------------
            # Update df_data attribute with concatenated dataframe.
            #-----------------------------------------------------------------------
            dataPreparator_v2.df_data = df_data.copy()

    elif 3 == step:	
        for substep in range(start_substep,end_substep+1) :
            method = dict_param_subsequence['dict_param_step'][substep]['method']
            if 1 == substep :
                parameter = dict_step['ipca_batch_size']
            elif 2 == substep :
                parameter = dict_step['percent_var']
            else  :
                print("\n*** ERROR : dataprepatator_subsequence_process() : sub-step= {} not yet supported!".format(substep))
                return None
            method(dataPreparator_v2, parameter)
    else :
        print("\n*** ERROR : no sub-steps supported for step = {}".format(step))
        return None
    
    
        
    return dataPreparator_v2
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def datapreparation_step_process(dict_param_sequence=None, is_debug=False ) :
    
    if dict_param_sequence is None :
        print("\n*** ERROR : parameter not defined for function!")
        return None

    
    
    #--------------------------------------------------------------------------
    # Build backup / restore file name
    #--------------------------------------------------------------------------
    data_type = dict_param_sequence["data_type"]
    root_file_name = dict_param_sequence["root_file_name"]

    n_sample = get_n_sample(dict_param_sequence)
    if n_sample is None :
        return None
    file_format =  dict_param_sequence["file_format"]
    
    
    #---------------------------------------------------------------------------
    # Get step number then step parameters and process it.
    #---------------------------------------------------------------------------
    dict_param_step = dict_param_sequence['dict_param_step']
    step = dict_param_sequence['step']
    dict_step = dict_param_step[step]
    
    if dict_step is None :
        print("\n*** INFO : step= {} of automated process skiped!".format(step))
        return None
        
    dict_param_dataprep = dict_step['dict_param_dataprep']
    
    

    #---------------------------------------------------------------------------
    # Steps processing
    #---------------------------------------------------------------------------
    if 1 == step :
        root_dataset_filename = dict_step["dataset_filename"]
        if None is root_dataset_filename :
            print("\n*** ERROR : step= {}: dataset filename for this step not provided!".format(step))
            return None

        #---------------------------------------------------------------------------
        # Load dataset
        #---------------------------------------------------------------------------
        dataset_filename = root_dataset_filename+'_'+str(data_type)+str(file_format)
        if not is_debug :
            ser_X_data_type, ser_y_data_type = p5_util.object_load(dataset_filename)    
            
            X_data_type = ser_X_data_type.sample(n_sample)
            index = X_data_type.index
            y_data_type = ser_y_data_type[index]
            
            X_data_type = X_data_type.tolist()
            y_data_type = y_data_type.tolist()
                        
            #-------------------------------------------------------------------
            # Apply operations over this step using fit_transform() method.
            #-------------------------------------------------------------------
            dataPreparator_v2 = DataPreparator_v2.DataPreparator_v2(**dict_param_dataprep)
            X = dataPreparator_v2.fit_transform_step(X_data_type,y_data_type)

        filename = root_file_name+'_'+str(data_type)+'_'+str(n_sample)+'_step'+str(step)+str(file_format)            
        

        #---------------------------------------------------------------------------
        # Save step result
        #---------------------------------------------------------------------------
        if not is_debug :
            p5_util.object_dump(dataPreparator_v2,filename)
        print("\nStep : {} Save data-preparation into file {}".format(step,filename))
        return filename
    elif 2 == step :
        print("\nStep : {}".format(step))
        
        #-----------------------------------------------------------------------
        # Not used. Already at None value.
        #-----------------------------------------------------------------------
        filename = dict_step['dataprep_step_filename']
        if None is filename :
            #-------------------------------------------------------------------
            # Select file issued from previous step.
            #-------------------------------------------------------------------
            filename = dict_param_sequence["previous_step_file_name"]
            if None is filename :        
                print("\n*** ERROR : step= {}: filename for this step not provided!".format(step))
                return None
        
        #---------------------------------------------------------------------------
        # Load DataPreparator_v2 object from step1
        #---------------------------------------------------------------------------
        if not is_debug :
        
            dataPreparator_v2 = p5_util.object_load(filename, is_verbose=True)
            
            #-------------------------------------------------------------------
            # Check is this step as already took place and then, 
            # apply a sequence of sub-steps.
            #-------------------------------------------------------------------
            dict_param_subsequence = dict_step['dict_param_subsequence']
            start_substep = dict_param_subsequence['start_substep']

            #-----------------------------------------------------------------------
            # Build file name in order to save DataPreparator_v2 oject
            #-----------------------------------------------------------------------
            filename = root_file_name+'_'+str(data_type)+'_'+str(n_sample)+'_step'+str(step)+str(file_format)

            if 0 < start_substep :
                #---------------------------------------------------------------
                # Step 2 already took place. Apply sub-step from step 2.
                # Sub-step sequence is described into dict_param_sequence 
                # dictionary.
                #---------------------------------------------------------------
                dataPreparator_v2 = \
                dataprepatator_subsequence_process(dict_param_sequence,\
                dataPreparator_v2)

                if not is_debug :
                    print("\nStep {} : Save data-preparation into file {} ...".format(step,filename))
                    p5_util.object_dump(dataPreparator_v2,filename)
                else : 
                    pass                
                
            else :
                #-------------------------------------------------------------------
                # If step2 transformation didn't took place or was not completed, 
                # then it is processed here.
                #
                # When bulk_row is >0, transformation is proceeded step 
                # by step, saving results from each step in a file.
                #
                # This step by step process may be interrupted. Then it may be 
                # restarted from the value= id_bulk_row.
                #
                # Otherwise, when id_bulk_row = 0, then transformation starts 
                # from beguining.
                #-------------------------------------------------------------------
                bulk_row = dict_step['bulk_row']
                dict_restart_step = dict_step['dict_restart_step']
                id_bulk_row = dict_restart_step['id_bulk_row']
                
                dataPreparator_v2.build_padded_matrix(bulk_row = bulk_row,\
                root_filename='./data/df_'+str(data_type)+'_'+str(n_sample)+'_step'+str(step),\
                id_bulk_row = id_bulk_row)
        

                if 0 == bulk_row :
                    pass
                else : 
                    #-----------------------------------------------------------
                    # Data are saved into files.
                    # Clean dataframe in case of step by step process.
                    #-----------------------------------------------------------
                    dataPreparator_v2.total_row = dataPreparator_v2.df_data.shape[0]
                    dataPreparator_v2.df_data = pd.DataFrame()                    

                if not is_debug :
                    print("\nStep {} : Save data-preparation into file {} ...".format(step,filename))
                    p5_util.object_dump(dataPreparator_v2,filename)
                    print("\nStep {} : Save data-preparation into file {} Done!".format(step,filename))
                
                if 0 == bulk_row :
                    pass
                else : 
                    print("\nStep {} : Step by step data-preparation : Restart process with (step, substep, previous file) = (2,1,{}) in configuration file!".format(step,filename))
        return filename
    elif 3 == step :
        print("\nStep : {}".format(step))
        ipca_batch_size = dict_step['ipca_batch_size']
        percent_var = dict_step['percent_var']

        if (percent_var > 1.) or (percent_var <= 0.) :
            print("\n*** ERROR : step= {}: percent_var has to belong interval ]0.,1.]; current value= {}".\
                  format(step, percent_var))
            return None

        filename = dict_step['dataprep_step_filename']
        if None is filename :
            filename = dict_param_sequence["previous_step_file_name"]
            if None is filename :        
                print("\n*** ERROR : step= {}: filename for this step not provided!".format(step))
                return None
        if not is_debug :
            #-------------------------------------------------------------------
            # Load dataPreparator_v2 object from step2
            #-------------------------------------------------------------------
            dataPreparator_v2 = p5_util.object_load(filename)

            #-------------------------------------------------------------------
            # PCA operator for dimension reduction.
            #-------------------------------------------------------------------
            xpca = dict_step['xpca']
            if xpca is not None :
                dataPreparator_v2.xpca = xpca
            else :
                pass    

        #-------------------------------------------------------------------
        # Check is this step as already took place and then, 
        # apply a sequence of sub-steps.
        #-------------------------------------------------------------------
        dict_param_subsequence = dict_step['dict_param_subsequence']
        start_substep = dict_param_subsequence['start_substep']
        end_substep   = dict_param_subsequence['end_substep']
        
        if 0 < start_substep :
            #---------------------------------------------------------------
            # Use 2 separated steps in order to build PCA operator then 
            # to proceed to dimension reduction.
            #---------------------------------------------------------------
            if not is_debug :
                dataPreparator_v2 = \
                dataprepatator_subsequence_process(dict_param_sequence,\
                dataPreparator_v2)
            else :
                pass
        else : 
            #---------------------------------------------------------------
            # In the same step, build PCA operator and proceed to dimension 
            # reduction.
            #---------------------------------------------------------------
            if not is_debug :
                dataPreparator_v2.build_matrix_padded_truncated(ipca_batch_size, \
                percent_var)
            else :
                pass
        if len(dict_param_step) == step :
            if 0 < start_substep :
                if len(dict_param_subsequence['dict_param_step']) == end_substep :
                    #-----------------------------------------------------------
                    # Lat step from sequence and last sub-step from sub-sequence
                    # File name will not have any step value extension .
                    #-----------------------------------------------------------
                    filename = root_file_name+'_'+str(data_type)+'_'+str(n_sample)+str(file_format)
                else : 
                    #-----------------------------------------------------------
                    # Lat step from sequence and thists is not la sub-step 
                    # from sub-sequence.
                    # Filename will have a step value extension .
                    #-----------------------------------------------------------
                    filename = root_file_name+'_'+str(data_type)+'_'\
                    +str(n_sample)+'_step'+str(step)+'_substep'+\
                    str(start_substep)+str(file_format)
            else : 
                #---------------------------------------------------------------
                # Last step and no sub-step: file name is the final one.    
                # File name will not have any step value extension .
                #---------------------------------------------------------------
                filename = root_file_name+'_'+str(data_type)+'_'+str(n_sample)+str(file_format)
                
        else :            
            filename = root_file_name+'_'+str(data_type)+'_'+str(n_sample)+'_step'+str(step)+str(file_format)
        if not is_debug :
            print("\nStep {} : Save data-preparation into file {} ...".format(step,filename))
            p5_util.object_dump(dataPreparator_v2,filename)
        print("\nStep {} : Save data-preparation into file {} Done!".format(step,filename))
        return filename
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def dataprepatator_sequence_process(dict_param_sequence, is_debug=False) :
    #--------------------------------------------------------------
    # Start and end steps in the process
    #--------------------------------------------------------------
    step = dict_param_sequence['step']
    step_end   = dict_param_sequence['step_end']

    for step in range(step, step_end+1) :
        #-----------------------------------------------------
        # Update current step value
        #-----------------------------------------------------
        dict_param_sequence['step'] = step

        #-----------------------------------------------------
        # process to this step and returns file issued from 
        # saved process.
        #-----------------------------------------------------
        filename = test_datapreparator.datapreparation_step_process(dict_param_sequence=dict_param_sequence, is_debug=is_debug)
        if step < step_end :
            #-----------------------------------------------------
            # Update file name process into next step parameters
            #-------------------------------------------------
            dict_step_next = dict_param_sequence['dict_param_step'][step+1]
            if dict_step_next is not None :
                dict_step_next['dataprep_step_filename'] = filename

                #-------------------------------------------------
                # Update next step into global process parameters.
                #-------------------------------------------------
                dict_param_sequence[step+1] = dict_step_next
            else :
                print("\n*** ERROR: process step= {} not defined!".format(step+1))
    return filename
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_cnn_datagenerator(filename_datapreparator=None, \
                        percent_var=1.0, dataPreparator_v2 = None,
                        dict_param_generator = None):
    '''This function allows to build datagenerator object for a Keras CNN 
    model.
    
    Such process includes : 
        * The load of DataPreparator_v2 object
        * The build of DataGenerator object used from Keras as a source of data.

    The configuration file, p9_util_config.py is used to read configuration and  
    hyper-parameters.
    
    Inputs :
        * filename_datapreparator : file name of dumped DataPreparator object.
        Such object is issued from data-preparation process.
        
        * percent_var : precentage of variance expected. This is used in order to build 
        CNN architecture with multiple channels. The number of channel depends of the 
        number of PCA dimensions fixed for dataset.
        
        * dataPreparator_v2 : Data-Preparator_v2 object used for data preparation.
        
        * dict_param_generator : parameter for CNN data generator. When None, 
        then module 'p9_util_config.dict_param_generator' is used.
    Output :
        * DataGenerator object type.
    
    '''
    if dataPreparator_v2 is None :
        if filename_datapreparator is None :
            print("\n*** ERROR : DataPreparator file name undefined!")
            return None
        #-----------------------------------------------------------------------
        # Load of DataPreparator object
        #-----------------------------------------------------------------------
        dataPreparator_v2 = p5_util.object_load(filename_datapreparator)
    else :
        print("Using DataPreparator given as parameter...")
        pass
    
    print("\nDataPreparator Dataframe shape= {}".format(dataPreparator_v2.df_data.shape))
    

    #---------------------------------------------------------------------------
    # Fixe DataGenerator parameters
    #---------------------------------------------------------------------------
    if dict_param_generator is None :
        dict_param_generator = p9_util_config.dict_param_generator
    else : 
        pass
    
    #---------------------------------------------------------------------------
    # Update aprameters depending from DataPreparator_v2
    #---------------------------------------------------------------------------
    dict_param_generator['binary_threshold'] = dataPreparator_v2.threshold

    #---------------------------------------------------------------------------
    # Extract PCA operator in order to extract the number of components 
    # given a percentage of variance rate.
    #---------------------------------------------------------------------------
    pca = dataPreparator_v2.xpca
    if pca is not None :
        nb_component = p3_util_plot.get_component_from_cum_variance(pca, percent_var) 
        print("\nComponents= {} for variance= {}%".format(nb_component, percent_var*100))
        dict_param_generator['keras_nb_channel'] = nb_component
    else :
        dict_param_generator['keras_nb_channel'] =0
    
    #---------------------------------------------------------------------------
    # Input dim for CNN network is defined by the tuple :
    # (number_of_measures, number_of_features).
    # 
    # for text classification : (number_of_tokens, embedding_dim)
    #
    # In case of multiplexed dimensions, for a given dimension, then, each token 
    # is assigned a coefficient that is the value of this digitalized token over 
    # this given dimension.
    # Then, for each text, they are dataPreparator_v2.max_length tokens.
    # Input dim is then (dataPreparator_v2.max_length, 1).
    #
    # When embedding dimensions are not multiplexed :
    #  If dimension are truncated using PCA operator, then input dimensions
    #  are : (dataPreparator_v2.max_length, self.df_data['matrix_padded_truncated'].shape[1])
    #
    #  If embedding dimension are not truncated, then input dimensions are :
    #  are : (dataPreparator_v2.max_length, self.df_data['matrix_padded'].shape[1])
    # 
    #---------------------------------------------------------------------------
    if dict_param_generator['is_dimension_mux'] :
        nb_feature = 1
        data_column_name = 'matrix_padded_truncated'
    else : 
        if 'matrix_padded_truncated' in dataPreparator_v2.df_data.columns :
            nb_feature = dataPreparator_v2.df_data['matrix_padded_truncated'].iloc[0].shape[1]
            data_column_name = 'matrix_padded_truncated'
        elif 'matrix_padded' in dataPreparator_v2.df_data.columns :
            nb_feature = dataPreparator_v2.df_data['matrix_padded'].iloc[0].shape[1]
            data_column_name = 'matrix_padded'
        else :
            if 0 < len(dataPreparator_v2.list_df_data_file) :
                #---------------------------------------------------------------
                # Data has been recorded on harddidk. Search if 
                # column name exists into dataframe.
                #---------------------------------------------------------------
                filename = dataPreparator_v2.list_df_data_file[-1]
                df_data = p5_util.object_load(filename)
                if 'matrix_padded_truncated' in df_data.columns :                
                    nb_feature = df_data['matrix_padded_truncated'].iloc[0].shape[1]
                    data_column_name = 'matrix_padded_truncated'
                elif 'matrix_padded' in df_data.columns :
                    nb_feature = df_data['matrix_padded'].iloc[0].shape[1]
                    data_column_name = 'matrix_padded'
                else : 
                    print("\n*** ERROR : build_cnn_datagenerator(): No column name \'matrix_padded_truncated\' nor \'matrix_padded\' CNN dimension is undefined! ")
                    return None

            else : 
                print("\n*** ERROR : build_cnn_datagenerator(): No column name \'matrix_padded_truncated\' nor \'matrix_padded\' CNN dimension is undefined! ")
                return None
    #print(dataPreparator_v2.max_length, nb_feature)
    keras_input_dim = (dataPreparator_v2.max_length, nb_feature)
    dict_param_generator['keras_input_dim'] = keras_input_dim
    
    print("")
    for key, value in dict_param_generator.items() :
        print("{} : {}".format(key,value))
        
    print("\nBuilding datagenerator...")
    generator = test_datapreparator.build_generator(dataPreparator_v2, \
                     dict_param_generator,\
                     data_column_name=data_column_name)
    
       
    return generator
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_n_sample(dict_param_sequence) :
    data_type = dict_param_sequence['data_type']
    #---------------------------------------------------------------------------
    # Compute n_sample parameter from data_type
    #---------------------------------------------------------------------------
    if 'valid' == data_type :
        n_sample = dict_param_sequence["n_sample_valid"]
    elif 'train' == data_type :
        n_sample = dict_param_sequence["n_sample_train"]
    else :
        print("\n*** ERROR : unsupported data_type= {} / Expected values : valid or train".format(data_type))
        return None

    return n_sample
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def datapreparator_filename(data_type, dict_param_sequence, step=None) :
    dict_param_sequence['data_type'] = data_type
    n_sample = get_n_sample(dict_param_sequence)
    if n_sample is None :
        return None

    root_file_name = dict_param_sequence['root_file_name']
    file_format= dict_param_sequence['file_format']
    if step is None :
        filename = root_file_name+'_'+str(data_type)+'_'+str(n_sample)+str(file_format)
    else : 
        filename = root_file_name+'_'+str(data_type)+'_'+str(n_sample)+'_step'+str(step)+str(file_format)
        
    return filename
#-------------------------------------------------------------------------------

