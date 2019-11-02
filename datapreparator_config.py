#!/usr/bin/python3.6
#-*- coding: utf-8 -*-
'''    This configuration allows to run an automated data-preparation process.
    
    This process is a sequence of 3 steps : 
    
        Step 1 : core data preparation using Spacy. 
        Core data preparation includes  :
            *   lematization, 
            *   stop words, 
            *   filtering out of vocubulary words with dictionary, 
            *   unknown words substitution, 
            *   entity recognition,
            *   cleaning, 
            *   tokenization.
            *   vectorization.
        At the end of this step, each text from dataset is digitalized as a vector.
        Input : 
            --> ./data/X_y_<data_type>.dill file
        Output :
            --> ./data/DataPreparator_v2_40Tokens_spacy_<data_type>_<sample>_step1.dill
            
            where sample is the number of texts sampled from train dataset.
            
        Step 2 : this a digitalization step. 
        A tensor is built from tokenized texts. 
        For each of the text : 
            * matrix rows are references over tokens composing a text;
            * matrix columns are embedded dimensions. 

        This step may be performed using hard disk in order ta save memory 
        resources. In this case, rows from dataframe are splitted as bulks and 
        each bulk is dumped into a file. 

        Input : 
            --> ./data/DataPreparator_v2_40Tokens_spacy_<data_type>_<sample>_step1.dill
            
        Output :
            --> ./data/DataPreparator_v2_40Tokens_spacy_<data_type>_<sample>_step2.dill
            
            where sample is th number of texts sampled from train dataset.
        
        Step 3 : a PCA is applied over the matrix built at step 2. 
        Dimensions of matrix are truncated with the selected dimensions issue 
        from variance rate.
        
        PCA eliminates redundant features. PCA also remove strong correlated features.
        It contribute to reduce over-fitting by removing dimensions capturing less variance.
        Such dimension my be due to noisy data.
        
            
        
        PCA also help to increase time-performance du to less sized dataset.
        
        Input : 
            --> ./data/DataPreparator_v2_40Tokens_spacy_<data_type>_<sample>_step2.dill
        Output :
            --> ./data/DataPreparator_v2_40Tokens_spacy_<data_type>_<sample>.dill
            
        
'''
import p9_util_spacy
import DataPreparator_v2

model_lang = p9_util_spacy.model_lang
#-------------------------------------------------------
# Parameters for DataPreparator_v2 :
# Cleaning, tokenization and digitalization of dataset 
# Digitalization leads to transform each text as a vector
# using word embeddings.
#-------------------------------------------------------
dict_param_dataprep={
    'other' : None, \
    'slice_length' :10000, \
    #---------------------------------------------------------------------------
    # Minimum number of letters in token
    #---------------------------------------------------------------------------
    'min_token_len':2, \
    
    #---------------------------------------------------------------------------
    # Maximum number of letters in token for not being removed.
    # If value is fixed to -1, then this limit is ignored.
    #---------------------------------------------------------------------------
    'max_token_len':-1,\
    
    #---------------------------------------------------------------------------
    # Mininimum number of words in text for not being removed.
    #---------------------------------------------------------------------------    
    'min_doc_len':1, \
    
    #---------------------------------------------------------------------------
    # Maximum number of words in text for not being removed.
    # If value is fixed to -1, then this limit is ignored.
    #---------------------------------------------------------------------------    
    'max_doc_len':-1,\
    
    'spacy_model_language' : model_lang,\
    'tokenizer' : None,\
    'max_padding_length': None,\
    'oov_keyword': None,\
    'entity_keyword' : None,\

    #---------------------------------------------------------------------------
    # Most frequent words are not filtered.
    #---------------------------------------------------------------------------
    'nb_word_most_frequent' : 0,\

    'is_df_copied' : False,\
    'is_tfidf' : False,\
    'threshold' : 0.0,\
}

#-------------------------------------------------------
# Parameters for each digitalization step
#-------------------------------------------------------
filename = None

#-------------------------------------------------------
# In this step : 
#  * read file containing dataset
#  * sample of dataset
#  * dataset digitalization in which each text is represented 
#    as a vector.
#-------------------------------------------------------
dict_step1={
    #-----------------------------------------------------------
    # This is the root name of file into data for train or 
    # validation to b prepared are stored.
    #-----------------------------------------------------------
    "dataset_filename":"./data/X_y_balanced",\
    
    #-----------------------------------------------------------
    # This is the set of parameters used to configure 
    # data-preparation for this step.
    #-----------------------------------------------------------
    "dict_param_dataprep" :dict_param_dataprep 
}

#-------------------------------------------------------
#  Dataset digitalization in which each text is represented 
#  as a 2D matrix (nb tokens X embedding dimension)
#  This step may contain sub-steps; this process save memory 
#  using hard-disk resources.
#-------------------------------------------------------
dict_step2 = {
    #-----------------------------------------------------------
    # This is the name of file into which DataPreparator_v2 object 
    # will be dumped at the end of this step.
    #-----------------------------------------------------------
    "dataprep_step_filename":filename,
    
    #-----------------------------------------------------------
    # This is the set of parameters used to configure 
    # data-preparation for this step.
    #-----------------------------------------------------------
    "dict_param_dataprep" :dict_param_dataprep,

    #---------------------------------------------------------------------------
    # When >0, this parameter is used to process data step by step 
    # spliting rows in a continuous enumerated set of bulks and dumped into
    # corresponding set of files.
    # 
    # This mean that the step after step 2 will be sub-step 1.  
    # This allows to save memory using hardisk to save bulks 
    # of data;
    # If this parameter is equal to 0, then data is processed in one step.
    #---------------------------------------------------------------------------
    "bulk_row" : 5000,
    
    #-----------------------------------------------------------
    # These parameters allows to configure the restart of this 
    # step at a range of dataframe rows under which process has 
    # already took place.
    # Process will restart at value of id_bulk_row.
    #-----------------------------------------------------------
    'dict_restart_step' : {
        'id_bulk_row' : 0,
    },
    
    #---------------------------------------------------------------------------
    # Sub-step 1 :
    # in this sub-step, data representation of texts as matrix are read from 
    # a set of files and concatened into a dataFrame.
    #
    # The following dictionary describes parameters of this sub-sequence.
    #---------------------------------------------------------------------------
    'dict_param_subsequence':{
        #-----------------------------------------------------------------------
        # Substep activation : start_substep >0
        # From where sequence of sub-steps start and end.
        # When start_substep value is 0, then sub-steps are skiped.
        #-----------------------------------------------------------------------    
        'start_substep' : 0,
        'end_substep' : 0,
        #-----------------------------------------------------------------------
        # Configuration for substep 1
        # In this substep, files are read and contents are 
        # aggregated into a dataframe.
        # Parameters in following dictionary describe how to process this step.
        #-----------------------------------------------------------------------
        'dict_param_step':{
            1 : {
                #---------------------------------------------------------------
                # While this value is >0, then this is the number of files to be 
                # read.
                # Files are read from first to fixed_count_file value.
                #
                # Otherwise, when value is 0, all files are read;
                #---------------------------------------------------------------
                'fixed_count_file' : 0,
            },
        },
    },
}

#-------------------------------------------------------------------------------
# Step 3 : dimension truncation using incremetal PCA
#-------------------------------------------------------------------------------
dict_step3 = {
    #-----------------------------------------------------------
    # Not used
    #-----------------------------------------------------------
    "dataprep_step_filename":filename,\
    
    #-----------------------------------------------------------
    # This is the set of parameters used to configure 
    # data-preparation for this step.
    #-----------------------------------------------------------
    "dict_param_dataprep" :dict_param_dataprep ,\

    #-----------------------------------------------------------
    # This value allows to fixe batch for incremental PCA.
    #-----------------------------------------------------------
    'ipca_batch_size' : 10000,\

    #-----------------------------------------------------------
    # This value is the percentage of variance to be expected 
    # when selecting a set of features issued from PCA transformation.
    #-----------------------------------------------------------
    'percent_var' : 0.9,\

    #--------------------------------------------------------------------    
    # For validation dataset, use PCA operator issued from train dataset 
    # Otherwise, for train dataset, PCA operator is built into thos step.
    #--------------------------------------------------------------------    
    'xpca' : None,\
    #---------------------------------------------------------------------------
    # Sub-step 1 :
    # in this sub-step, data representation of texts as matrix are read from 
    # a set of files and concatened into a dataFrame.
    #
    # The following dictionary describes parameters of this sub-sequence.
    #---------------------------------------------------------------------------
    'dict_param_subsequence':{
        #-----------------------------------------------------------------------
        # From where sequence of sub-steps start and end.
        # When start_substep value is 0, then sub-steps are skiped.
        #-----------------------------------------------------------------------    
        'start_substep' : 2 ,
        'end_substep' : 2 ,
        #-----------------------------------------------------------------------
        # Configuration for substep 1
        # In this substep, files are read and contents are 
        # aggregated into a dataframe.
        # Parameters in following dictionary describe how to process this step.
        #-----------------------------------------------------------------------
        'dict_param_step':{
            1 : {
                #---------------------------------------------------------------
                # Apply method from DataPreparator_v2 for building IPCA operator.
                #---------------------------------------------------------------
                'method' : DataPreparator_v2.DataPreparator_v2.build_ipca_operator,
            },
            2 : {
                #---------------------------------------------------------------
                # Apply method from DataPreparator_v2 for dimension reduction with 
                # IPCA operator.
                #---------------------------------------------------------------
                'method' : DataPreparator_v2.DataPreparator_v2.matrixpadded_truncate,
            
            },        
    },

}
}

#-------------------------------------------------------
# Parameters for digitalization steps sequences
# This is the description of the automated process.
#-------------------------------------------------------
dict_param_step={   
    1 : dict_step1,\
    2 : dict_step2,\
    3 : dict_step3,\
}

#-------------------------------------------------------
# Parameters for the global process, means , values 
# that remain same, whatever the step of the 
# digitalisation process.
#-------------------------------------------------------

data_type = 'train'
n_sample_train =  60000
n_sample_valid =  10000
if data_type == 'valid' :
    n_sample = n_sample_valid
else :
    n_sample = n_sample_train

dict_param_sequence={
    #---------------------------------------------------------------------------
    # The start and the end of a sequence
    #---------------------------------------------------------------------------
    'step' : 1,\
    'step_end' : 2,\

    #---------------------------------------------------------------------------
    # step=step_end=1, then previous_step_file_name is not taken into account
    #---------------------------------------------------------------------------
    "previous_step_file_name":"./data/DataPreparator_v2_40Tokens_spacy__en_core_web_lg__train_60000_step1.dill",\

    #---------------------------------------------------------------------------
    # This parameter contains informations related to the steps of the process
    #---------------------------------------------------------------------------
    'dict_param_step' : dict_param_step,\
    "data_type":data_type,\
    "root_file_name":"./data/DataPreparator_v2_MaxTokens_spacy__"+str(model_lang)+'_',\
    "n_sample_train":n_sample_train,\
    "n_sample_valid":n_sample_valid,\
    "file_format":".dill",\

}
