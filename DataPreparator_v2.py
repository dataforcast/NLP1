#!/usr/bin/python3.6
#-*- coding: utf-8 -*-
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA

from sklearn.base import TransformerMixin 

import spacy
import pandas as pd
import keras

import p3_util_plot
import p5_util
import p9_util
import p9_util_spacy
import KerasTokenizer

import DataPreparator
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class DataPreparator_v2(TransformerMixin):
    '''This class allows to :
        * clean data
        * tokenize data
        * vectorize data
    '''
    COLUMN_NAME_DOC    = p9_util_spacy.COLUMN_NAME_DOC
    COLUMN_NAME_TARGET = p9_util_spacy.COLUMN_NAME_TARGET
    COLUMN_NAME_TOKEN  = p9_util_spacy.COLUMN_NAME_TOKEN
    COLUMN_NAME_COUNT  = p9_util_spacy.COLUMN_NAME_COUNT
    COLUMN_NAME_VECTOR = p9_util_spacy.COLUMN_NAME_VECTOR

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------    
    def __init__(self,\
                 other = None,
                 slice_length=10000, \
                 min_token_len=2, max_token_len=15,\
                 min_doc_len=5, \
                 max_doc_len=500,
                 spacy_model_language = 'en_core_web_md',
                 tokenizer=None,
                 max_padding_length = None,
                 oov_keyword="unknown",
                 entity_keyword=None,
                 nb_word_most_frequent=100,
                 is_df_copied=False,
                 is_tfidf=False,
                 threshold=None):
        #-----------------------------------------------------------------------
        # Specific parameters to this object
        #-----------------------------------------------------------------------
        self.is_tfidf = is_tfidf
        self.xpca = None
        self.is_df_copied = is_df_copied

        self.df_data = pd.DataFrame()
        self._corpus = None
        self._target = None
        self.slice_length = slice_length
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.min_doc_len   = min_doc_len
        self.max_doc_len   = max_doc_len
        self.max_length = max_padding_length
        self._is_keras_vectorizer = False
        self._is_spacy_vectorizer = True
        self._is_matrix_2D = True
        self._is_spacy_data_preparation = True
        self._oov_keyword = 'unknown'
        self._entity_keyword = None
        self.csr_tfidf = None
        self.dict_token_tfidf = dict()
        self.llist_error_doc = list()
        
        # Most frequent words that are not stop-words will not be Lemmatized
        self.nb_word_most_frequent = nb_word_most_frequent
        self.list_word_most_frequent = list()
        

        if tokenizer is None :
            self.kerasTokenizer = KerasTokenizer.KerasTokenizer()
        else :
            self.kerasTokenizer = tokenizer

        self.total_row = 0
        self.list_df_data_file = list()
        self.dict_param_sequence = dict() 
        self.threshold = threshold
            #self.spacy_nlp     = spacy.load(spacy_model_language)
        if other is not None :             
            if not isinstance(other, DataPreparator.DataPreparator) and (not isinstance(other, DataPreparator_v2)) :
                print("\n*** ERROR : other is not instance of DataPreparator neither DataPreparator_v2!")
            else :
                if isinstance(other, DataPreparator_v2) :
                    #----------------------------------------------------------- 
                    # Copy specific attributes from DataPreparator_v2 object
                    #----------------------------------------------------------- 
                    self.is_tfidf = other.is_tfidf
                    self.xpca = other.xpca
                else :
                    pass
                #----------------------------------------------------------- 
                # Copy attributes from DataPreparator object 
                #----------------------------------------------------------- 
                if self.is_df_copied :
                    self.df_data = other.df_data.copy()
                else :
                    self.df_data = pd.DataFrame()
                self._corpus = None
                self._target = None
                self.csr_tfidf = None
                self.llist_error_doc = other.llist_error_doc.copy()
                
                #---------------------------------------------------------------
                # Parameters issued from other object.
                #---------------------------------------------------------------
                self.slice_length  = other.slice_length
                self.min_token_len = other.min_token_len
                self.max_token_len = other.max_token_len
                self.min_doc_len   = other.min_doc_len
                self.max_doc_len   = other.max_doc_len
                self.max_length    = other.max_length
                self._is_keras_vectorizer = other._is_keras_vectorizer
                self._is_spacy_vectorizer = other._is_spacy_vectorizer
                self._is_matrix_2D = other._is_matrix_2D
                self._is_spacy_data_preparation = other._is_spacy_data_preparation
                self._oov_keyword =other._oov_keyword
                self._entity_keyword = other._entity_keyword
                self.dict_token_tfidf = other.dict_token_tfidf.copy()
                self.nb_word_most_frequent = other.nb_word_most_frequent
                self.list_word_most_frequent = other.list_word_most_frequent.copy()
                self.kerasTokenizer = other.kerasTokenizer
                                
                self.threshold = other.threshold


                #---------------------------------------------------------------
                # Drop some columns from other object
                #---------------------------------------------------------------
                if self.COLUMN_NAME_DOC in self.df_data.columns :
                    del(self.df_data[self.COLUMN_NAME_DOC])
            
                if self.COLUMN_NAME_VECTOR in self.df_data.columns :
                    del(self.df_data[self.COLUMN_NAME_VECTOR])
                    
                self.max_length = other.max_length
                if True :
                    self.dict_param_sequence = other.dict_param_sequence.copy()
                self.total_row = other.total_row
                self.list_df_data_file = other.list_df_data_file.copy()
                
    #---------------------------------------------------------------------------
    def get_params(self) :
        return {'slice_length' : self.slice_length, \
        'min_token_len' : self.min_token_len, \
        'max_token_len' : self.max_token_len,\
        'min_doc_len' : self.min_doc_len,\
        'max_doc_len' : self.max_doc_len,\
        'max_length' : self.max_length,\
        
        '_is_keras_vectorizer' : self._is_keras_vectorizer,\
        '_is_spacy_vectorizer' : self._is_spacy_vectorizer,\
        '_is_matrix_2D' : self._is_matrix_2D,\
        '_is_spacy_data_preparation' : self._is_spacy_data_preparation,\
        '_oov_keyword' : self._oov_keyword,\
        '_entity_keyword' : self._entity_keyword,\
        'nb_word_most_frequent' : self.nb_word_most_frequent,\
        'is_tfidf' : self.is_tfidf,\
        'xpca' : self.xpca,\
        'is_df_copied' : self.is_df_copied}
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def __len__(self) :
        return len(self.df_data)
    #---------------------------------------------------------------------------    
    
    
    #---------------------------------------------------------------------------
    #   Properties
    #---------------------------------------------------------------------------
    def _get_tokenizer(self) :
        return self.kerasTokenizer
    def _set_tokenizer(self, tokenizer) :
        pass
        
    def _get_max_padding_length(self):
        return self.max_length
    def _set_max_padding_length(self, max_length):
        pass

    def _get_is_keras_vectorizer (self):
        return self._is_keras_vectorizer
    def _set_is_keras_vectorizer (self, is_keras_vectorizer):
        self._is_spacy_vectorizer = not is_keras_vectorizer
        self._is_keras_vectorizer = is_keras_vectorizer
        self._update_vector_column()

    def _get_is_spacy_vectorizer (self):
        return self._is_spacy_vectorizer
    def _set_is_spacy_vectorizer (self, is_spacy_vectorizer):
        self._is_keras_vectorizer = not is_spacy_vectorizer
        self._is_spacy_vectorizer = is_spacy_vectorizer
        self._update_vector_column()
        

    def _get_X (self):
        if self.COLUMN_NAME_VECTOR not in self.df_data.columns :
            print("\n*** ERROR : apply fit() and transform() first!")
            return None
        else :
            return p9_util.convert_ser2arr(self.df_data[self.COLUMN_NAME_VECTOR])


    def _set_X (self, X):
        pass        
        
    def _get_y (self):
        if self.COLUMN_NAME_TARGET not in self.df_data.columns :
            if (0 == len(self.list_df_data_file)) \
            or (self.list_df_data_file is None) :
                print("\n*** WARNING : No target!!")
                return None
            else :
                if self._target is not None :
                    return self._target
                else :
                    #---------------------------------------------------------------
                    # Data that is stored into files on harddisk is read and 
                    # processed.
                    #---------------------------------------------------------------
                    if 0 == self.total_row:
                        for df_data_file in self.list_df_data_file :
                            df_data = p5_util.object_load(df_data_file, is_verbose=False)
                            self.total_row += len(df_data)
                    else :
                        pass
                    start_row = 0

                    y = np.zeros(self.total_row)
                    for df_data_file in self.list_df_data_file :
                        df_data = p5_util.object_load(df_data_file, is_verbose=False)
                        end_row = start_row + len(df_data)
                        y[start_row:end_row] = p9_util.convert_ser2arr(df_data[self.COLUMN_NAME_TARGET])
                        start_row = end_row
                    self._target = y.copy()
                    return y
                
        else :
            return p9_util.convert_ser2arr(self.df_data[self.COLUMN_NAME_TARGET])

    def _set_y (self, y):
        pass        
    
    tokenizer = property(_get_tokenizer,_set_tokenizer)
    max_padding_length = property(_get_max_padding_length,\
    _set_max_padding_length)
    is_spacy_vectorizer =  property(_get_is_spacy_vectorizer,_set_is_spacy_vectorizer )
    is_keras_vectorizer =  property(_get_is_keras_vectorizer,_set_is_keras_vectorizer )
    X =  property(_get_X,_set_X)
    y =  property(_get_y,_set_y)
    
    
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _update_vector_column(self) :
        if self.COLUMN_NAME_VECTOR not in self.df_data.columns :
            pass
        else :
            del(self.df_data[self.COLUMN_NAME_VECTOR])
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _spacy_clean(self,X, df_data, column_name) :
        '''Cleaning process using spacy librarie includes :
            * Tokenization of any document using spacy tokenizer
            * Cleaning any document from punctuation
            * Cleaning any document from token out of range length.
            * Applying lemmatization over any token in document.
           Once the cleaning process is done, a DataFrame is created with 
           column name column_name that contains cleaned documents.
        Input :
            *   X : a list of documents.
            column_name : the name of the DataFrame column containing cleaned 
                          documents.
        Output :
            *   DataFrame
                          
        '''    
        columns_name_in = self.COLUMN_NAME_DOC
        
        df_data = p9_util_spacy.spacy_clean(self._corpus, df_data, column_name,\
         self.min_token_len, self.max_token_len,\
         word_most_frequent=self.nb_word_most_frequent)
        return df_data 
    #---------------------------------------------------------------------------
            
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _compute_counting(self, column_source, column_result):
        '''Computes the number of elements in dataframe column 'column_source'.
        The max number of elements is updated.
        '''
        
        self.df_data[column_result] = \
        self.df_data[column_source].apply(lambda list_token: len(list_token))
        if self.max_length is None :
            self.max_length = max(self.df_data[column_result])
        else :
            pass
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _convert_iter_type2array(self, obj):
        '''Returns an array if obj is convertible, None otherwise.
        '''
        if obj is None :
            return None
        else :
            pass

        if hasattr(obj,'__array__') :
            if isinstance(obj, np.ndarray):
                pass
            else :
                obj = np.array(obj)
        else :
            if hasattr(obj,'__iter__') :
                obj = np.array(obj)
            else :
                print("\n*** ERROR : _convert_iter_type2array :  object neither iter type nor array type!")
                return None
        return obj
    #---------------------------------------------------------------------------        
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def fit(self,X,y=None, index_df=None) :
        '''
            Input : 
                X : data that feeds this object 
                y : label related to X
                index_df : index of original dataframes. It is used to keep 
                relation between original observations and transformed 
                observations. 
        '''
        #-----------------------------------------------------------------------
        # Checking X parameter and convert X as an array if required.
        #-----------------------------------------------------------------------
        if X is None :
            print("\n*** ERROR : X parameter is None!")
            return
        else : 
            if not isinstance(X, list) and not isinstance(X, np.ndarray):
                print("\n***ERRROR : X parameter should be a list or array instance!")
                return
            else :
                if isinstance(X, list) :
                    X = self._convert_iter_type2array(X)                
                else :
                    pass
        #-----------------------------------------------------------------------
        # Get list of most frequent words; it will be used for cleaning corpus.
        #-----------------------------------------------------------------------
        self.list_word_most_frequent = p9_util.list_word_most_frequent(list(X), \
                              nb_most_frequent=self.nb_word_most_frequent)    
        print("fit : list_word_most_frequent length= {}".\
        format(len(self.list_word_most_frequent)))


        self.df_data = pd.DataFrame(pd.Series(X))
        self.df_data.rename(columns={0:self.COLUMN_NAME_DOC}, inplace=True)
        self._corpus = X.copy()
        if y is not None :
            y = self._convert_iter_type2array(y) 
            self._target = y.copy()
            self.df_data[self.COLUMN_NAME_TARGET] = pd.Series(y)
        if index_df is not None :
            self.df_data['original_index'] = pd.Series(index_df)
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _return_X_y(self) :
        
        if self._target is None :
            #print("_return_X_y : return X; target= {}".format(self._target))
            return self.X
        else :
            #print("_return_X_y : returne X and target= {}".format(self._target))
            return self.X,self.y
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _update_max_length(self, df_dataprep) :
        if self.max_length is None :
            df_dataprep[self.COLUMN_NAME_COUNT] = \
            df_dataprep[self.COLUMN_NAME_TOKEN].apply(lambda list_token: len(list_token))
            self.max_length = max(df_dataprep[self.COLUMN_NAME_COUNT])
        else :
            pass
        return df_dataprep
    #---------------------------------------------------------------------------
        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def fit_transform(self, X,y=None, index_df=None, predictor_model=None) :
        '''fit object with data and transforms data in order to feed model 
        given as paramater.
        This function process in one step data given as parameter. This 
        process stands for : 
            * corpus standardisation
            * corpus digitalization
            * Shaping data to feed predictor model.

            And if predictor_model is not None : 
            * apply prediction 
        Input : 
            X : corpus to be processed
            y : corpus labels
            index_df : an array containing indexes of the dataframe rows 
            before data transformation. This array of indexes is used in order
            to merge original dataframe with new one issued from data transformation.
            predictor_model : predictor model
        Output :
            if predictor_model is None, then shaped data to feed a predictor is returned.
            Otherwise, prediction is returnes as a Series object.
        '''
        #-----------------------------------------------------------------------
        # Fit 
        #-----------------------------------------------------------------------
        time_start0 = time.time()        
        time_start = time.time()
        x= self.fit(X, y=y, index_df=index_df)    
        time_end = time.time()
        time_last = time_end-time_start
        print("\nFit object with data : time= {0:.2f} sec".format(time_last))
        
        #-----------------------------------------------------------------------
        # Data-preparation with spacy; tokenization takes place into this
        # function.
        #-----------------------------------------------------------------------
        time_start = time.time()
        dict_param={'oov_keyword'  : self._oov_keyword,
            'entity_keyword' :self._entity_keyword,#'entity',
            'min_token_len': self.min_token_len,
            'max_token_len': self.max_token_len,
            }
        df_dataprep = p9_util_spacy.spacy_dataprep(self.df_data[self.COLUMN_NAME_DOC], \
                                                 **dict_param)

        #-----------------------------------------------------------------------
        # Drop rows with empty tokens
        #-----------------------------------------------------------------------
        column_name = p9_util.COLUMN_NAME_COUNT
        df_dataprep = df_dataprep[df_dataprep[column_name]>0].copy(deep=True)

        #-----------------------------------------------------------------------
        # Update max_length if required from df_dataprep that contains the 
        # max number of tokens for all documents from dataset.
        #-----------------------------------------------------------------------
        df_dataprep = self._update_max_length(df_dataprep)
                                                 
        time_end = time.time()
        time_last = time_end-time_start
        print("\nData preparation + tokenization: time= {0:.2f} sec".format(time_last))

        #-----------------------------------------------------------------------
        # Digitalization of tokens : a tensor is created.
        # Call  tree : 
        #   --> spacy_token2Tensor()
        #       --> spacy_list_token_2_tensor()
        #           --> spacy_list_token_2_matrix_2D()
        #
        #-----------------------------------------------------------------------
        time_start = time.time()
        ser_tensor = p9_util_spacy.spacy_token2Tensor(df_dataprep.docs, \
                                                  df_dataprep.tokens,\
                                                  dict_token_tfidf = self.dict_token_tfidf)
        time_end = time.time()
        time_last = time_end-time_start
        print("\nData digitalization : time= {0:.2f} sec".format(time_last))

        #-----------------------------------------------------------------------
        # Digitalization of tokens : a tensor is created.
        #-----------------------------------------------------------------------
        dict_param={
        'min_token_len': self.min_token_len,# Min number of characters in token
        'max_token_len': self.max_token_len,# Max number of characters in token
        'min_word_len' : self.min_doc_len,# Min number of words in text
        'max_word_len' : self.max_doc_len,# Max number of words in text
        'max_length'   : self.max_length,# Max length for padding
        'is_spacy_data_prep' : self._is_spacy_data_preparation,# Data preparation with Spacy
        'is_matrix_2D' : self._is_matrix_2D,# Use Spacy vectors dimension for documents vectorization, rather then norm
        'oov_keyword'  : self._oov_keyword,# Keyword that replaces out of vocabulary (oov) words
        'entity_keyword' :self._entity_keyword,#'entity',# Keyword that replaces taged entity words
        'return_df_data' : True,# df_data dataframe is returned from process
        'is_tokenizer' : False,# Do not use tokenization process has a simple tokenizer. 
                               # When True, then this function is used along with CountVectorizer.
        'df_dataprep' : df_dataprep,# Use df_dataprep given as parameter in tokenizer; this way, data-preparation 
                                    # will not be applied into this function.
        'ser_vector' : ser_tensor,# Use vectors transform it as Series in process, do not build it
        'target' : self._target,# Use target to build dataframe to be returned.
        } 
        
        #-----------------------------------------------------------------------
        # process of spacy_tokenizer() : if not already performed then :
        #   --> spacy_dataprep()
        #      --> spacy_clean()
        #      --> Computes number of tokens for each row of dataframe
        #      --> Builds a new dataframe with Series computed in this function and updates
        #          object dataframe. 
        #
        # If spacy_dataprep() has already been performed : 
        #   --> Remove rows from dataframe outside range [min_doc_len, max_doc_len]
        #   --> Returns X, y, dataframe depending of parameters into dict_param
        #       See p9_util_spacy.spacy_tokenizer() for more details.
        #-----------------------------------------------------------------------
        time_start = time.time()
        list_doc = df_dataprep[self.COLUMN_NAME_DOC].tolist()
        X, y, df_dataprep = p9_util_spacy.spacy_tokenizer(list_doc, **dict_param)
        if p9_util.COLUMN_NAME_INDEX in self.df_data :
            #-------------------------------------------------------------------
            # Indexes issued from df_dataprep are used as filters to keep rows 
            # that remain from datapreparation process
            #-------------------------------------------------------------------
            df_dataprep[p9_util.COLUMN_NAME_INDEX] = \
            self.df_data.index[df_dataprep.index].copy(dep=True)
        else :
            pass

        self.df_data = df_dataprep.copy(deep=True)
        time_end = time.time()
        time_last = time_end-time_start
        print("\nResize tensor rows: time= {0:.2f} sec".format(time_last))
                
        #-----------------------------------------------------------------------
        # Add a column to dataframe with matrix of embeddings tokens.
        #-----------------------------------------------------------------------
        time_start = time.time()
        self.df_data = p9_util_spacy.build_padded_matrix(self.df_data, \
                    dict_token_coeff=self.dict_token_tfidf, max_row = self.max_length)        
        time_end = time.time()
        time_last = time_end-time_start
        print("\nPadd / truncate tensor rows: time= {0:.2f} sec".format(time_last))

        #-----------------------------------------------------------------------
        # Deactivate file list issued from step by step processing; in this process 
        # data is cached on harddisk.
        #-----------------------------------------------------------------------
        self.list_df_data_file=list()

        if self.xpca is not None :
            #-------------------------------------------------------------------
            # Dimension reduction thanks to PCA
            #-------------------------------------------------------------------
            time_start = time.time()
            ipca_batch_size = self.dict_param_sequence['dict_param_step'][3]['ipca_batch_size']
            percent_var = self.dict_param_sequence['dict_param_step'][3]['percent_var']

            self.build_matrix_padded_truncated(ipca_batch_size, percent_var)      
            time_end = time.time()
            time_last = time_end-time_start
            print("\nDimension reduction: time= {0:.2f} sec".format(time_last))
            
            print("\nTotal time for data transformation= {0:.2f} secs".format(time_end-time_start0))
            print()
            
            X = self.df_data.matrix_padded_truncated
        else :
            X = self.df_data.matrix_padded	
            
        #-----------------------------------------------------------------------
        # drop unused columns
        #-----------------------------------------------------------------------
        for column in self.df_data.columns :
            if column in ['matrix_padded','original_index'] :
                pass
            else : 
                del(self.df_data[column])
        
        #-----------------------------------------------------------------------
        # Prepare input data in order to feed model prediction function : 
        #    * Complete tuple of dimensions, out of 1st dimension.
        #    * Compute shape for data feeding predictor.
        #    * Initialize array with shape previously computed
        #    * Transfer data into input
        #-----------------------------------------------------------------------
        list_tuple = list()
        list_tuple.append(len(X))
        for shape in X.iloc[0].shape:
            list_tuple.append(shape)

        input_shape = tuple(i for i in list_tuple)
        x_input = np.zeros(input_shape)

        for item in range(len(X)):
            x_input[item]= X.values[item].copy()

        #-----------------------------------------------------------------------
        # Feed model with input data
        #-----------------------------------------------------------------------
        if predictor_model is not None :
            self.df_data['predict'] = predictor_model.predict(x)[:,1]
            return self.df_data['predict']
        else :
            return x_input

    #--------------------------------------------------------------------------- 

        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def transform(self, X=None,y=None) :
        '''Transformation is the tokenization leading to: 
            *   unknown words detection and replacement with a key-word
            *   entity detection and replacement with a key-word
            *   cleaning sentences from baned words
            *   tokenization of any sentence resulting in a list of tokens for 
                each sentence.
            *   vectorization of tokens
            Input :
                * Arrays X,y; when X is None, then data that has been fited using 
                fit() method is processed. Then a dataframe belonging to this 
                object is feeded with some resulted steps of transformation.

                Otherwise, X is processed with steps described above.

            Ouput : 
                * Arrays X,y that has been transformed within the 
                  data preparation process.
        '''

        if X is None :
            if self.COLUMN_NAME_DOC not in self.df_data.columns :
                #---------------------------------------------------------------
                #  Tokenization process has to take place.
                #---------------------------------------------------------------
                print("\n***ERROR : transform() : apply method fit first!")
                return None, None
            else :
                if self.COLUMN_NAME_VECTOR not in self.df_data.columns :
                    #-----------------------------------------------------------
                    # No document given as parameter 
                    # corpus has been fited using fit() method
                    # vectorization did not take place.
                    # Vectorization is applied
                    #-----------------------------------------------------------
                    #print("transform : fit OK, Vectorization...")
                    dict_param={
                    'oov_keyword'  : self._oov_keyword,
                    'entity_keyword' :self._entity_keyword,#'entity',
                    'min_token_len': self.min_token_len,
                    'max_token_len': self.max_token_len,
                    }
                    df_dataprep = p9_util_spacy.spacy_dataprep(self.df_data[self.COLUMN_NAME_DOC], \
                                                 **dict_param)
                                                 
                    #-----------------------------------------------------------------------
                    # Drop rows with empty tokens
                    #-----------------------------------------------------------------------
                    column_name = p9_util.COLUMN_NAME_COUNT
                    df_dataprep = df_dataprep[df_dataprep[column_name]>0].copy(deep=True)

                    #-----------------------------------------------------------
                    # Update max_length if None using df_dataprep issued from 
                    # data-preparation.
                    #-----------------------------------------------------------
                    df_dataprep = self._update_max_length(df_dataprep)
                                                            
                    if self.is_tfidf :
                        tfidfVectorizer = TfidfVectorizer(tokenizer = p9_util_spacy.spacy_tokenizer)
                        self.csr_tfidf = tfidfVectorizer.fit_transform(df_dataprep[self.COLUMN_NAME_DOC])
                        for token, index in tfidfVectorizer.vocabulary_.items() :
                            self.dict_token_tfidf[token] = tfidfVectorizer.idf_[index]
                    else :
                        self.dict_token_tfidf = dict()
                        
                    ser_vector = p9_util_spacy.spacy_vectorizer(df_dataprep.docs, \
                                                  df_dataprep.tokens,\
                                                  dict_token_tfidf = self.dict_token_tfidf)

                    dict_param={
                    'min_token_len': self.min_token_len,# Min number of characters in token
                    'max_token_len': self.max_token_len,# Max number of characters in token
                    'min_word_len' : self.min_doc_len,# Min number of words in text
                    'max_word_len' : self.max_doc_len,# Max number of words in text
                    'max_length'   : self.max_length,# Max length for padding
                    'is_spacy_data_prep' : self._is_spacy_data_preparation,# Data preparation with Spacy
                    'is_matrix_2D' : self._is_matrix_2D,# Use Spacy vectors dimension for documents vectorization, rather then norm
                    'oov_keyword'  : self._oov_keyword,# Keyword that replaces out of vocabulary (oov) words
                    'entity_keyword' :self._entity_keyword,#'entity',# Keyword that replaces taged entity words
                    'return_df_data' : True,# df_data dataframe is returned from process
                    'is_tokenizer' : False,# Do not use tokenization process has a simple tokenizer. 
                                           # Last one are used with CountVectorizer.
                    'df_dataprep' : df_dataprep,# Use df_dataprep given as parameter in tokenizer
                    'ser_vector' : ser_vector,# Use vectors in Series in process, do not build it
                    'target' : self._target,# Use target to build dataframe to be returned.
                    }
                    list_doc = df_dataprep[self.COLUMN_NAME_DOC].tolist()
                    X, y, self.df_data=p9_util_spacy.spacy_tokenizer(list_doc, **dict_param)

                    #X, y , self.df_data= \
                    #p9_util_spacy.spacy_tokenizer(self._corpus,self._target,**dict_param)
                    return self._return_X_y()
                else : 
                    print("transform : fit OK, Corpus vectorized!")
                    #---------------------------------------------------------------------
                    # No document given as parameter, 
                    # and corpus has been fited,
                    # and vectorization already took place.
                    # X and y values are returned.
                    #---------------------------------------------------------------------
                    return self._return_X_y()
                    
        else : 
            #----------------------------------------------
            # Document or corpus is given as parameter
            #----------------------------------------------
            if self.COLUMN_NAME_DOC not in self.df_data.columns :
                #---------------------------------------------------------------
                #  Corpus has not been fit.
                #---------------------------------------------------------------
                print("\n***ERROR : transform() : apply method fit first!")
                return None, None
            else :
                pass

            if self.COLUMN_NAME_VECTOR not in self.df_data.columns :
                print("Data input digitalization...")
                #---------------------------------------------------------------
                # Vectorization has not been processed over the whole corpus 
                # that was previously fit.
                # The given X is vectorized using already vectorized corpus.
                #---------------------------------------------------------------
                dict_param={
                'min_token_len': self.min_token_len,
                'max_token_len': self.max_token_len,
                'min_word_len' : self.min_doc_len,
                'max_word_len' : self.max_doc_len,
                'max_length'   : self.max_length,
                'is_spacy_data_prep' : self._is_spacy_data_preparation,
                'is_matrix_2D' : self._is_matrix_2D,
                'oov_keyword'  : self._oov_keyword,
                'entity_keyword' :self._entity_keyword,#'entity',
                'return_df_data' : True,
                'is_tokenizer' : False,
                }
                X_, y_, self.df_data = \
                p9_util_spacy.spacy_tokenizer(self._corpus, self._target, **dict_param)

                return self._return_X_y()

            else :
                print("Corpus Vectorized: process given data...")
                #---------------------------------------------------------------
                # Vectorization of whole corpus already took place.
                # fit already took place.
                # Vectorization is applied to given X.
                #---------------------------------------------------------------         
                dict_param={
                'oov_keyword'  : self._oov_keyword,
                'entity_keyword' :self._entity_keyword,#'entity',
                'min_token_len': self.min_token_len,
                'max_token_len': self.max_token_len,
                }
                #----------------------------------------------------------------
                # Spacy tokenization of incoming document
                #----------------------------------------------------------------
                ser_corpus = pd.Series(X)
                df_dataprep = p9_util_spacy.spacy_dataprep(ser_corpus, **dict_param)
                
                #-----------------------------------------------------------------------
                # Drop rows with empty tokens
                #-----------------------------------------------------------------------
                column_name = p9_util.COLUMN_NAME_COUNT
                df_dataprep = df_dataprep[df_dataprep[column_name]>0].copy(deep=True)
                
                ser_doc = df_dataprep[self.COLUMN_NAME_DOC]
                ser_token = df_dataprep[self.COLUMN_NAME_TOKEN]

                #----------------------------------------------------------------
                # Spacy vectorization of tokenized document using corpus vectors.
                #----------------------------------------------------------------
                ser_vector = p9_util_spacy.spacy_vectorizer(ser_doc, ser_token, \
                                              dict_token_tfidf = self.dict_token_tfidf)
                
                #----------------------------------------------------------------
                # In case vectorization fails, then add document into list of 
                # erratic documents.
                #----------------------------------------------------------------
                if 0. == np.array(ser_vector.tolist()).sum() :
                    #print("transform : ser_token={}".format(ser_vector))
                    self.llist_error_doc.append(ser_doc.tolist())
                else :
                    pass

                X = p9_util.convert_ser2arr(ser_vector)
                return X

    #---------------------------------------------------------------------------
    

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def _tokenizer(self, corpus, target) :
    
        X = self._convert_iter_type2array(corpus)
        df_data = pd.DataFrame(pd.Series(X))
        df_data.rename(columns={0:self.COLUMN_NAME_DOC}, inplace=True)
        _corpus = X.copy()

        
        if self._is_spacy_data_preparation :
            #----------------------------------------------------------------------
            # Suspect words replacement
            #----------------------------------------------------------------------
            df_data['docs'] = df_data['docs'].apply(lambda doc: p9_util_spacy.spacy_oov_replace_suspicious(doc,replaced='suspect'))

            #----------------------------------------------------------------------
            # Entities detection and replacement
            #----------------------------------------------------------------------
            df_data['docs'] = df_data['docs'].apply(lambda doc: p9_util_spacy.spacy_entity_replace(doc))

        df_data = self._spacy_clean(df_data[self.COLUMN_NAME_DOC], \
        df_data, self.COLUMN_NAME_TOKEN)
        return df_data
        
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def fit_transform_step(self, X,y=None) :
        '''fit and object with data and transforms data in a step by step context.
        Step by step process takes place into test_datapreparator package.
        '''
        self.fit(X,y)
        return self.transform(None,None)
    #--------------------------------------------------------------------------- 


    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------

    def vectorValue2BinaryvectorLabel(self,vector_value=None):
        #-----------------------------------------------------------------
        # Fixing decimal_count=1 leads to have labels ranged 
        # from 0 to 10**1 ==> [0, 10]
        # Returned matrix will have 10 columns, one column for each class.
        #-----------------------------------------------------------------
        if vector_value is None :
            if self._target is None :
                print("\n*** ERROR : fit with Y target!")
                return None
            else :
                vector_value = self._target.copy()
        else :
            vector_value = self._convert_iter_type2array(vector_value)            

        vector_label_bin = p9_util.convert_vectorFloat_2_binaryLabel(vector_value, threshold=self.threshold, \
        direction = 1, decimal_count = 1)

        return vector_label_bin

        if False :
            decimal_count = 1
            nb_classes, matrix_label = p9_util.y_cont_2_label(vector_value, decimal_count=decimal_count)

            #-----------------------------------------------------------------------------
            # Transformation of y_train_label and y_test_label matrix into vectors named 
            # y_train_label_vect and y_test_label_vect.
            #-----------------------------------------------------------------------------
            vector_label = np.array([p9_util.get_label_from_row(matrix_label[row]) for row in range(0,len(matrix_label))])


            #--------------------------------------------------------------------------
            # Vectors binarization.
            # Threshold value will select vector values to 0 and vectors values to 1.
            #--------------------------------------------------------------------------
            threshold = 0
            direction = 1
            vector_label_bin = p9_util.multivalue2_binary(vector_label, threshold, direction)

        return vector_label_bin    
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def build_padded_matrix(self, bulk_row=0, root_filename=None,id_bulk_row = 0) :
        print("\nbuild_padded_matrix : Tokens to tensor transformation...")
        is_debug = False
        if bulk_row >0 :
            #-------------------------------------------------------------------
            # Filename for dataframe built step by step.
            #-------------------------------------------------------------------
            dataprep_filename = root_filename.split('/')[-1]
            folder_filename = root_filename.split(dataprep_filename)[0]
            dataprep_filename = 'df_data_'+dataprep_filename
            dataprep_filename = folder_filename+dataprep_filename

            start_row = 0
            i = 0
            #-------------------------------------------------------------------
            # Process is applied by bulk of bulk_row rows
            #-------------------------------------------------------------------
            length = self.df_data.shape[0]
            count_step = length//bulk_row
            #tail = length - count_step*bulk_row
            tail = length%bulk_row
            if tail >0 :
                #--------------------------------------------------------------
                # Last file will be processed in the tail section.
                #--------------------------------------------------------------
                count_step -=1

            #-------------------------------------------------------------------
            # Compute intermediate step from bulk_row and id_bulk_row.
            #-------------------------------------------------------------------
            i_step = id_bulk_row//bulk_row
            is_intermediate = False
            for i in range(i_step, count_step+1) : 
                #---------------------------------------------------------------
                # Intermediate loop in step by step file process 
                #---------------------------------------------------------------
                is_intermediate = True
                start_row = i*bulk_row
                end_row =  start_row + bulk_row
                print("Start row, end row= {} --> {}".format(start_row, end_row), end='\r')
                if end_row <= length :
                    if not is_debug :
                        df = self.df_data.iloc[start_row:end_row]
                        dfp = p9_util_spacy.build_padded_matrix(df, \
                                dict_token_coeff=self.dict_token_tfidf, \
                                max_row=self.max_length) 

                    print("{} --> {} / {} Done!".format(start_row, end_row, length), end='\r')
                    filename = dataprep_filename+'_'+str(i)+".dill"
                    if not is_debug :
                        p5_util.object_dump(dfp,filename, is_verbose=True)
                    self.list_df_data_file.append(filename)
                    start_row = end_row


                else :
                    pass
                    
            if tail >0 :
                if is_intermediate :
                    #-----------------------------------------------------------
                    # Increase name from previous file name built into 
                    # inermediate loop
                    #-----------------------------------------------------------
                    i+=1
                else : 
                    #-----------------------------------------------------------
                    # No intermediate loop took place. This is the first and 
                    # last file.
                    #-----------------------------------------------------------
                    i=0
                end_row   = start_row + tail
                if not is_debug :
                    df = self.df_data.iloc[start_row : end_row]
                    dfp = p9_util_spacy.build_padded_matrix(df, \
                            dict_token_coeff=self.dict_token_tfidf, \
                            max_row=self.max_length) 
                print("{} --> {} / {} Done!".format(start_row,end_row,length), end='\r')
                filename = dataprep_filename+'_'+str(i)+".dill"

                if not is_debug :
                    p5_util.object_dump(dfp,filename, is_verbose=True)
                self.list_df_data_file.append(filename)
        else :
            self.df_data = p9_util_spacy.build_padded_matrix(self.df_data, \
            dict_token_coeff=self.dict_token_tfidf, max_row = self.max_length)
    #---------------------------------------------------------------------------
        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def build_ipca_operator(self, batch_size) :
        ''' Build PCA operator over the whole digitalized corpus.
        For doing so, matrix is reshaped then all texts are merged into a single 
        column.
        '''
        len_matrix_padded_values = 0
        if len (self.list_df_data_file) >0 :
            #----------------------------------------------------------------
            # Read first file to get embedding dimension.
            #----------------------------------------------------------------
            df_data = p5_util.object_load(self.list_df_data_file[0], is_verbose=False)
            matrix_padded_values = np.array(list(df_data['matrix_padded'].values))
            dimension = matrix_padded_values.shape[-1]
            if False :
                for df_data_file in self.list_df_data_file :
                    df_data = p5_util.object_load(df_data_file)
                    matrix_padded_values = df_data['matrix_padded'].values
                    len_matrix_padded_values += matrix_padded_values.shape[0]
            else : 
                len_matrix_padded_values = self.total_row
        else :
            matrix_padded_values = np.array(list(self.df_data['matrix_padded'].values))
            len_matrix_padded_values = len(matrix_padded_values)

            #----------------------------------------------------------------------
            # Get the embedding dimension 
            #----------------------------------------------------------------------
            dimension = matrix_padded_values.shape[-1]
            
            #-----------------------------------------------------------------------
            # Reshape the tensor shaped as (M,N,D) where : 
            #   --> M is the number textes
            #   --> N is the number of tokens per text
            #   --> D is the embedding dimension 
            # into KxD matrix where :
            #   --> K =MxN is the total number of text (rows)
            #   --> D is the embedding dimension
            #-----------------------------------------------------------------------            
            matrix_padded_values = matrix_padded_values.reshape((-1,dimension))


        #print("Dimension, Batch size= ({},{})".format(dimension, batch_size))
        #batch_size = max(batch_size, dimension)
        #print("Dimension, Batch size= ({},{})".format(dimension, batch_size))
        
        #print("matrix_padded_values= {}".format(matrix_padded_values.shape))
        #print("Reshaped matrix_padded_values= {}".format(matrix_padded_values.shape))
        if self.xpca is None :
            print("\nBuild of Incremental PCA operator...")
            self.xpca = IncrementalPCA(n_components=dimension, batch_size=batch_size)
            new_batch_size = max(batch_size, dimension)
            if batch_size !=  new_batch_size:
                print("\nWARN : batch size recomputed from {} to {}".format(batch_size, new_batch_size))
                batch_size = new_batch_size
            else :
                pass

            epoch = len_matrix_padded_values//batch_size
            if epoch == 0 :
                epoch = 1
            end_row = 0
            if 0 == len (self.list_df_data_file) : 
                print("Epoch= {}".format(epoch))
                for i in range(epoch) :
                    start_row = i*batch_size
                    end_row = start_row + batch_size

                    print("Batch range = {} --> {} / {}".format(start_row, end_row, len_matrix_padded_values), end='\r')
                    self.xpca = self.xpca.partial_fit(matrix_padded_values[start_row:end_row,:])
                    print("PCA partial fit : {}/{}".format(i+1,epoch), end='\r')
            else : 
                i_file = 0
                for df_data_file in self.list_df_data_file :
                    df_data = p5_util.object_load(df_data_file)
                    matrix_padded_values = np.array(list(df_data['matrix_padded'].values))
                    matrix_padded_values = matrix_padded_values.reshape((-1,dimension))
                    len_matrix_padded_values = len(matrix_padded_values)
                    epoch = len_matrix_padded_values//batch_size
                    print("\nFile #{} Epoch= {}".format(i_file, epoch))
                    for i in range(epoch) :
                        start_row = i*batch_size
                        end_row = start_row + batch_size
                        self.xpca = self.xpca.partial_fit(matrix_padded_values[start_row:end_row,:])
                        print("Batch range = {} --> {} / {}".\
                              format(start_row, end_row, len_matrix_padded_values), end='\r')
                    
                    tail = len_matrix_padded_values-epoch*batch_size
                    if 0 < tail :
                        start_row = end_row
                        end_row = start_row + tail
                        self.xpca = self.xpca.partial_fit(matrix_padded_values[start_row:end_row,:])
                        print("Batch range = {} --> {} / {}".\
                              format(start_row, end_row, len_matrix_padded_values), end='\r')
                        
                    i_file += 1
        else :
            print("\nPCA operator already built !")

        #return self
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def matrixpadded_truncate(self, percent_var) :
        print("\nDimensions reduction...")
        #----------------------------------------------------------------------
        # Get max components considering expected percentage of variance
        #----------------------------------------------------------------------
        max_component = p3_util_plot.get_component_from_cum_variance(self.xpca,percent_var)
        if max_component is None :
            print("\n***ERROR : matrixpadded_truncate() : decomposition failed!")
            return

        #----------------------------------------------------------------------
        # Transform all matrix_padded into truncated matrix_padded
        #----------------------------------------------------------------------
        if len (self.list_df_data_file) >0 :
            #----------------------------------------------------------------
            # Read files and apply PCA transformation
            #----------------------------------------------------------------
            for filename in self.list_df_data_file:
                df_data = p5_util.object_load(filename)
                ser_matrix_padded = df_data['matrix_padded']
                ser_matrix_padded_truncated = ser_matrix_padded.apply(lambda matrix_padded : self.xpca.transform(matrix_padded)[:,:max_component])
                df_data['matrix_padded_truncated'] = ser_matrix_padded_truncated
                
                #----------------------------------------------------------------------
                # Drop unsued columns
                #----------------------------------------------------------------------
                if 'matrix_padded' in self.df_data.columns :
                    del(df_data['matrix_padded'])

                if 'tokens' in self.df_data.columns :
                    del(df_data['tokens'])    

                if 'vector' in df_data.columns :
                    del(df_data['vector'])    

                #----------------------------------------------------------------------
                # Save dataframe into filename.
                #----------------------------------------------------------------------
                p5_util.object_dump(df_data, filename)
                print("Processed file = {}".format(filename))
                print("")
        else : 
            ser_matrix_padded = self.df_data['matrix_padded']
            ser_matrix_padded_truncated = ser_matrix_padded.apply(lambda matrix_padded : self.xpca.transform(matrix_padded)[:,:max_component])

            #----------------------------------------------------------------------
            # Aggregate result into dataframe
            #----------------------------------------------------------------------
            self.df_data['matrix_padded_truncated'] = ser_matrix_padded_truncated

        #----------------------------------------------------------------------
        # Drop unsued columns
        #----------------------------------------------------------------------
        if 'matrix_padded' in self.df_data.columns :
            del(self.df_data['matrix_padded'])

        if 'tokens' in self.df_data.columns :
            del(self.df_data['tokens'])    
        
        #return self
    #---------------------------------------------------------------------------
        
        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def build_matrix_padded_truncated(self, batch_size, percent_var):
        ''' Reduce dimension of corpus represented as a matrix.
        Reduction is achieved with PCA decomposition. The result of dimension 
        reduction is driven by expected percentage of variance.
        
        Input :
            * batch_size : size of incremental PCA batch of samples.
            This size has to be greater or equal to the word dimensions.
            If this is not the case, then batch size is re-computed as max(dimension, batch_size).
            It the number of samples can't feed a batch, then an error is returned.
            * percent_var : percentage of explained variance expected after dimension reduction.
            This valuee leads the new dimension value ater reduction.

        Output :
            
        '''
        #-----------------------------------------------------------------------
        # Build PCA operator in an incremental manner
        #-----------------------------------------------------------------------
        self.build_ipca_operator(batch_size)

        #-----------------------------------------------------------------------
        # Use PCA in order to truncate matrix dimension based on expected variance.
        #-----------------------------------------------------------------------
        self.matrixpadded_truncate(percent_var)
        
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    
    
    
    #-------------------------------------------------------------------------------
    #
    #-------------------------------------------------------------------------------
    def targetUpdate2BinaryLabel(self, threshold=None):
        '''Convert target composed of continuous values into a binary label 
        vector.
        Target vector may be in local dataframe or into dataframes recorded 
        on hardidk. 
        
        Input : 
            * threshold : value over which value will be converted to 1.
                          When this value is None, then threshold value from 
                          this object is used.
                          Otherwise,  threshold value from this object is updated.
        '''
        if threshold is None :
            #-------------------------------------------------------------------
            # use threshold from this object.
            #-------------------------------------------------------------------
            threshold = self.threshold
        else :
            #-------------------------------------------------------------------
            # Update threshold
            #-------------------------------------------------------------------
            self.threshold = threshold
            
        y = self.y
        #-----------------------------------------------------------------------
        # tuple_transformation : 
        #   * threshold : 0.
        #   * Direction : >=
        # This leads to : if y > 0.0 
        #                    then value is 1
        #                 else :
        #                    value is 0
        #-----------------------------------------------------------------------
        y_label = p9_util.convert_vectorFloat_2_binaryLabel(y, threshold=threshold, \
        direction = 1, decimal_count = 1)
        ser = pd.Series(y_label, dtype=int)        
        column_name = self.COLUMN_NAME_TARGET
        if column_name in self.df_data :
            del(self.df_data[column_name])
        else :
            pass
        self.df_data[column_name] = ser.copy(dep=True)

    #-------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------
    #
    #-------------------------------------------------------------------------------
    def targetUpdate2BinaryLabel_deprecated(self, threshold=None):
        '''Convert target composed of continuous values into a binary label 
        vector.
        Target vector may be in local dataframe or into dataframes recorded 
        on hardidk. 
        
        Input : 
            * threshold : value over which value will be converted to 1.
                          When this value is None, then threshold value from 
                          this object is used.
                          Otherwise,  threshold value from this object is updated.
        '''
        if threshold is None :
            #-------------------------------------------------------------------
            # use threshold from this object.
            #-------------------------------------------------------------------
            threshold = self.threshold
        else :
            #-------------------------------------------------------------------
            # Update threshold
            #-------------------------------------------------------------------
            self.threshold = threshold
            
        y = self.y
        y_label = self.vectorValue2BinaryvectorLabel(vector_value=y)

        #-----------------------------------------------------------------------
        # tuple_transformation : 
        #   * threshold : 0.
        #   * Direction : >=
        # This leads to : if y > 0.0 
        #                    then value is 1
        #                 else :
        #                    value is 0
        #-----------------------------------------------------------------------
        y_label = p9_util.matrixLabel2vectorBinaryLabel(y_label,tuple_transformation=(threshold,1)) 
        ser = pd.Series(y_label, dtype=int)        
        column_name = self.COLUMN_NAME_TARGET
        if column_name in self.df_data :
            del(self.df_data[column_name])
        else :
            pass
        self.df_data[column_name] = ser.copy()

    #-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------


