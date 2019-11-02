#!/usr/bin/python3.6
#-*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.base import TransformerMixin 

import spacy
import pandas as pd
import keras

import p3_util_plot
import p5_util
import p9_util
import p9_util_spacy
import KerasTokenizer

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class DataPreparator(TransformerMixin):
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
                 nb_word_most_frequent=100):
        if other is None :             
            self.df_data = pd.DataFrame()
            self._corpus = None
            self._target = None
            self.slice_length = slice_length
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
        
            self.spacy_nlp     = spacy.load(spacy_model_language)
        else :
            if not isinstance(other, type(self)) :
                print("\n*** ERROR : other is not instance of DataPreparator!")

            else :
                self.df_data = pd.DataFrame()
                self._corpus = None
                self._target = None
                self.csr_tfidf = None
                self.llist_error_doc = list()
                
                #-----------------------------------------------------------------------
                # Parameters issued from oher object.
                #-----------------------------------------------------------------------
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
                self.spacy_nlp     = other.spacy_nlp
            
    #---------------------------------------------------------------------------

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
                #---------------------------------------------------------------
                # Data are stored into files on harddisk
                #---------------------------------------------------------------
                y = np.zeros(self.total_row)
                start_row = 0
                for df_data_file in self.list_df_data_file :
                    df_data = p5_util.object_load(df_data_file)
                    end_row = start_row + len(df_data)
                    y[start_row:end_row] = p9_util.convert_ser2arr(df_data[self.COLUMN_NAME_TARGET])
                    start_row = end_row
                
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
        '''Returns an array if obj is conertible, None otherwise.
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
    def fit(self,X,y=None) :
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
                    #---------------------------------------------------------------------
                    # No document given as parameter 
                    # corpus has been fited using fit() method
                    # vectorization did not take place.
                    # Vectorization is applied
                    #---------------------------------------------------------------------
                    #print("transform : fit OK, Vectorization...")
                    dict_param={
                    'oov_keyword'  : self._oov_keyword,
                    'entity_keyword' :self._entity_keyword,#'entity',
                    'min_token_len': self.min_token_len,
                    'max_token_len': self.max_token_len,
                    }
                    df_dataprep = p9_util_spacy.spacy_dataprep(self.df_data[self.COLUMN_NAME_DOC], \
                                                 **dict_param)
                    
                    tfidfVectorizer = TfidfVectorizer(tokenizer = p9_util_spacy.spacy_tokenizer)
                    self.csr_tfidf = tfidfVectorizer.fit_transform(df_dataprep[self.COLUMN_NAME_DOC])
                    for token, index in tfidfVectorizer.vocabulary_.items() :
                        self.dict_token_tfidf[token] = tfidfVectorizer.idf_[index]
                    
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
                #print("Corpus Not Vectorized")
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
                #print("Corpus Vectorized")
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
    def transform_deprecated_2(self, X=None,y=None) :
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
                print("\n***ERROR : apply method fit first!")
                print(self.df_data)
                return None, None
            else :
                #---------------------------------------------------------------
                # fit already took place.
                # Tokenization is applied to corpus fited.
                # Customized spacy tokenization process is applied over 
                # recorded  corpus 
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

                X, y , self.df_data= \
                p9_util_spacy.spacy_tokenizer(self._corpus,self._target,**dict_param)
                return X,y
        else : 
            if self.COLUMN_NAME_DOC not in self.df_data.columns :
                #---------------------------------------------------------------
                #  Tokenization process has to take place.
                #---------------------------------------------------------------
                print("\n***ERROR : apply method fit first!")
                print(self.df_data)
                return None, None
            else :
                pass
            
            if self.COLUMN_NAME_VECTOR not in self.df_data.columns :
                #---------------------------------------------------------------
                #   Vectorization has not been processed over the whole corpus.
                #   The given X is vectorized using already vectorized corpus.
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
                p9_util_spacy.spacy_tokenizer(self._corpus, self._target, \
                **dict_param)
                

                dict_param['return_df_data'] = False
                X = p9_util_spacy.spacy_tokenizer(X, None, **dict_param)

                return X
            else :
                #---------------------------------------------------------------
                #   Vectorization of whole corpus already took place.
                #---------------------------------------------------------------
                #---------------------------------------------------------------
                # fit already took place.
                # Tokenization is applied to corpus fited.
                # Customized spacy tokenization process is applied over 
                # recorded  corpus 
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
                'return_df_data' : False,
                'is_tokenizer' : False,
                }

                X = p9_util_spacy.spacy_tokenizer(X, None, **dict_param)
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
    def transform_deprecated(self, X=None,y=None) :
        '''Transformation is the tokenization leading to: 
            *   unknown words detection and replacement with a key-word
            *   entity detection and replacement with a key-word
            *   cleaning sentences from baned words
            *   tokenization of any sentence resulting in a list of tokens for 
                each sentence.
            *   vectorization of tokens
            Input :
                * Arrays X,y; when None, then data that has been fited using 
                fit() method is processed. Then a dataframe beolonging to the 
                object is feeded with some resulted steps of transformation.
                Otherwise, X,y is processed with steps described above.
            
            Ouput : 
                * Arrays X,y that has been transformed within the 
                  data preparation process.
        '''
        
        #-----------------------------------------------------------------------
        #  Data is cleaned and tokenized using Scapy librarie.
        #  Tokenized text are stored into colmn 'tokens' from dataframe.
        #-----------------------------------------------------------------------
        if self.COLUMN_NAME_DOC not in self.df_data.columns :
            print("\n***ERROR : apply method fit first!")
            print(self.df_data)
            return None, None
        else :
            pass
            
        if self.COLUMN_NAME_TOKEN not in self.df_data.columns :
            if self._is_spacy_data_preparation :
                #----------------------------------------------------------------------
                # Suspect words replacement
                #----------------------------------------------------------------------
                self.df_data['docs'] = self.df_data['docs'].apply(lambda doc: p9_util_spacy.spacy_oov_replace_suspicious(doc,replaced='suspect'))

                #----------------------------------------------------------------------
                # Entities detection and replacement
                #----------------------------------------------------------------------
                self.df_data['docs'] = self.df_data['docs'].apply(lambda doc: p9_util_spacy.spacy_entity_replace(doc))
            else :
                pass

            self.df_data = self._spacy_clean(self.df_data[self.COLUMN_NAME_DOC],\
             self.df_data, self.COLUMN_NAME_TOKEN)
        else :
            print("\nClean already processed!")

        #-----------------------------------------------------------------------
        # Count tokens in column 'tokens' and save result into column 'counting'
        #-----------------------------------------------------------------------
        if self.COLUMN_NAME_COUNT not in self.df_data.columns :
            self._compute_counting(self.COLUMN_NAME_TOKEN,self.COLUMN_NAME_COUNT)
        else: 
            print("\nCounting column "+str(self.COLUMN_NAME_TOKEN)+" already processed!")

        #-----------------------------------------------------------------------
        #  Vectorization takes place using Keras or Spacy
        #-----------------------------------------------------------------------
        if self.COLUMN_NAME_VECTOR not in self.df_data.columns :
            if self._is_keras_vectorizer :
                dtype_for_padding = 'int'
                self.kerasTokenizer.fit_on_texts(self.df_data[self.COLUMN_NAME_TOKEN])
                self.df_data[self.COLUMN_NAME_VECTOR] = \
                self.kerasTokenizer.texts_to_sequences(self.df_data[self.COLUMN_NAME_TOKEN])
                self._is_matrix_2D = False


            elif self._is_spacy_vectorizer: 
                if self._is_matrix_2D : 
                    self.df_data[self.COLUMN_NAME_VECTOR] \
                    = self.df_data[self.COLUMN_NAME_TOKEN].apply(lambda list_token: \
                                                                 p9_util_spacy.spacy_list_token_2_vector(list_token))
                else :
                    dtype_for_padding = 'float'
                    self.df_data[self.COLUMN_NAME_VECTOR] \
                    = self.df_data[self.COLUMN_NAME_TOKEN].apply(lambda list_token: \
                                                                 p9_util_spacy.spacy_token_norm_vectorization(list_token))
                #print(self.df_data[self.COLUMN_NAME_VECTOR])                                              
            else : 
                print("\n*** ERROR : no vectorizer option selected!")
                return None, None
                        
            #-------------------------------------------------------------------
            # Vectorized texts are padded for all texts having same length
            #   * An array is built from pad_sequences
            #   * A Series is built from previous array 
            #-------------------------------------------------------------------
            if self._is_matrix_2D :
                pass
            else :
                arr = keras.preprocessing.sequence.pad_sequences(self.df_data[self.COLUMN_NAME_VECTOR],\
                     maxlen=self.max_length, padding='post',dtype=dtype_for_padding)
                     
                self.df_data[self.COLUMN_NAME_VECTOR] = pd.Series([arr[i] for i in range(len(arr))])
            
            self.df_data[self.COLUMN_NAME_TARGET] = pd.Series(self._target)
            
            #-----------------------------------------------------------------------
            #    Remove rows from dataframe outside range [min_doc_len, max_doc_len]
            #-----------------------------------------------------------------------
            list_index_drop = self.df_data[self.df_data[self.COLUMN_NAME_COUNT]<self.min_doc_len].index
            self.df_data.drop(index=list_index_drop, inplace=True)

            list_index_drop = self.df_data[self.df_data[self.COLUMN_NAME_COUNT]>self.max_doc_len].index
            self.df_data.drop(index=list_index_drop, inplace=True)
        else : 
            print("\nAlready vectorized!")

        #-----------------------------------------------------------------------
        # return X and y as arrays.
        #-----------------------------------------------------------------------
        X = p9_util.convert_ser2arr(self.df_data[self.COLUMN_NAME_VECTOR])
        y = p9_util.convert_ser2arr(self.df_data[self.COLUMN_NAME_TARGET])
        #ser = self.df_data[self.COLUMN_NAME_VECTOR]
        #X = ser.tolist()
        return X,y

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def fit_transform(self, X,y=None) :
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
#-------------------------------------------------------------------------------


