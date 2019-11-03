#!/usr/bin/python3.6
#-*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import time
import string

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

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


from sklearn.utils.class_weight import compute_class_weight

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import gensim
from gensim.models import Word2Vec

import nltk
#nltk.download('stopwords')  # run once
from nltk.corpus import stopwords



import p4_util
import p5_util
import p8_util 
import p9_util


#-------------------------------------------------------------------------------
#   Constants for the whole package.
#-------------------------------------------------------------------------------
COLUMN_NAME_DOC    = 'docs'
COLUMN_NAME_TARGET = 'target'
COLUMN_NAME_TOKEN  = 'tokens'
COLUMN_NAME_COUNT  = 'counting'
COLUMN_NAME_VECTOR = 'vector'
COLUMN_NAME_INDEX  = 'original_index'

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_sample(X, y, ratio=-1) :
    return p8_util.get_sample(X,y,ratio)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def clean_X_label(X, label) :
    
    ser_ = pd.Series(X)

    list_index = [i for i in ser_.index if len(ser_[i])==0]
    
    ser_.drop(list_index, inplace=True)
    list_to_clean_1 = ser_.tolist()
    
    print("Cleaned empty text = {}".format(len(list_index)))
    ser_ = pd.Series(label)
    ser_.drop(list_index, inplace=True)

    list_to_clean_2 = ser_.tolist()
    
    return list_to_clean_1, list_to_clean_2
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def load_dataset_jigsaw(sampling_ratio =None) :
    
    X_train, y_train, X_test, y_test = \
    p8_util.load_dataset_jigsaw(sampling_ratio = sampling_ratio)
    
    
    return X_train, y_train, X_test, y_test
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def preprocess_dataset_jigsaw(X_train, X_test) :
    x_train, x_test = \
    p8_util.preprocess_dataset_jigsaw(X_train, X_test)
    
    return x_train, x_test
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_extension_from_name(name, extension, separator) :
    '''Returns extension from name given as parameter.
    Extension is a string that is defined as following : separator+extension
    
    '''

    list_split = name.split(separator)
    ret_extension=''
    for split in list_split :
        
        if extension in split:
            ret_extension=separator+extension
            break
    return ret_extension
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def X_y_filter_max_length(list_X, list_y, maxLength):
    '''Keep texts from list_X with number of words <= maxLength and 
    with number of words > 0
    
    Input : 
        * list_X : this is a list of tokenized texts.
        * list_y : target values matching with list_X. It is provided in order 
        (X, target) to be matched after filtering.
        * maxlength : filter criteria; texts with words > maxLength are avoided.
    Output :
        * Filtered dataset (X,y)
    '''
    list_X_filtered = list()
    list_y_filtered = list()
    for ((index, list_token), target) in zip(enumerate(list_X), list_y) :
        if len(list_token) <= maxLength and len(list_token) > 0 :
            list_X_filtered.append(list_token)
            list_y_filtered.append(target)
        else :
            pass
    return list_X_filtered, list_y_filtered    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_trained_w2vec(model_filename, X, epochs, embeddings_dim=100, is_saved=True):
    ''' Provides a word2vec model.
    If given parameter is_saved is True, then model is computed from X and dumped into a file.
    Otherwise, model is read from a dumped file.
    
    Input :
        * model_filename : model file name used to save or load word2vec model.
        * X : list of tokenized sentences used to build model.
        * epochs : number of times model is trained over X
        * embeddings_dim : dimension of the vectors words.
        * is_saved : flag from which value will lead to build & save or load word2vec model.
            If False : W2VEC model is built and saved.
            Otherwise, W2VEC model is loaded from file.
    Output :
        * trained word2vec model.
    '''
    #-------------------------------------------------------------
    # Check if a model with same name still exists
    #-------------------------------------------------------------
    if model_filename.split('/')[-1] in os.listdir('./data'):
        is_w2vec_saved = True
    else :
        is_w2vec_saved = False
        if is_saved is True :
            print("\n*** ERROR : File name= {} Unknown! W2VEC model can't be loaded!".format(model_filename))
            return None

    print(is_w2vec_saved, is_saved)
    
    if is_w2vec_saved is False :
        #-------------------------------------------------------------------------------------
        # Some words are obfuscated. Then they may appear once in all corpus.
        # In order to avoid this, then min_count is fixed to 1.
        # Window is fixed to 4 : 2 words before and 2 words after central word.
        #-------------------------------------------------------------------------------------
        print("Training W2VEC model...")
        model_w2vec = Word2Vec(min_count=2, workers=6, size=embeddings_dim)
        model_w2vec.build_vocab(X)  # prepare the model vocabulary
        model_w2vec.train(X, total_examples=model_w2vec.corpus_count, epochs=epochs, compute_loss=True) 
        print("Done!\n")
        if is_saved is False :
            model_w2vec.save(model_filename)
            print("Model saved!\n")
    else :
        print("Loading W2VEC model...")
        model_w2vec = Word2Vec.load(model_filename)
        print("Done!\n")
    return model_w2vec
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def w2vec_embeddings_build(w2vec, tokenizer=None): 
    '''Build embeddings matrix based on W2VEC model.
    Number of matrix rows = vocabulary size issued from W2VEC model.
    Number of matrix columns = dimension of W2VEC vectors.
    '''
    
    #---------------------------------------------------------------------------
    # Tokenizer is selected depending of the one provided as function parameter.
    #---------------------------------------------------------------------------
    if tokenizer is None :
        vocabulary_size = len(w2vec.wv.vocab)
        vocab = w2vec.wv.vocab
    else :
        vocabulary_size = len(tokenizer.word_index)+1
        vocab = tokenizer.word_index

    #---------------------------------------------------------------------
    # Embeddings matrix initialization
    #---------------------------------------------------------------------
    embeddings_dim = w2vec.vector_size
    w2vec_embeddings_matrix = np.zeros((vocabulary_size, embeddings_dim))
    
    row = 0
    for word in vocab.keys() :
        if word in w2vec.wv :
            w2vec_embeddings_matrix[row] = w2vec.wv[word]
            row += 1
    return w2vec_embeddings_matrix, vocab
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_corpus_matrix(w2vec, llist_X, monitoring=10000) :
    '''
        Input :
            * w2vec : Word 2 Vec model in which each word of vocabulary has
            learned coordonates in corpus vector space.
            
            * llist_X : list of list of tokens.
              Each element from llist_X is a text from corpus that has been 
              tokenized.
              Each text from corpus is tokenized as a list of tokens.
        Output :
            * A matrix of the given corpus with :
              Columns : dimension of the vector space
              Raws : texts from corpus
              
    '''
    
    #---------------------------------------------------------------------
    # Dimension is extracted from word2vec model.
    #---------------------------------------------------------------------
    dim = w2vec.wv.vectors.view().shape[1]



    #---------------------------------------------------------------------
    # Corpus of text is rebuilt with words belonging to w2vec vocabulary only.
    # * llist_X is a list of lists, where former are lists of tokenized sentences.
    #---------------------------------------------------------------------
    llist_X_filtered = list()
    list_index_excluded = list()
    index = 0
    for list_token in llist_X:
        list_X_filtered = [token for token in list_token if token in w2vec.wv.vocab]
        if len(list_X_filtered) >0 :
            llist_X_filtered.append(list_X_filtered)
        else :
            # No token from list_word belongs to W2VEC vocabulary.
            list_index_excluded.append({index:list_token})
        index +=1

    #---------------------------------------------------------------------
    # Embeddings matrix and vectors initialization
    #---------------------------------------------------------------------
    w2vec_corpus_matrix = np.zeros((len(llist_X_filtered), dim))
    #embeddings_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        
    #---------------------------------------------------------------------
    # Each tokenized sentence from corpus is digitalized as a vectors. 
    # Tokens from tokenized sentences are converted as vectors using W2VEC 
    # model. 
    # Mean of each vectors are computed given a vector, named sentence_vector. 
    # That vector represents the tokenized sentence. 
    #
    # Any tokenized sentence, list_token, is extracted from llist_X_filtered
    # All tokens from list_token belong to W2VEC model vocabulary.
    #
    # Once sentence_vector is built, it is stacked to embeddings matrix.
    #---------------------------------------------------------------------
    index = 0
    for list_token in llist_X_filtered :
        sentence_vector = np.mean( [ w2vec.wv[token] for token in list_token ], axis=0 )
        #---------------------------------------------------------------------------------------
        # Matrix of embeddings is computed.
        # Each raw is a sentence from corpus while each column is a feature 
        # of the corpus extracted with WORD2VEC.
        #---------------------------------------------------------------------------------------
        #embeddings_matrix = np.vstack((corpus_matrix,sentence_vector))
        w2vec_corpus_matrix[index] = sentence_vector
        index += 1
        if index % 10000 == 0 :
            print("Building Embeddings matrix : {}/{}".\
            format(index,len(llist_X_filtered)))
    #--------------------------------------------------------
    # Remove first raw from matrix used formatrix initialization
    #--------------------------------------------------------
    #embeddings_matrix = embeddings_matrix[1:]
    #print("\n Number of unique words from W2EVC vocabulary = {}".format(len(list_unique_word)))
    return w2vec_corpus_matrix, list_index_excluded
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------    
def get_excluded_dict_index_list(X, w2vec_model):
    ''' Returns a dictionary {index:list_token} where : 
        * index identifies the index of list of tokens into X
        * list_token : the list of token that has been excluded from w2vec model.
        list_token represents a sentence from corpus.
        
        Input : 
            X : a list of list of tokens.
            w2vec_model : Word2Vec model 
        Output : 
            Dictionary  {index:list_token} as described above.
            list_token 
    '''
    llist_X_filtered = list()
    dict_index_list = dict()
    index = 0
    for list_word in X:
        list_X_filtered = [word for word in list_word if word in w2vec_model.wv.vocab]
        if len(list_X_filtered) == 0 :
            dict_index_list[index] = list_word
        else :
            pass
        index +=1
    return dict_index_list
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------    
def bow_matrix_build(document, maxLength=None, tokenizer=None, is_padded=False) :
    '''Tokenize and padd a document.
    Keras tokenizer is used in order to vectorize texts.
    Eacn vector is sized as the text length.
    This means each vector as different size.   
    Word values issued from keras tokenizer denote values from keras dictionary.
    Keras dictionary is reached with tokenizer.word_index attribute.
        
    maxLength is used in order to padd any document.
    
    Input : 
        *   document : a string to be tokenized and padded.
        *   maxLength : maximum size of document when padded.
        *   tokenizer : when None, then keras tokenizer is used in order to 
                        tokenize and padd the given document.
        *   is_padded : when this flag is True, then each document is padded.
                        This result of a vectorized corpus with all vectors 
                        with same length (dimention).
    Output :
        *   A matrix of the document tokenized with sequences of integers and 
            padded if is_padded Flag is fixed to True.
        *   The tokenizer used in the process.
    '''
    if tokenizer is None :
        tokenizer = keras.preprocessing.text.Tokenizer()
        
        # Build vocabulary based on corpus X
        # Note : X is not provided??
        tokenizer.fit_on_texts(document)
    else :
        pass

    #---------------------------------------------------------------------------
    # Documents are encoded with sequences integers
    #---------------------------------------------------------------------------
    document_encoded = tokenizer.texts_to_sequences(document)
    
    if is_padded :
        #-----------------------------------------------------------------------
        # Document is padded : this is due to the fact keras tokenizer encodes
        # texts and encoded text has the length of text.
        # Then each encoded text has a different length.
        #-----------------------------------------------------------------------
        if maxLength is None :
            print("\n***ERROR : `maxLength` parameter should be fixed when flag `is_padded` is activated!")
            return None, None
        else :
            document_padded = keras.preprocessing.sequence.pad_sequences(document_encoded,\
             maxlen=maxLength, padding='post')
            return document_padded, tokenizer
    else : 
        #-----------------------------------------------------------------------
        # Document is encoded with vector dimension as text length.
        #-----------------------------------------------------------------------        
        return document_encoded, tokenizer
#-------------------------------------------------------------------------------    
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------    
def X_y_truncate_max_length(list_list_token, list_target, maxLength):
    '''Truncate any document as maxLength size.
    Input :
        * list_list_token : a list of tokenized documents; each document is a 
        list of tokens.
        * list_target : a list of targets mathcing with list_token
        * maxLength : maximum length for any document.
    Output :
        * arrays of truncated documents with array of matching targets.
    '''
    list_list_truncated = list()
    list_y_target = list()
    for ((index, list_token), target) in zip(enumerate(list_list_token), list_target) :
        if len(list_token) <= maxLength and len(list_token) > 0 :
            list_list_truncated.append(list_token)
            list_y_target.append(target)
        else :
            pass
    return np.array(list_list_truncated), np.array(list_y_target)    
    
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------    
def y_cont_2_label(y_cont, decimal_count=1, value_type='vector'):
    '''This function converts continuous values into qualitative values.
    It is used to convert a regression problem into a classification problem.
    
    Input:
        * y_cont : a vector continuous values.
        * decimal count : number of decimals to be taken into acccount when 
            y_cont is converted into label value.
        * decimal_exponent : y_cont is multiplied with 10**decimal_exponent 
        In order ot get labels from range to [0,10**decimal_exponent]
        * value_type : type of supported value : vector or scalar.
    Output :
        * nb_classes : number of classes issued from conversion
        * y_label_encoded : a vector of labeled values.
    '''
    # 2 first digits are kept from first np.round, then last digit 
    # is removed.
    
    factor = 10**decimal_count
    y_cont = y_cont*factor
    y_label =np.around(y_cont, decimals=0)
    y_label = y_label.astype(int)
    if value_type == 'vector' :
        nb_classes =  y_label.max()-y_label.min()+1
        oneHotEncoder = OneHotEncoder(sparse=False)

        y_label_reshape = y_label.reshape(len(y_label), 1)
        y_label_encoded = oneHotEncoder.fit_transform(y_label_reshape).astype(int)

        return nb_classes, y_label_encoded
    elif value_type == 'scalar' :
        return y_label
    else : 
        print("\n*** ERROR : unknow type of value = {}".format(value_type))
        return None
        
#-------------------------------------------------------------------------------    
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------    
def df_weight_newFeature(df, ser_corr, list_feature_weight, newFeature):
    ''' Build a new feature based on previous features weighted with values from
    ser_corr.
        Input :
            * df : DataFrame containing features and new feature to be built.
            * ser_corr : Series containing correlation coefficient between feature and target.
            * list_feature_weight : the list of features from which weights are used to build newFeature.
            * newFeature :  name of the new feature issued from this process.
        Output :
            * DataFrame with newFeature as a new column.
    '''
    if 'new_feature' in df.columns :
        del(df['new_feature'])

    #-------------------------------------------------------------------------
    # Initialization of new_feature column
    #-------------------------------------------------------------------------
    df[newFeature] = np.zeros(df.shape[0])


    #list_feature_weight = [feature for feature in list_feature if feature not in ['rating','funny','wow','sad','likes','disagree']]
    for feature_weight in list_feature_weight :
        weight = ser_corr[feature_weight]
        print(feature_weight, weight)
        df['new_feature'] += df[feature_weight]*weight    

    #-------------------------------------------------------------------------
    # Normalization
    #-------------------------------------------------------------------------
    df[newFeature]=(df[newFeature]-df[newFeature].min())/(df[newFeature].max()-df[newFeature].min())
    return df    
#-------------------------------------------------------------------------------    
    
#-------------------------------------------------------------------------------    
#
#-------------------------------------------------------------------------------    
def print_col_stat(df, col, threshold, verbose=False):
    '''Computes percentage of values that are greater then given threshold.
    Values are issued from column col.
    col belongs to dataframe df.
    '''
    total = len(df)
    arr_index = np.where(df[col]>threshold)[0]
    ser_text_threshold = df[col][arr_index]
    len_text_threshold= len(ser_text_threshold)
    percent = len_text_threshold/total
    if verbose : 
        print("Number of texts where toxicity > "+str(threshold)+" : {}".format(len_text_threshold))
        print("Percentage of texts where toxicity > "+str(threshold)+" : {0:f}".format(percent))
    return percent    
        
#-------------------------------------------------------------------------------    
#
#-------------------------------------------------------------------------------    
def dataset_target_filter_threshold(threshold, direction, X, y):
    '''Returned a filtered dataset based on threshold for target array values.

    An array of indexes complaint with threshold criteria is extracted from target array.
    Filtered values are returned as an array that has been forged by sliding 
    target array values with array of indexes.
        
    Input : 
        *   threshold : threshold value used in filter operation.
                        when None, then X and y are returned with no change.
        *   direction : when >0 then filter operator is > 
                        when =0 then filter operator is ==
                        when <0 then filter operator is < 
        *   X   :   observations from dataset to be filtered
        *   y   :   targets from dataset to be filtered.
    Output :
        * X,y arrays with filtered values.
    '''
    if threshold is None :
        return X, y
        
    if direction > 0 :
        arr_index = np.where(y > threshold)[0]
    elif direction < 0 :
        arr_index = np.where(y < threshold)[0]
    else : 
        arr_index = np.where(y == threshold)[0]
    
    y_filtered = y[arr_index]
    X_filtered = X[arr_index]
    return X_filtered, y_filtered        
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------    
#
#-------------------------------------------------------------------------------    
def data_scale(X_train, X_test, scaler_name='Standard'):
    '''This function do scale train and test input data.
    Test dataset is scaled based on train scaler.
    Input : 
        X_train : train dataset 
        X_test  : test dtaset
        scaler_name : name for scaler used in module sklearn.preprocessing
    Output :
        scaler used to transform data, scaled X_train, scaled X_test.
    '''
    if scaler_name == 'MinMax' :
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaler_name == 'Standard' :
        #-----------------------------------------------------------------------
        # Mean is 0 and variance is scaled.
        #-----------------------------------------------------------------------
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    else :
        print("\n***ERROR : Scaler not supported : {}".format(scaler_name))
        return None, None

    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return scaler, X_train, X_test
#-------------------------------------------------------------------------------  

#-------------------------------------------------------------------------------    
#
#-------------------------------------------------------------------------------    
def remove_history_index(history, metric, metric_val, index_history = 0) :

    history_save= history.history.copy()

    if index_history is None :
        return history,history_save

    del(history.history['loss'][index_history])
    del(history.history['val_loss'][index_history])

    del(history.history[metric][index_history])
    del(history.history[metric_val][index_history])
    history.history = history_save.copy() 
    return history, history_save
#-------------------------------------------------------------------------------  

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def dict_2_df_word_counting(dico) :
    '''Convert a dictionary formated as {word:count_word} into dataframe with 
    index, and columns=[word,counting]
    
    Input : 
        * dico : dictionary formated as {word:count_word} where count_word is the number of words.
    Output : 
        Dataframe.
    '''

    df = pd.DataFrame.from_dict(dico, orient='index', columns=['counting'])
    df.reset_index(inplace=True)
    df.rename(columns={"index":'word'}, inplace=True)
    return df
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def tokenize_free_stop_words(corpus, slide_start, slide_length, min_length=2, max_length=15):
    '''Tokenize documents from a given corpus and remove stop words from any 
    tokenized document. 
        
    For memory efficiency, corpus is slided as bulk of documents, 
    ranged from [slide_start,slide_end].
    
    Then each document is converted into a list of tokens, using 
    gensim.utils.simple_preprocess function.
    
    A list of lists of tokenized sentences free of stowords is returned, all 
    tokens lowered.
    
    Input : 
        *   corpus : list of documents to be processed.
        *   slide_start : index in corpus where process starts.
        *   slide_length : number of documents in the bulk process.
        *   min_length : words under this size are ignored.
        *   max_length : words over thi size are ignored.

    Output :
        *   list of tokenized documents, each tokenized document is a 
            list of strings, free of stop-words.
        *   index of the last document processed from corpus.
    '''

    slide_end   = slide_start+slide_length
    #---------------------------------------------------------------------------
    # Get a sample from corpus by sliding
    #---------------------------------------------------------------------------
    corpus_slided = corpus[slide_start : slide_end]
    
    #---------------------------------------------------------------------------
    # In this step, each sentence is converted into a list of tokens with 
    # gensim.util package. In this process : 
    #   *   non alpfa-numeric words are removed.
    #   *   accent are removed
    #   *   
    #---------------------------------------------------------------------------
    llist_tokenized_sentence = \
    [gensim.utils.simple_preprocess(sentence, deacc=True, \
    min_len=min_length, max_len=max_length) for sentence in corpus_slided]

    #---------------------------------------------------------------------------
    # Remove stop words from each list of tokens.
    #---------------------------------------------------------------------------
    llist_tokenized_sentence_stop_words_sample = list()
    list_stop_word = gensim.parsing.preprocessing.STOPWORDS
    for list_tokenized_sentence in llist_tokenized_sentence :
        llist_tokenized_sentence_stop_words_sample.append([token.lower() for token in list_tokenized_sentence if token.lower() not in list_stop_word])

    #---------------------------------------------------------------------------
    # Remove empty lists 
    #---------------------------------------------------------------------------
    if False :
        llist_tokenized_sentence_stop_words_sample = \
        [list_tokenized_sentence_stop_words_sample for \
         list_tokenized_sentence_stop_words_sample in llist_tokenized_sentence_stop_words_sample \
         if len(list_tokenized_sentence_stop_words_sample)>0]
    
    return llist_tokenized_sentence_stop_words_sample, slide_end  
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------    
#
#-------------------------------------------------------------------------------    
def csr_2_sorted_dataframe(vectorizer, csr_matrix) :
    df_word_counting = pd.DataFrame()
    for count_list in range(0,len(csr_matrix.todense())):
        df_sorted = pd.DataFrame(csr_matrix.todense(), columns=vectorizer.get_feature_names())

        df_word_counting = pd.concat([df_word_counting,df_sorted], axis=0, ignore_index=True)
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------    
#
#-------------------------------------------------------------------------------    
def csr_2_sorted_dataframe_deprecated(vectorizer_, csr_matrix):
    '''This function convert a CSR matrix, issue from a vectorizer, 
    into a dense dataframe values.
    It computes a dataframe that contains frequency words from vocabulary.
    
    Vocabulary is issued from vectorizer_ operator.
    The returned dataframe is structured as following : 
        --> Columns : [word, counting]
        --> Index : index of each word from vocabulary.
    '''
    #------------------------------------------------------------------------------
    # Values from corpus are dataframe raws while dataframe columns are extracted 
    # from vectorizer features names. 
    #------------------------------------------------------------------------------
    df = pd.DataFrame(csr_matrix.toarray(), columns=vectorizer_.get_feature_names())

    #------------------------------------------------------------------------------
    # Cumulative sum is applied over each column (feature) of the dataframe.
    # This leads to counting each feature in the corpus.
    # This results in a Series where : 
    # --> columns is  vocabulary count
    # --> indexes are vocabulary word.
    #------------------------------------------------------------------------------
    ser_word_sum = df.sum(axis=0)
    
    #------------------------------------------------------------------------------
    # A dataframe is built from Series for having 
    # structure [word,counting] as columns.
    #------------------------------------------------------------------------------
    df_ = pd.DataFrame(ser_word_sum.values.reshape((1,-1)), columns=vectorizer_.get_feature_names())
    
    #------------------------------------------------------------------------------
    # Let's take transposee in order to have structured dataframe as following :
    # row : word from vocabulary
    # column : vocabulary count.
    # default column name is 0; it will be turned as 'counting'
    # Value from column '0' are sorted, ascending order.
    #------------------------------------------------------------------------------    
    df_sorted = df_.T.sort_values(0, axis=0)
    df_sorted.rename(columns={0:'counting'}, inplace=True)

    #------------------------------------------------------------------------------
    # reset_index lead to replace indexes, that are vocabulary word, with 
    # enumerated values.
    # This lead to a dataframe structured as following : 
    #
    #
    #         index   |  counting
    # index ----------+---------
    #  1      word1   |  count1
    #  2      word2   |  count2
    #        .......
    #  N      wordN   |  countN
    #
    #
    #------------------------------------------------------------------------------
    df_sorted.reset_index(inplace=True)

    #------------------------------------------------------------------------------
    # Final datafrmae is structured as following
    #
    #         word   |  counting
    # index ----------+---------
    #  1      word1   |  count1
    #  2      word2   |  count2
    #        .......
    #  N      wordN   |  countN
    #
    #
    #------------------------------------------------------------------------------
    df_sorted.rename(columns={'index':'word'}, inplace=True)   
    
    return df_sorted
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------    
#
#-------------------------------------------------------------------------------
def load_or_dump_tokenized_corpus(dump=False, content_type = 'bow', extension='',\
    X_train=None, y_train=None, X_test=None, y_test=None, tokenizer=None) :
    #extension='_full'
    #extension=''

    if dump :
        filename = './data/keras_tokenizer'+extension+'.dump'
        p5_util.object_dump(tokenizer,filename)

        filename = './data/X_train_'+content_type+extension+'.dump'
        p5_util.object_dump(X_train_encoded,filename)

        filename = './data/y_train_'+content_type+extension+'.dump'
        p5_util.object_dump(list_y_train_clean,filename)

        filename = './data/X_test_'+content_type+extension+'.dump'
        p5_util.object_dump(X_test_encoded,filename)

        filename = './data/y_test_'+content_type+extension+'.dump'
        p5_util.object_dump(list_y_test_clean,filename)

    else : 

        filename = './data/keras_tokenizer'+extension+'.dump'
        tokenizer = p5_util.object_load(filename)

        filename = './data/X_train_'+content_type+extension+'.dump'
        X_train = p5_util.object_load(filename)

        filename = './data/y_train_'+content_type+extension+'.dump'
        y_train = np.array(p5_util.object_load(filename))

        filename = './data/X_test_'+content_type+extension+'.dump'
        X_test = p5_util.object_load(filename)

        filename = './data/y_test_'+content_type+extension+'.dump'
        y_test = np.array(p5_util.object_load(filename))


        vocab_size = len(tokenizer.word_index) + 1
    
    embedding_dim = 100
    max_length = X_train.shape[1]
    print("\nX_train_encoded shape = {}".format(X_train.shape))
    print("X_test_encoded shape  = {}".format(X_test.shape))
    print("Y train shape= {}".format(y_train.shape))
    print("Y test shape= {}".format(y_test.shape))
    print("Vocabulary size= {}".format(len(tokenizer.word_index) + 1))
    return X_train, X_test, y_train, y_test, tokenizer
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def count_multi_label(matrix_label, axis=1) :
    '''Returns a dictionary structured as following : {label:count} where 
    count is the number of element from class label.
    Input :
        y : matrix (NxM) where either column or rows are one-hot encoded classes.
        axis : direction where to count. When value is 1, then counting takes 
                place along with columns.
    '''
    dict_label_count = dict()
    try :
        matrix_label.shape[1]
    except :
        # Classes are enumerated and ranged in a single column
        nb_classes = matrix_label.max()-matrix_label.min() + 1

        dict_label_count = {label:len(np.where(matrix_label==label)[0]) for label in range(0,nb_classes)}
        return dict_label_count
    if axis==1 :
        # Transpose vector in order to count along with axis=0
        matrix_label = matrix_label.T
    else :
        pass
    
    nb_classes = matrix_label.shape[0]
    list_count_label = list(np.sum(matrix_label, axis=1))
    dict_label_count = {label:count for (count, label) in [(list_count_label[label],label) for label in range(0,nb_classes)]}
    return dict_label_count   
#-------------------------------------------------------------------------------    


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_label_from_row(y_row, indicator=1) :
    '''Returns the column number from an array [1 x M] where indice is fixed to 1.
    This function is used when labels are one-hot encoded.
    The column number is the the label value.
    columns :  0   1   ...     p ......  M
             +---+---+------+----+-----+---+
      y_row: | 0 | 0 | ...  |  1 | ... | M |
             +---+---+------+----+-----+---+
                               ^
                               |
                               +----- : label is p
    Input : 
        * y_row : vector of encoded labels with M columns and 1 row 
        * indicator : when one-hot encoded, default value is 1.
    Output :
        * label : the position in vectors columns of indicator value 1.           
    '''
    label = np.where(y_row==indicator)[0][0]
    return label
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------------
def display_class_performance(class_estimator, Xeval, y_eval, \
                              is_binary=False, y_pred=None, nb_classes=None,\
                              title="Confusion matrix") :
    '''Compute predictions and display results issued from classification computation.
       These results are displayed under the followings forms : 
        --> Classes distribution, using seaborn librarie.
        --> Confusion matrix, text form, not normalized
        --> Normalised confusion matrix as a graphical array
        --> ROC curve in case of binary classification.

       Inputs :
            * class_estimator : estimator trained for classification
            * Xeval : dataset used for evalation 
            * y_eval : targets used for evaluation; matrix of shape [Rows,Cols]
                        where Cols is the number of classes, Rows the number of 
                        observations to be evaluated.
            * is_binary : binary clasification flag. When True, then ROC curve is displayed
            * y_pred : when None, predictions are computed from trained estimator.
                    Otherwise, this value is used for displaying results.
            * nb_classes : number of classes to be classified.
        Output :
            * Predictions computed with trained estimator.
    '''
    #---------------------------------------------------------------------------
    # If nb_classes provided  as function parameter is None, 
    # then, number of classes is defined by number of columns of target 
    # evaluation vector.
    #---------------------------------------------------------------------------
    if nb_classes is None :
        nb_classes = y_eval.shape[1]
    else :
        pass
    
    #--------------------------------------------------------------
    # prédire sur le jeu de test
    #--------------------------------------------------------------
    if y_pred is None :
        y_pred_label = class_estimator.predict(Xeval, verbose=1)
    else :
        y_pred_label = y_pred.copy()
        
    #-----------------------------------------------------------------
    # Display prediction classes ditribution
    #-----------------------------------------------------------------
    _=sns.kdeplot(y_pred_label, shade=True)

    #--------------------------------------------------------------
    # In case y_eval is a [NxM] matrix, where N is the number of 
    # observations and M the number of classes, then matrix is transformed 
    # into a vector thanks to function get_label_from_row.
    #--------------------------------------------------------------
    if len(y_eval.shape) >1 :
        nb_row = len(y_eval)
        y_test_label_row = [get_label_from_row(y_eval[row], indicator=1) for row in range(0,nb_row)]
    else :
        y_test_label_row = y_eval.copy()
        
    if len(y_pred.shape) >1 :
        nb_row = len(y_pred)
        y_pred_label_row = [get_label_from_row(y_pred[row], indicator=1) for row in range(0,nb_row)]
    else :
        y_pred_label_row = y_pred.copy()
        
        
    #--------------------------------------------------------------
    # Transform list into an array.
    #--------------------------------------------------------------
    y_test_label_row = np.array(y_test_label_row)
    y_pred_label_row = np.array(y_pred_label_row)

    #--------------------------------------------------------------
    # Computes classes as a list
    #--------------------------------------------------------------
    list_classes = [label for label in range(0,nb_classes)]
    print(list_classes)

    #--------------------------------------------------------------
    # Compute confusion matrix
    #--------------------------------------------------------------
    print("\nConfusion matrix for all classes : ")
    print()
    print(metrics.confusion_matrix(y_test_label_row, y_pred_label_row, labels=list_classes))
    print()

    cm = confusion_matrix(y_test_label_row, y_pred_label_row)
    p4_util.plot_confusion_matrix(cm, list_classes,
                              normalize=True,
                              title=title,
                              cmap=plt.cm.Blues)

    p4_util.plot_confusion_matrix(cm, list_classes,
                              normalize=False,
                              title=title,
                              cmap=plt.cm.Blues)
    #--------------------------------------------------------------
    # ROC curve for binary classification
    #--------------------------------------------------------------
    if is_binary :
        fpr, tpr, thr = metrics.roc_curve(y_test_label_row, y_pred_label_row)
        print("\nTP rate = {}".format(tpr))
        print("FP rate = {}".format(fpr))
        # calculer l'aire sous la courbe ROC
        auc = metrics.auc(fpr, tpr)
        print("AUC = {}".format(auc))

        # créer une figure
        fig = plt.figure(figsize=(6, 6))

        # afficher la courbe ROC
        plt.plot(fpr, tpr, '-', lw=2, label='11 classes AUC=%.2f' % auc)
    
    return y_pred_label_row
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def multivalue2_binary(y, threshold, direction, value_true=1, value_false=0) :
    '''
        Input :
            * threshold : value used in test, e.g. y[i] < threshold
            * direction : encoded test operator  : 
                   +-----------+---------------+
                   | direction | test operator |
                   +-----------+---------------+
                   |    -1     |      <        |
                   +-----------+---------------+
                   |    0      |      ==       |
                   +-----------+---------------+
                   |    +1     |      >=       |
                   +-----------+---------------+
            * value_true : assigned value if expresion is True 
            * value_false : assigned value if expresion is False 
    '''
    if direction == -1 :
        y_bin = np.where(y<threshold,value_true,value_false)
    elif direction == 0:
        y_bin = np.where(y==threshold,value_true,value_false)
    else :
         y_bin = np.where(y>=threshold,value_true,value_false)
    return y_bin
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_vector_from_classweight(y,class_weight) :

    range_label = range(0,len(class_weight))
    y_weight = np.array(y, float)
    for label, weight in zip(range_label,class_weight) :
        arr_index = np.where(y==label)
        print("Class : {} Weight={}".format(label,weight))
        for index in arr_index :
            y_weight[index] = weight
    print("")
    return y_weight
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_classweight_vector(y_vect, mode='balanced', class_weights=None) :

    if class_weights is None :
        class_weights = compute_class_weight(mode, np.unique(y_vect), y_vect)
    else : 
        pass

    print("\nClasses weights = {}".format(class_weights))
    print("\n")

    range_label = range(0,len(class_weights))
    y_weight = np.array(y_vect, float)
    for label, weight in zip(range_label,class_weights) :
        arr_index = np.where(y_vect==label)
        print("Class : {} Weight={}".format(label,weight))
        for index in arr_index :
            y_weight[index] = weight
    print("")
    return y_weight, class_weights
#-------------------------------------------------------------------------------   

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_df_wordCounting_from_vectorizer(csr, countVectorizer) :   
    '''Returns a dataframe with 2 columns : 
       --> word : word in csr matrix 
       --> counting : number of occurencies of a word in csr 
    Input :
        * csr : CSR matrix issued from countVectorizer.
        * countVectorizer :  a vectorizer used to count word in corpus. 
    '''
    #-------------------------------------------------------------------------------------
    # csr.sum(axis=0) : cumulative of all words count; lead to a matrix [1xcolumns] where 
    # columns is the number of features from vectorizer.
    #-------------------------------------------------------------------------------------
    df_wordCounting = pd.DataFrame(csr.sum(axis=0), \
    columns=countVectorizer.get_feature_names())
    
    df_wordCounting = df_wordCounting.T.sort_values(0, axis=0)

    df_wordCounting.reset_index(inplace=True)
    df_wordCounting.rename(columns={0:'counting'}, inplace=True)
    df_wordCounting.rename(columns={'index':'word'}, inplace=True)
    return df_wordCounting 
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_wordCounting_free_stopword(df_word, column_name='word', is_verbose=False) :
    '''Remove rows from dataframe df_wc having stop words.
    The gensim stop words list is used.
    Input :
        *   df_word : dataframe with at least 1 column containing words.
            -> word column contain words issued from a corpus vactorization with 
            a count vectorizer operator.
            -> column_name : name for the column from  df_word containing list of 
            words.
    Output : a dataframe free of rows containing a stop word in column_name.
    '''
    if column_name not in df_word.columns :
        print("\n***ERROR : no column with name = ".format(column_name))
        return df_word
    else :
        pass
    if is_verbose :
        print("")
        print("Shape of incoming dataframe= {}".format(df_word.shape))
    list_stop_word = gensim.parsing.preprocessing.STOPWORDS

    list_stop_word_corpus = [ stop_word for stop_word in list_stop_word if stop_word in df_word['word'].values]
    list_index_stop_word = [index_stop_word for index_stop_word in [df_word[df_word['word']==stop_word_corpus].index[0] for stop_word_corpus in list_stop_word_corpus]]

    df_word.drop(list_index_stop_word, inplace=True)
    if is_verbose :
        print("Shape of outgoing dataframe= {}".format(df_word.shape))
    return df_word
#------------------------------------------------------------------------------- 

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def corpus_tokenize(corpus, target=None, slide_length = 50000,\
    min_token_len=2, max_token_len=15, min_doc_len=5, max_doc_len=100) :
        
    '''Tokenize a corpus of documents and remove stopwords in any document from 
    corpus.
    
    For memory efficiency, this process is splitted into n_iteration steps.
    Number of iteration, say n_iteration, is calculated dividing then 
    corpus length by  slide_length.
    
    For each iteration :
        -> Corpus rows are slided with slide length = slide_length.
        -> A list of tokens free of stop words is built 
        -> This list of tokens is aggregated with the all lists from previous 
        steps.

    Input :
        * corpus : a list of documents, each document is a string, composed of 
        words.
        * slide_length : bulk of texts tokenized at each iteration.
        * min_token_len : tokens under this length are ignored from texts.
        * max_token_len : tokens over this length are ignored from texts.
        * min_doc_len : miminum number of words in sentence for sentence to be kept 
        in corpus.
        * max_doc_len : maximum number of words in sentence for sentence to be kept 
        in corpus.
    Output : 
        * pandas DataFrame with 3 columns : 
            column 1 : count : number of tokens into tokenized sentence 
            column 2 : y : target given as parameter
            column 3 : text : tokenized texts as a string, free of stop words.
    '''
    
    # Remove words punctions from corpus 
    corpus = [text.translate(str.maketrans('', '', string.punctuation)) for text in corpus]
    
    #----------------------------------------
    # Define number of steps it will take 
    #----------------------------------------
    n_iteration = len(corpus)//slide_length
    tail = len(corpus)%slide_length


    llist_tokenized_sentence_free_stop_words = list()
    slide_start_ = 0
    slide_end_ = 0
    print()
    for iteration in range(0, n_iteration) :
        start_time = time.time()
        #-----------------------------------------------------------------------
        # Remove stop words from corpus, 
        # sliding between [slide_start_, slide_start_+sample_size]
        #-----------------------------------------------------------------------
        list_tokenized_sentence_stop_words_sample, slide_end_ = \
        tokenize_free_stop_words(corpus, slide_start_, slide_length, \
        min_length=min_token_len, max_length=max_token_len)

        #-----------------------------------------------------------------------
        # Result list is aggregated
        #-----------------------------------------------------------------------
        llist_tokenized_sentence_free_stop_words += \
        list_tokenized_sentence_stop_words_sample

        #-----------------------------------------------------------------------
        # Some monitoring, making sure all going well....
        #-----------------------------------------------------------------------
        delta_time  = round(time.time()-start_time, 0)
        print("Elapsed time for iteration "+str(iteration)+" = {} / (start,end)=({},{})".format(delta_time, slide_start_, slide_end_), end='\r')

        #-----------------------------------------------------------------------
        # Shift start value of sliding
        #-----------------------------------------------------------------------
        slide_start_ = slide_end_

    #-----------------------------------------------------------
    #    Proceed to aggregation of remained samples.
    #-----------------------------------------------------------
    list_tokenized_sentence_stop_words_sample, slide_end_ = \
    tokenize_free_stop_words(corpus, slide_start_, tail, \
    min_length=min_token_len, max_length=max_token_len)
    
    llist_tokenized_sentence_free_stop_words += \
    list_tokenized_sentence_stop_words_sample

    print(" ")
    print("Number of prepared texts= {}".format(len(llist_tokenized_sentence_free_stop_words)))
    if False :
        p5_util.object_dump(llist_tokenized_sentence_free_stop_words,"./data/llist_tokenized_sentence_free_stop_words.dump")

    #---------------------------------------------------------------------------
    #    Rebuild sentences free of stop words from list of tokens
    #---------------------------------------------------------------------------
    if False :
        list_sentence_free_stop_words = [ " ".join(list_tokenized_sentence_free_stop_words) for\
                                         list_tokenized_sentence_free_stop_words in \
                                         llist_tokenized_sentence_free_stop_words]

    ser_corpus_free_stopword = pd.Series(llist_tokenized_sentence_free_stop_words)
    
    
    #---------------------------------------------------------------------------
    #    Count words into any sentence and store it into Series
    #---------------------------------------------------------------------------
    ser = ser_corpus_free_stopword.apply(lambda x : len(x)) 
    #return ser, target, ser_corpus_free_stopword

    #---------------------------------------------------------------------------
    #    Build dataframe with 3 columns : count, sentence,target
    #---------------------------------------------------------------------------
    df = pd.DataFrame({'count':ser.tolist(), 'target':target,\
        'text':ser_corpus_free_stopword.values})            

    
    #---------------------------------------------------------------------------
    #    Remove rows from dataframe outside range [min_doc_len, max_doc_len]
    #---------------------------------------------------------------------------
    index_drop = df[df['count']<min_doc_len].index
    df.drop(index=index_drop, inplace=True)

    index_drop = df[df['count']>max_doc_len].index
    df.drop(index=index_drop, inplace=True)

    
    return df   
#------------------------------------------------------------------------------- 

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def tfidf_vectorization(ser_X, vectorizer = None,   
                    ngram_range=(1, 1)\
                    , max_features=500):
    print("")
    if vectorizer is None :
        vectorizer = \
        make_pipeline(CountVectorizer(ngram_range=ngram_range, max_features=max_features)\
        ,TfidfTransformer())
        
        print("Fit vectorizer pipeline...")
        vectorizer.fit(ser_X)

    else :
        pass
    
    print("Transform data with TF-IDF vectorizer...")
    csr_matrix = vectorizer.transform(ser_X)
    

    return vectorizer, csr_matrix   
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def make_partition(data, label, partition_size, data_type="train", \
        data_format='ndarray', is_debug=False,dict_partition=None, dict_label=None ):
    '''Build a partition of data.
    A partition is a set of files each one containing parts of data.
    Supported format of data : 'csr', 'ndarray'
    
    Input : 
        *   data : array to be recorded
        *   label : array of values related to data 
        *   partition_size : number of partitions
        *   data_type : supported values : either train or validation; this value 
        is used for file names.
        *   is_debug : flag value; when True, then verbose is activated. Dictionaries 
            are built but no file is created.
        *   dict_partition : structured as {file_id: filename}; when this value 
        is not None, then dictionary contains information from previous function 
        calls. New informations are added in this function call.
        *   dict_label : structured as {filename: label_values}; when this value 
        is not None, then dictionary contains information from previous function 
        calls. New informations are added in this function call.
        
    Output :
        *   2 dictionaries, one containing partitions structured as {file_id: filename}
        the other structured as {filename: label_values} where label_values are
        values from label array from partition.
        	
    '''
    print("Building "+str(data_type)+" partition...\n")
    
    #---------------------------------------------------------------------------
    # Number of files to store data into partions files.
    #---------------------------------------------------------------------------
    partition_size = min(data.shape[0], partition_size)
    file_count= data.shape[0]//partition_size
    
    if 0 == file_count :
        print("\n*** ERROR : Files count = {} / Data rows={} / partition size= {}".\
        format(file_count, data.shape[0],partition_size))
        print("*** ERROR : partition_size= {} > Number of rows= {}".format(partition_size, data.shape[0]))
        return None, None

    partition_tail = data.shape[0]%partition_size
    partition_start = 0
    
    max_iter=10 # Used for debug
    
    if (dict_partition is not None) and (dict_label is not None):
        pass
    else :
        dict_partition = dict()
        dict_label = dict()

    file_count_start = len(dict_partition)
    if file_count_start != len(dict_label) :
        print("\n*** ERROR : make_partition() : no same number of values for data and labels!")
        return None, None
        
    file_count_end = file_count_start + file_count
    
    for file_id in range(file_count_start,file_count_end) :
        partition_end = partition_start + partition_size
        if data_format == 'csr' :
            X = data[partition_start:partition_end].toarray()
        elif data_format == 'ndarray' :
            X = data[partition_start:partition_end]
        else : 
            print("\n***ERROR : data format not supported : {}".format(data_format))
            return None, None

        filename = "./data/"+str(data_type)+"_X_"+str(file_id)+".npy"
        if not is_debug :
            np.save(filename,X)
        dict_partition[file_id]=filename
        dict_label[filename] = label[partition_start:partition_end]
        print("Partition file : {} / {} Done!".format(file_id-file_count_start,file_count), end='\r')
        partition_start = partition_end
        
    if partition_tail >0 :
        file_id += 1
        partition_end = partition_start + partition_tail
        if data_format == 'csr' :
            X = data[partition_start:partition_end].toarray()
        elif data_format == 'ndarray' :
            X = data[partition_start:partition_end]
        
        filename = "./data/"+str(data_type)+"_X_"+str(file_id)+".npy"
        if not is_debug :
            np.save(filename,X)
        dict_partition[file_id]=filename
        dict_label[filename] = label[partition_start:partition_end]
        print("Partition file : {} / {} Done!".format(file_id-file_count_start,file_count), end='\r')
    return dict_partition, dict_label
#-------------------------------------------------------------------------------
    

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def convert_ser2arr(ser) :
    '''Convert values in pd.Series into an array, where values in 
    pd.Series are vectors.
    Array returned is shaped as [N,P], where : 
        *   N is the number of records of ser 
        *   P is the size of vectors 
    '''
    #print("convert_ser2arr: ser= {}".format(ser))
    if ser is None :
        return None
    list_arr = [ser[index] for index in ser.index]
    return np.asarray(list_arr)

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def glove_build_dict_word_vector(glove_filename):
    '''Builds a Glove dictionary structured as : {word:coef_vector}
    where coef_vector are vectors with dimension N.
    
    Input : 
        * glove_filename : the Glove file name
    Output :
        * Glove dictionary 
        * vectors dimension
    '''
    dict_embeddings_index = dict()
    try:
        with open(glove_filename, 'rb') as (dataFile):
            for line in dataFile:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                dict_embeddings_index[word] = coefs

    except FileNotFoundError as fileNotFoundError:
        print("\n*** ERROR : {} : ".format(fileNotFoundError))
    except ModuleNotFoundError as moduleNotFoundError:
        print("\n*** ERROR : {} : ".format(moduleNotFoundError))

    # Get vectors dimension
    key = list(dict_embeddings_index.keys())[0]

    embedding_dim=dict_embeddings_index[key].shape[0]
    return dict_embeddings_index, embedding_dim
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def glove_weight_matrix_build(glove_filename, tokenizer, tokenizer_type='keras') :
    '''Builds matrix of glove weights for any word in vocabulary issued from 
    tokenizer.
    
    First, Glove dictionary is extracted from Glove file name.
    Then matrix of weights is built using coefficients in glove dictionary.
    
    '''
    weight_matrix = None                              
    dict_word_vector, vector_dimension = \
    glove_build_dict_word_vector(glove_filename)
    
    if tokenizer_type == 'keras' :
        vocab_size = len(tokenizer.word_index) + 1
        weight_matrix = np.zeros((vocab_size, vector_dimension))
        #-----------------------------------------------------------------------
        # Retrieve each word from vocabulary issued from keras tokenizer
        # and get associated vector from dict_word_vector issueed from Glove
        #-----------------------------------------------------------------------
        for word, index in tokenizer.word_index.items():
            weight_vector = dict_word_vector.get(word)
            if weight_vector is not None:
                weight_matrix[index] = weight_vector
    else : 
        print("\n***ERROR : tokenizer type not supported: {}".format(tokenizer_type))
        return None

    return weight_matrix
#------------------------------------------------------------------------------- 

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def vectorValue2matrixLabel(vector_value,decimal_count = 1) :
    '''Transforms a vector of coninuous values into a matrix of integers labels.
    Fixing decimal_count=1 leads to have labels ranged from 0 to 10**1 ==> [0, 10]
    In a such case, the returned matrix will have 10 columns, 
    one column for each label.
    
    Inputs :
        *   vector_value : vector of continuous values.
        *   decimal_count : number of decimals to take into account from 
        vector values when values are converted into labels.
    Output :
        *   A matrix (N,P) where 
            --> N is the input vector dimention.
            --> P is the number of differents labels.
    '''    

    nb_classes, matrixLabel = p9_util.y_cont_2_label(vector_value, decimal_count=decimal_count)
    return matrixLabel
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def matrixLabel2vectorBinaryLabel(matrixLabel,tuple_transformation=(0,1)) :
    '''Transforms a matrix of labels (=integer values) into a vector of labels.
    
    Inputs :
        *   matrixLabel : matrix of labels
        *   tuple_transformation : tuple object in which :
            --> tuple_transformation[0] is the threshold value from which 
                label will be set to 0
            --> tuple_transformation[1] is the direction of evaluation for 
                binarization defining the operand as detailed below: 
                   +---------------------+---------+ 
                   |Evaluation Direction | Operand |
                   +---------------------+---------+ 
                   |          -1         |   <=    |
                   +---------------------+---------+ 
                   |           0         |   ==    |
                   +---------------------+---------+ 
                   |           1         |   >=    |
                   +---------------------+---------+ 
    '''
    #--------------------------------------------------------------------------
    # Vectors binarization.
    # Threshold value will select vector values to 0 and vectors values to 1.
    #--------------------------------------------------------------------------
    threshold = tuple_transformation[0]
    direction = tuple_transformation[1]
    
    print(threshold, direction)
    
    vectorBinaryLabel = p9_util.multivalue2_binary(matrixLabel, threshold, \
    direction)
    return vectorBinaryLabel
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def list_word_most_frequent(corpus, nb_most_frequent=100, is_verbose=False) :
    ''' Returns the list of most frequent words from a given corpus.

    Empty list is returned when nb_most_frequent is 0.
    
    '''
    list_frequent_word = list()
    
    if 0 < nb_most_frequent :    
        vectorizer = CountVectorizer()
        vectorized_corpus = vectorizer.fit_transform(corpus)
        df_wordCounting = p9_util.get_df_wordCounting_from_vectorizer(vectorized_corpus, vectorizer)
        df_wordCounting = p9_util.df_wordCounting_free_stopword(df_wordCounting, is_verbose = is_verbose)



        ser_item_name = df_wordCounting.word
        ser_item_count = df_wordCounting.counting
        df_item_dict={item:count for item, count in zip(ser_item_name, ser_item_count)}

        list_item_sorted = sorted(df_item_dict.items(), key=lambda x: x[1], reverse=True)

        dict_item_sorted = dict()
        for tuple_value in list_item_sorted :
            dict_item_sorted[tuple_value[0]] = tuple_value[1]
        len(dict_item_sorted)

        index = 0
        for word in dict_item_sorted.keys() :
            if index < nb_most_frequent : 
                list_frequent_word.append(word)
            else :
                break
            index += 1
        else : 
            pass
    return list_frequent_word
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def matrix_zeropadd_row(input_matrix, max_row, dict_vector_coeff=None) :
    '''Truncate or padd a matrix with nb_row.
    Result is a returned matrix with max_row.
    If nb_row < max_row then matrix is padded with 0.
    Otherwise, matrix rows are truncated.
    '''
    nb_col = input_matrix.shape[1]
    nb_row = input_matrix.shape[0]
    #---------------------------------------------------------------------------
    # Truncate or padd matrix
    #---------------------------------------------------------------------------
    if nb_row < max_row :
        #-----------------------------------------------------------------------
        # Padding of input matrix...
        #-----------------------------------------------------------------------
        input_matrix_padded = np.zeros((max_row,nb_col))
        input_matrix_padded[:nb_row] = input_matrix
    else :
        #-----------------------------------------------------------------------
        # Truncation of input matrix...
        # List of most max_row important tokens are selected depending of 
        # dictionary of coefficients given as parameter. 
        #-----------------------------------------------------------------------
        if dict_vector_coeff is None :
            return input_matrix[:max_row]
        else :
            print("\n***ERROR : matrix_zeropadd_row() : case to be completed!")
            return None
    return input_matrix_padded
#-------------------------------------------------------------------------------    


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def multiplexer(x_input, dim_transpose, nb_dim = -1) :
    # Apply multiplexing
    x_mux = x_input.transpose(dim_transpose)

    # Checksum has tto be 0.0 to multiplexer be validated
    if nb_dim == -1 :
        return x_mux
    else : 
        return x_mux[:nb_dim]
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def checksum(xm,x, pos_dim, pos_item) :
    checksum = 0.
    for dim in range(xm.shape[pos_dim]) :
        for item in range(xm.shape[pos_item]) :
            checksum += (xm[dim,item,:]-x[item,:,dim]).max()
    print(checksum)
    if checksum == 0.0 :
        return True
    return False
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def datapreparator_v2_targetUpdate2Label(dataPreparator_v2):
    
    if (0 == len(dataPreparator_v2.list_df_data_file)) \
    or (dataPreparator_v2.list_df_data_file is None) :
        y = dataPreparator_v2.y
    else :
        total_row = dataPreparator_v2.total_row
        y = np.zeros(total_row)
        start_row = 0
        for df_data_file in dataPreparator_v2.list_df_data_file :
            df_data = p5_util.object_load(df_data_file)

            end_row = start_row + len(df_data)
            y[start_row:end_row] = df_data.target.values
            start_row = end_row

    y_label = dataPreparator_v2.vectorValue2BinaryvectorLabel(vector_value=y)

    #--------------------------------------------------------------------------------
    # tuple_transformation : 
    #   * threshold : 0.
    #   * Direction : >=
    # This leads to : if y > 0.0 
    #                    then value is 1
    #                 else :
    #                    value is 0
    #--------------------------------------------------------------------------------
    y_label = p9_util.matrixLabel2vectorBinaryLabel(y_label,tuple_transformation=(0.0,1)) 
    if False :
        ser = pd.Series(y_label, index = dataPreparator_v2.df_data.index, dtype=int)
    else : 
        ser = pd.Series(y_label, dtype=int)
    
    column_name = dataPreparator_v2.COLUMN_NAME_TARGET
    dataPreparator_v2.df_data[column_name] = ser.copy()

    return dataPreparator_v2
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def X_y_balance_binary(filename, n_sample_per_class, threshold):
    X, y = p5_util.object_load(filename)
    
    tmp = filename.split('X_y_')[1]
    data_type, extension = tmp.split('.')

    
    root_filename = filename.split('.'+str(extension))[0]
    filename_balanced = './data/X_y_balanced_'+str(data_type)+'.'+str(extension)
    
    if 0 >= n_sample_per_class :
        ser = y[y.values >= threshold]
    else :
        ser_0 = y[y.values <= threshold]
        ser_1 = y[y.values  > threshold]
        
        
        ser_0 = ser_0.sample(n_sample_per_class)
        ser_1 = ser_1.sample(n_sample_per_class)

        
        ser = pd.concat([ser_0, ser_1])
    
    len_ser = len(ser)
    ser = ser.sample(n=len_ser)

    arr_index = np.array(ser.index)
    p5_util.object_dump((X[arr_index], y[arr_index]),filename_balanced, is_verbose=True)
    return filename_balanced
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def keras_model_load(filename) :
    model = None
    if not os.path.exists(filename)  :
        print("\n*** ERROR : unknown file = {}".format(filename))
    else :
        print("\nLoading model...")
        model = keras.models.load_model(filename)
    return model
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def convert_vectorFloat_2_binaryLabel(vector_value, threshold=0, direction = 1, \
decimal_count = 1) :
    '''Convert a vector of continuous values into a vector of binaries values.
    
        Input :
            * vector_value : vector of float values
            * threshold : value above which label is converted to 1.
            * direction : 1 for >, 0 for ==, -1 for < 
            * decimal_count : number of decimal to take into account from float value.
        Output :
            * An array of labeled values.
    '''
    
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
    factor = 10**decimal_count
    threshold *= factor
    vector_label_bin = p9_util.multivalue2_binary(vector_label, threshold, direction)
    return vector_label_bin
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_column_cont2labelRange(df, column_name, left_limit, right_limit, label_range, \
                              t_left_operator=(1,1), t_right_operator=(-1,0), \
                              is_verbose=True) :
    '''Change values inside an interval limited by 2 limit values, into a label value.
    
    Input : 
        * df : Dataframe in which values will be labelized
        * column_name : dataframe colum name to be labelized
        * left_limit, right_limit : interval of values to be labelized
        * t_operator_left, t_operator_right : tuple encoded comparaison operators :
            
             +------------+-----------+----------+
             |  1st code  | 2nd code  | operator |       
             +------------+-----------+----------+
             |     1      |     1     |     >    |
             +------------+-----------+----------+
             |     1      |     0     |     >=   |
             +------------+-----------+----------+
             |    -1      |     1     |     <>   |
             |     1      |    -1     |     <>   |
             +------------+-----------+----------+
             |     0      |     0     |     ==   |
             +------------+-----------+----------+
             |    -1      |     0     |     <=   |
             +------------+-----------+----------+
             |    -1      |    -1     |     <    |
             +------------+-----------+----------+
             
             Default interval values limits: ]left_limit, right_limit]

        * label_range :  value assigned to range of values included inside left_limit, right_limit
        
     Output :
         Dataframe with modified column values.
    '''
    if is_verbose:
        print("")
        print("Left operator= {} Label = {} Right operator={}".format(t_left_operator, label_range, t_right_operator))
    if t_left_operator == (1,1) :
        #-----------------------------------------------------------------------
        # 1st operator code : > 
        # 2nd operator code : >
        # Resulting operator : > 
        # Range value : ]left_limit, right_limit ?
        #-----------------------------------------------------------------------
        arr_index = df[left_limit < df[column_name]].index
        df_left = df.loc[arr_index]
        if t_right_operator == (-1,-1) :
            #-------------------------------------------------------------------
            # 1st operator code : <
            # 2nd operator code : <
            # Resulting operator : > 
            # Range value : ]left_limit, right_limit [
            #-------------------------------------------------------------------
            arr_index = df_left[df_left[column_name]<right_limit].index
        elif t_right_operator == (-1,0) :
            #-------------------------------------------------------------------
            # 1st operator code : <
            # 2nd operator code : =
            # Resulting operator : >= 
            # Range value : ]left_limit, right_limit ]
            #-------------------------------------------------------------------
            arr_index = df_left[df_left[column_name]<=right_limit].index
        elif (t_right_operator == (-1,1)) or (t_right_operator == (1,-1)) :
            #-------------------------------------------------------------------
            # 1st operator code : <
            # 2nd operator code : >
            # Resulting operator : <> 
            # Range value : ]left_limit, right_limit ]
            #-------------------------------------------------------------------
            print("\n*** ERROR on right operator code : {}".format(t_right_operator))
            return df

    elif t_left_operator == (1,0) :
        #-----------------------------------------------------------------------
        # 1st operator code : > 
        # 2nd operator code : =
        # Resulting operator : >= 
        # Range value : [left_limit, right_limit ?
        #-----------------------------------------------------------------------
        arr_index = df[left_limit <= df[column_name]].index
        df_left = df.loc[arr_index]
        if t_right_operator == (-1,-1) :
            #-------------------------------------------------------------------
            # 1st operator code : <
            # 2nd operator code : <
            # Resulting operator : > 
            # Range value : [left_limit, right_limit [
            #-------------------------------------------------------------------
            arr_index = df_left[df_left[column_name]<right_limit].index
        elif t_right_operator == (-1,0) :
            #-------------------------------------------------------------------
            # 1st operator code : <
            # 2nd operator code : =
            # Resulting operator : >= 
            # Range value : [left_limit, right_limit ]
            #-------------------------------------------------------------------
            arr_index = df_left[df_left[column_name]<=right_limit].index
        elif (t_right_operator == (-1,1)) or (t_right_operator == (1,-1)) :
            #-------------------------------------------------------------------
            # 1st operator code : <
            # 2nd operator code : >
            # Resulting operator : <> 
            # Range value : ]left_limit, right_limit ]
            #-------------------------------------------------------------------
            print("\n*** ERROR on right operator code : {}".format(t_right_operator))
            return df
    elif t_left_operator == (0,0) :
        #-----------------------------------------------------------------------
        # 1st operator code : =
        # 2nd operator code : =
        # Resulting operator : == 
        # Range value : [left_limit, right_limit ?
        #-----------------------------------------------------------------------
        arr_index = df[left_limit == df[column_name]].index
    else :
        print("\n*** ERROR : Wrong encoded left operator (2nd code): {}".format(t_left_operator))
        return df
    df[column_name].loc[arr_index]=label_range
    if is_verbose :
        print("Number of modified "+str(column_name)+"  with label= {} : {}".format(label_range, len(df[column_name].loc[arr_index])))
    return df    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_balanced_groups(df, col, list_label_value, min_group_size, is_verbose=True):
    df_concat = pd.DataFrame()
    for label_value in list_label_value :
        #----------------------------------------------------------------------
        # select all items belonging to group label_value
        #----------------------------------------------------------------------
        tag_name = 'level_'+str(label_value)
        arr_group_index = df[df[col]==tag_name].index
        df_group = df.loc[arr_group_index]

        if is_verbose :
            print("\nSize of group= {} : {}".format(tag_name, len(arr_group_index)))

        if 0 < len(arr_group_index) and len(arr_group_index) >= min_group_size:
            #----------------------------------------------------------------------
            # Sample items with safe tags with sae size then sample with insult tag
            #----------------------------------------------------------------------
            df_group = df_group.sample(min_group_size)
            if is_verbose:
                print("Resampled size of group= {} : {}".format(tag_name, len(df_group)))


            #----------------------------------------------------------------------
            # Concatenate safe group with insult group.
            #----------------------------------------------------------------------
            df_concat = pd.concat([df_concat, df_group])
            if is_verbose :
                print("Number of items in concatened groups= {}".format(len(df_concat)))
        else :
            print("\n*** WARN: group= {} : size= {} <= {} ".format(tag_name, len(arr_group_index), min_group_size))
    df_concat = df_concat.sample(len(df_concat))
    return df_concat    
#-------------------------------------------------------------------------------    

