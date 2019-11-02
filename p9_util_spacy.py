#!/usr/bin/python3.6
#-*- coding: utf-8 -*-

'''This file contains all spacy utils functions.
'''
import numpy as np
import pandas as pd
import spacy
import p9_util
import p9_util_spacy

#SPACY_NLP_SM = spacy.load('en_core_web_sm')
# en_core_web_md : written text (blogs, news, comments)
#                  685k keys, 20k unique vectors (300 dimensions)
#
# en_core_web_lg : written text (blogs, news, comments)
#                  685k keys, 685k unique vectors (300 dimensions)
#
# en_vectors_web_lg : written text (blogs, news, comments)
#                  1.1m keys, 1.1m unique vectors (300 dimensions)
#
#
 
#model_lang = 'en_vectors_web_lg'
#model_lang = 'en_core_web_md'
model_lang = 'en_core_web_lg'
SPACY_LANGUAGE_MODEL = spacy.load(model_lang)

DDICT_SPACY_LANG_ENTITY_LABEL = {'en':{'PERSON':"people",\
'NORP':'religious',\
'FAC':'building',\
'ORG':'agency',\
'GPE':'country',\
'LOC':'location',\
'PRODUCT':'product',\
'EVENT':'event',\
'WORK_OF_ART':'title',\
'LAW':'law',\
'LANGUAGE':'language',\
'DATE':'period',\
'TIME':'time',\
'PERCENT':'percent',\
'MONEY' : 'money',\
'QUANTITY':'measurement',\
'ORDINAL':'ordinal',\
'CARDINAL':'numeral',\
}}

COLUMN_NAME_DOC    = p9_util.COLUMN_NAME_DOC
COLUMN_NAME_TARGET = p9_util.COLUMN_NAME_TARGET
COLUMN_NAME_TOKEN  = p9_util.COLUMN_NAME_TOKEN
COLUMN_NAME_COUNT  = p9_util.COLUMN_NAME_COUNT
COLUMN_NAME_VECTOR = p9_util.COLUMN_NAME_VECTOR

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_is_token_valid(spacy_token, \
                        list_invalid_pos_tags = ['PROP','PROPN', 'PUNCT', 'NUM'],\
                        list_valid_token =['not'],\
                        min_token_length=2,
                        max_token_length=15,
                        is_not_like_email=True,
                        is_not_like_url=True,
                        is_verbose = False,
                        ) :
    '''Returns token validity flag (True or False) using Spacy librarie.
    Invalid spacy token are token : 
        with followings tags : 
            * PROP, PROPN, PUNCT, NUM
        For which :
            * length <  min_token_length
            * length >  max_token_length
        That belong to :
            * list of spacy stop-words
        Are like :
            * white spaces
            * punctuation such as brackets, quotes...
    If is_not_like_email flag is True, then token is checked against beeing email.
    If is_not_like_url flag is True, then token is checked against beeing URL.
    if token belongs to list_valid_token givena s parameter, then function 
    returns True.
    '''

    is_valid = True
    
    if spacy_token.text in  list_valid_token :
        return is_valid
    
    if spacy_token.is_stop :
        if is_verbose : print("stop")
        is_valid = False
    elif spacy_token.pos_ in list_invalid_pos_tags :    
        if is_verbose : print("pos = {}".format(spacy_token.pos_))
        is_valid = False
    elif len(spacy_token.text) < min_token_length :
        if is_verbose : print("min_token_length")
        is_valid = False
    elif len(spacy_token.text) > max_token_length  and (0<max_token_length):
        if is_verbose : print("max_token_length")
        is_valid = False
    elif spacy_token.is_space :
        if is_verbose : print("is_space")
        is_valid = False
    elif spacy_token.is_punct :
        if is_verbose : print("is_punct")
        is_valid = False
    elif spacy_token.is_digit :
        if is_verbose : print("is_digit")
        is_valid = False
    elif spacy_token.is_bracket :
        if is_verbose : print("is_bracket")
        is_valid = False
    elif spacy_token.is_quote :
        if is_verbose : print("is_quote")
        is_valid = False
    else :
        if is_not_like_email :
            if spacy_token.like_email :   
                if is_verbose : print("like_email")
                is_valid = False
            else :
                pass
        
        if is_not_like_url :
            if spacy_token.like_url :   
                if is_verbose : print("like_url")
                is_valid = False
            else :
                pass
    return is_valid
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_is_token_vocabulary(doc_string, lang='en') :
    '''Returns True if token belongs language vocabulary, False otherwise.
    '''
    
    return [not spacy_token.is_oov for spacy_token in SPACY_LANGUAGE_MODEL(doc_string)]    
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_entity_label_is_valid(lang='en', is_verbose=False) :
    '''Returns True if all labels from entity-label dictionaty are valid.
    Validity label means : 
        * all labels do not belongs to stop-words
        * all labels do belong to spacy vocabulary
        * all labels having vector coefficient.
    
    '''
    if lang in DDICT_SPACY_LANG_ENTITY_LABEL.keys() :
        DICT_SPACY_LANG_ENTITY_LABEL = DDICT_SPACY_LANG_ENTITY_LABEL[lang]
    else :
        print("\n***ERROR : language= {} not yet supported!".format(lang))
        return False

    #------------------------------------------------------
    # Checking entities labels to belong stop words 
    #------------------------------------------------------
    is_stop_word = False
    for entity_label in DICT_SPACY_LANG_ENTITY_LABEL.values() :
        spacy_doc = SPACY_LANGUAGE_MODEL(entity_label)
        for spacy_token in spacy_doc :
            if spacy_token.is_stop :
                if is_verbose :
                    print("***WARN : Label= {} is a stop word!".format(entity_label))
                is_stop_word = True
    if is_verbose :
        if not is_stop_word :
            print("\n No stop words in labels replacing entities")    

    #------------------------------------------------------
    # Checking entities labels to belong vocabulary
    #------------------------------------------------------
    is_oov = False
    for entity_label in DICT_SPACY_LANG_ENTITY_LABEL.values() :
        spacy_doc = SPACY_LANGUAGE_MODEL(entity_label)
        for spacy_token in spacy_doc :
            if spacy_token.is_oov :
                if is_verbose :
                    print("***WARN : Label= {} is out of vocabuary".format(entity_label))
                is_oov = True
    if is_verbose :
        if not is_oov :
            print("\n No labels replacing entities out of vocabulary!")    

    #------------------------------------------------------
    # Checking entities labels having vectorization
    #------------------------------------------------------
    has_vector = True
    for entity_label in DICT_SPACY_LANG_ENTITY_LABEL.values() :
        spacy_doc = SPACY_LANGUAGE_MODEL(entity_label)
        for spacy_token in spacy_doc :
            if not spacy_token.has_vector :
                if is_verbose :
                    print("***WARN : Label= {} has no vector!".format(entity_label))
                has_vector = False
    if is_verbose :
        if has_vector :
            print("\n All labels replacing entities are vectorized!")        

    #------------------------------------------------------
    # Checkings summarization
    #------------------------------------------------------
    is_valid = has_vector & (not is_oov) & (not is_stop_word)
    return is_valid
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_oov_replace_suspicious(document, replaced=None):
    '''Replace all words out of vocabulary (oov), when found, with a 
    word given as parameter.
    
    Returns document with replaced words.
    '''
    if replaced is None :
        return document
    else :
        pass

        
    spacy_doc = p9_util_spacy.SPACY_LANGUAGE_MODEL(document)
    new_string = ''
    for token in spacy_doc:

        if token.is_oov :
            new_string += replaced+' '
        else :
            new_string += token.text+' '
    # Last character is removed, it is white space.
    new_string = new_string[:-1]
    return new_string
#-------------------------------------------------------------------------------  

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_entity_replace(document, replaced=None):
    '''Replace all words of type entity in document, when found, with a label 
    issue from label-entity dictionary.
    
    Returns document with replaced words.
    '''
    if replaced is None :
        return document
    else :
        pass
    
    if 0 == len(p9_util_spacy.SPACY_LANGUAGE_MODEL(document)) :
        return document
    else :
        #-----------------------------------------------------------------------
        # An entity is added at the start of the document, in order the document 
        # to be processed from beguining.
        #-----------------------------------------------------------------------
        document = replaced+" "+document
        len_added = len(replaced+" ")
    spacy_doc = p9_util_spacy.SPACY_LANGUAGE_MODEL(document)
    lang = spacy_doc.lang_
    
    if lang in p9_util_spacy.DDICT_SPACY_LANG_ENTITY_LABEL.keys() : 
        DICT_SPACY_ENTITY_LABEL = p9_util_spacy.DDICT_SPACY_LANG_ENTITY_LABEL['en']
    else :
        print("\n*** ERROR : spacy_entity_replace() : language not supported : {}".format(lang))
        
    new_doc=''
    index=0
    k_start=0
    k_end=0
    for ent in spacy_doc.ents :

        #print(doc_string[ent.start_char : ent.end_char], end='')
        #print(ent.label_, end=' ')
        #print(ent.start_char,ent.end_char)
        k_start= ent.end_char
        entity = str(ent.label_)
        
        #-----------------------------------------------------------------------
        # Replace entity wih a name belonging to vacabulary
        #-----------------------------------------------------------------------
        label = DICT_SPACY_ENTITY_LABEL[entity]
        #new_doc +=str(ent.label_)
        new_doc += label.lower()
        #new_doc += ' '
        if index+1 < len(spacy_doc.ents) :
            k_end = spacy_doc.ents[index+1].start_char
        else : 
            k_end = len(document)
        plain_text = document[k_start:k_end]
        new_doc += str(plain_text)
        #print("Plain : {}-->{} : {}".format(k_start,k_end,plain_text))
        index +=1
    #-----------------------------------------------------------------------
    # Shift from first entity added at the beguining of the process, plus 1
    # because process adds a white space when replacing word with entity label.
    #-----------------------------------------------------------------------
    return new_doc[len_added+1:]
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_token_norm_vectorization(list_token_string) :
    return [p9_util_spacy.SPACY_LANGUAGE_MODEL(token_string).vector_norm for \
    token_string in list_token_string ]
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_list_token_2_matrix_2D(list_token_string, dict_token_coeff=None) :
    '''Transforms a list of tokens into a 2D matrix where :
        *   The number of rows is the number of tokens in the given list 
        *   The number of columns is the dimension vector of any token assigned 
            from Scapy model 'spacy_nlp_md' (embedding dimension)
        
    Input : 
        *   list_token_string : a list of tokens as strings, issued from a text 
            tokenization. 
        *   dict_token_coeff : dictionary of coefficients for each token, 
            structured as : {token_string:coeff}
    output :
        *   2D matrix (nb tokens X Spacy word dimension)
    '''
    nb_col = p9_util_spacy.SPACY_LANGUAGE_MODEL.vocab.vectors_length

    if 0 == len(list_token_string) : 
        print("\n***WARNING : empy list of tokens")
        return None
    else :
        pass

    if dict_token_coeff is None :
        list_token_coeff = [1 for token in range(len(list_token_string))]
        dict_token_coeff = {token_string:1. for token_string  in list_token_string}
    else :
        list_token_coeff = list()
        for token in list_token_string :
            if token in dict_token_coeff.keys() :
                list_token_coeff.append(dict_token_coeff[token])
            else :
                list_token_string.remove(token)
            
    nb_row = len(list_token_string)
    vector_shape = (max(1,nb_row), nb_col)
    matrix_2D = np.zeros(vector_shape)
    
    if 0 == nb_row :
        #print("spacy_list_token_2_matrix_2D : nb_row= {}".format(nb_row))
        #-----------------------------------------------------------------------
        # Returns the zero vector when list of tokens is empty.
        #-----------------------------------------------------------------------
        return matrix_2D
    #-----------------------------------------------------------------------
    # All tokens out of dict_token_coeff have been removed.
    # list_token_string and list_token_coeff have the same number of items.
    # Their position in each list are synchronized???
    #-----------------------------------------------------------------------
    index = 0   
    for token_string  in list_token_string :
        token_coeff = dict_token_coeff[token_string]
        matrix_2D[index] = token_coeff*p9_util_spacy.SPACY_LANGUAGE_MODEL(token_string).vector
        index +=1
        
    return matrix_2D
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_list_token_2_tensor(list_token_string, dict_token_coeff = None, \
                              is_matrix=False) :
    ''' Transforms a list of tokens into either a 2D matrix or a 1D matrix, 
    means, a vector.
    When token to vector transformation takes place, then :
        * List of tokens is firstly transformed into a 2D matrix; each token 
        from list is vectorized. Coordonates of any vector are weighted with 
        coefficients from dict_token_coeff, when not None. Such coefficient 
        may be TF-IDF values assigned to each token.
        * List of tokens vectorization is then achieved by summing any vector 
        coefficients over each column.


        2D matrix for each tokenized text
        =========

                      Text over        Text over
                      0th              Jth 
                      dimension        dimension
           Tokenized +----------+-----+---------+
             text    |          |     |         |
               |     | coeff_0  | ... | coeff_J | <-- Embeding dimensions
               v     |          |     |         |
           +---------+----------+-----+---------+
           | token_0 |  M[0,0]  | ... | M[0,J]  | 
           +---------+----------+-----+---------+
                                   .
                                   .
                                   .
           +---------+----------+-----+---------+
           | token_I |  M[I,0]  | ... | M[I,J]  |
           +---------+----------+-----+---------+
                           |              |
                           |              |
          Vector           v              v
          ======    Sum(M[I,0])       Sum(M[I,J])
                    I=0,N             I=0,N

      coeff_O,...,coeff_M are embeddings vectors coefficients.
      
    '''
    if 0 == len(list_token_string) : 
        print("\n***WARNING : empy list of tokens")
        return None
    else :
        pass

    matrix_2D = spacy_list_token_2_matrix_2D(list_token_string,\
                                             dict_token_coeff)
    if is_matrix :
        return matrix_2D
    else :
        vector =  matrix_2D.sum(axis=0) 
        return vector / len(vector)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_clean(list_corpus, min_token_len, max_token_len,\
                list_word_most_frequent=list()) :
    '''Cleaning process using spacy librarie includes :
        * Tokenization of any document using spacy tokenizer
        * Cleaning any document from punctuation
        * Cleaning any document from token out of range length.
        * Applying lemmatization over any token in document.
       Once the cleaning process is done, a DataFrame is created with 
       column name column_name that contains cleaned documents.
    Input :
        *   list_corpus : a list of documents.
        *   min_token_len : minimum number of tokens in document for not beeing removed
        *   max_token_len : maximum number of tokens in document for not beeing removed.
    Output :
        *   pandas Series with tokenized list of words.
                      
    '''
    spacy_nlp = p9_util_spacy.SPACY_LANGUAGE_MODEL
    
    ser_corpus = pd.Series(list_corpus)
    ser = \
    ser_corpus.apply(lambda document: [ token.lemma_ if (token.text not in list_word_most_frequent) else token.text for token in spacy_nlp(document) if (\
                                    \
                                  p9_util_spacy.spacy_is_token_valid(token, \
                                  min_token_length=min_token_len,\
                                  max_token_length=max_token_len)    )] )

    return ser 
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_dataprep(ser_corpus, oov_keyword=None, \
                   entity_keyword=None, min_token_len=2, max_token_len=15):
    '''Process data preparation of corpus given as parameter.
    1)  Replace suspicious word with a key word.
    2)  Detect entities in document and replace it with a key word.
    3)  Clean corpus considering min number and max number of tokens from any 
    text of corpus and build tokens.
    4)  Returns a dataframe with columns issued from this data-preparatin 
    process.
    '''
    if oov_keyword is not None :
        ser_corpus = \
        ser_corpus.apply(lambda doc: spacy_oov_replace_suspicious(doc,replaced=oov_keyword))
    else : 
        pass
    if entity_keyword is not None :
        ser_corpus = ser_corpus.apply(lambda doc: p9_util_spacy.spacy_entity_replace(doc, replaced=entity_keyword))
    else : 
        pass

    #---------------------------------------------------------------------------
    # Cleaning...    
    #---------------------------------------------------------------------------
    ser_token = spacy_clean(ser_corpus,min_token_len,max_token_len)
    
    #---------------------------------------------------------------------------
    # Count tokens
    #---------------------------------------------------------------------------
    ser_count = \
    ser_token.apply(lambda list_token: len(list_token))
    df_data = pd.DataFrame({p9_util_spacy.COLUMN_NAME_DOC:ser_corpus, \
                            p9_util_spacy.COLUMN_NAME_COUNT:ser_count,\
                            p9_util_spacy.COLUMN_NAME_TOKEN:ser_token})
    return df_data
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_token2Tensor(ser_doc,ser_token, dict_token_tfidf=None, is_matrix=False):
    '''
        
    '''    
    if dict_token_tfidf is not None :
        if 0 == len(dict_token_tfidf) :
            dict_token_tfidf = None
        else :
            pass
    else :
        pass
    ser_tensor =\
    ser_token.apply(lambda list_token: \
                    spacy_list_token_2_tensor(list_token,\
                    dict_token_coeff=dict_token_tfidf, is_matrix=is_matrix))
    return ser_tensor
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_vectorizer(ser_doc,ser_token, dict_token_tfidf=None, is_matrix=False):
    '''
        
    '''    
    if dict_token_tfidf is not None :
        if 0 == len(dict_token_tfidf) :
            dict_token_tfidf = None
        else :
            pass
    else :
        pass
    ser_vector =\
    ser_token.apply(lambda list_token: \
                    spacy_list_token_2_tensor(list_token,\
                    dict_token_coeff=dict_token_tfidf, is_matrix=is_matrix))
    return ser_vector
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def spacy_tokenizer(list_corpus, target=None, is_tokenizer=True, \
                min_token_len=2, max_token_len=15, min_word_len=1, \
                max_word_len=500, max_length = None, is_matrix_2D=True, \
                is_spacy_data_prep=True, oov_keyword=None, entity_keyword=None,\
                return_df_data=False, csr_tfidf=None, df_dataprep=None, \
                ser_vector=None):
    '''Process steps : 
        1) Process data-preparation; data-preparation transforms corpus into 
        a standardized dataset. 
        Tokenization takes place into data-preparation.
        2) Remove rows from dataframe outside range [min_doc_len, max_doc_len]
        
        If this function is called with flag is_tokenizer fixed to True, then 
        this function is used along with objects such CountVectorizer; due to 
        that, the list of tokens is retunred.
        This use case takes place for BOW model.
        
        Otherwise, return depends of return_df_data flag; in this case,
        following return context takes place : 
              +-----------------+----------------+-------------------+
              | return_df_data  |     y          | Returned values   |
              +-----------------+----------------+-------------------+
              |       False     |  is None       |    X              |
              |       False     |  is not None   |    X, y           |
              |       True      |  is not None   |    X,y,df_data    |
              +-----------------+----------------+-------------------+
        
    '''
    
    #list_corpus = corpus.copy()
    if df_dataprep is None :
        ser_corpus = pd.Series(list_corpus)
        if is_spacy_data_prep :
            df_dataprep = spacy_dataprep(ser_corpus, oov_keyword=oov_keyword, \
            entity_keyword=entity_keyword, \
            min_token_len=min_token_len, max_token_len=max_token_len)    
        else :
            pass
    else : 
        pass


    #---------------------------------------------------------------------------
    # Target to be aggregated to dataframe
    #---------------------------------------------------------------------------
    if target is not None :
        ser_target = pd.Series(target)
    else :
        ser_target = None

    #---------------------------------------------------------------------------
    # Create dataframe from built Series
    #---------------------------------------------------------------------------
    ser_token  = df_dataprep[p9_util_spacy.COLUMN_NAME_TOKEN]
    ser_corpus = df_dataprep[p9_util_spacy.COLUMN_NAME_DOC]
    ser_count  = df_dataprep[p9_util_spacy.COLUMN_NAME_COUNT]
    
    df_data = pd.DataFrame({p9_util_spacy.COLUMN_NAME_DOC:ser_corpus, \
                            p9_util_spacy.COLUMN_NAME_TOKEN:ser_token, \
                            p9_util_spacy.COLUMN_NAME_COUNT:ser_count,\
                            p9_util_spacy.COLUMN_NAME_VECTOR: ser_vector, \
                            p9_util_spacy.COLUMN_NAME_TARGET:ser_target})

    #-----------------------------------------------------------------------
    # Remove rows from dataframe outside range [min_doc_len, max_doc_len]
    #-----------------------------------------------------------------------
    list_index_drop = df_data[df_data[p9_util_spacy.COLUMN_NAME_COUNT]<min_word_len].index
    df_data.drop(index=list_index_drop, inplace=True)
    
    if 0 < max_word_len :
        list_index_drop = df_data[df_data[p9_util_spacy.COLUMN_NAME_COUNT]>max_word_len].index
        df_data.drop(index=list_index_drop, inplace=True)
    else :
        pass
    if not is_tokenizer :
        #-----------------------------------------------------------------------
        # return X and y as arrays.
        #-----------------------------------------------------------------------
        X = p9_util.convert_ser2arr(ser_vector)
        if ser_target is not None :
            y = p9_util.convert_ser2arr(ser_target)
        else :
            y=None

        if return_df_data :
            return X,y,df_data
        else :
            if y is None :
                return X
            else :
                return X,y
    else : 
        #-----------------------------------------------------------------------
        # Used as a tokenizer and returns a tokenized doc from dataframe 
        # column COLUMN_NAME_TOKEN
        #-----------------------------------------------------------------------
        ret_ = list()
        try :
            ret_ = df_data[p9_util_spacy.COLUMN_NAME_TOKEN].tolist()[0]
        except IndexError as indexError:
            print("\n*** ERROR : p9_util_spacy.spacy_tokenizer() : "+str(indexError))
            print("p9_util_spacy.spacy_tokenizer() : df_data[p9_util_spacy.COLUMN_NAME_TOKEN].tolist()= "+str(df_data[p9_util_spacy.COLUMN_NAME_TOKEN].tolist()))
            print("p9_util_spacy.spacy_tokenizer() : tokens issued from spacy tokenizer= "+str(ser_token))
        return ret_
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def build_padded_matrix(df_data, dict_token_coeff=None, max_row=None)   :
    
    if dict_token_coeff is None :
        pass
    else :
        if 0 == len(dict_token_coeff) :
            dict_token_coeff = None
        else :
            pass
    #---------------------------------------------------------------------------
    # When calling this function, then this columns is deleted from dataframe.
    #---------------------------------------------------------------------------
    if COLUMN_NAME_VECTOR in df_data.columns :
        del (df_data[COLUMN_NAME_VECTOR])
    else :
        pass

    #---------------------------------------------------------------------------
    # Build the matrix from tokens; each text is standardized with tokens list.
    # Each token from this list is represented with an embedding vector issued 
    # from Spacy.
    #---------------------------------------------------------------------------
    print("\nbuild_padded_matrix : build of matrix for each text...")
    ser_matrix =\
        df_data['tokens'].apply(lambda list_token: \
                        p9_util_spacy.spacy_list_token_2_tensor(list_token,\
                        dict_token_coeff=dict_token_coeff, is_matrix=True))
    
    df_data['matrix'] = ser_matrix
    if max_row is None :
        max_row = df_data['counting'].max()
    else :
        pass

    #---------------------------------------------------------------------------
    # Truncate the number of rows in each matrix that represents a text.
    # This lead to resctrict the number of token per text or to padd the 
    # this number of tokens. Padding leads to add rows with zero values for 
    # each dimension.
    #---------------------------------------------------------------------------
    print("\nbuild_padded_matrix : Padd / truncate matrix rows...")
    ser_matrix_padded =\
        df_data['matrix'].apply(lambda input_matrix: \
                        p9_util.matrix_zeropadd_row(input_matrix,max_row))

    del(df_data['matrix'])
    df_data['matrix_padded'] = ser_matrix_padded
    

    return df_data
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def scapy_weight_matrix_build(spacy_nlp,llist_token_string):
    '''
    Build a weight matrix.
    Dictionary {word:glove_coefficient} is built from Glove file name.
    Glove file has been prealably downloaded.
    Once built, dictionary allows to build a vector for every word in vocabulary 
    issued from tokenizer.
    Using glove file defined here-under, each word is a vector of dimension 100. 
    This dimension is referenced in the Glove file name part as 100d

    Endly, weights matrix is built from vocabulary issued from tokenizer.

    Such process is summarized with sequences here-under :

        * dict_glove_word_coeff <-- processing Glove file name
        * vocabulary_word, index <-- tokenizer
        * weight_vector = dict_glove_word_coeff[vocabulary_word]
        * weight_matrix[index] = weight_vector
    
    Input :
        * spacy_nlp : Spacy language model. May be one of the Spacy model 
            with embeddings vector : en_core_web_md, en_core_web_lg, 
            en_vectors_web_lg
        * llist_token_string : list of list of tokens. This is the tokenized 
        corpus.
    Output :
        * weight matrix structured as an array [vocab_size, vector_dimension] 
        where vector_dimension is the dimension of spacy_nlp embeddings vectors.
        
        * dictionary of structured as follwing : {index:token_string}
        where index is a row of weight matrix and token_string a token from the 
        set of tokenized corpus.
    '''
    set_token_string = set()
    index = 0
    
    for list_token_string in llist_token_string :
        for token_string in list_token_string :
            set_token_string.add(token_string)

    vocab_size = len(set_token_string)
    vector_dimension = spacy_nlp.vocab.vectors_length
    weight_matrix = np.zeros((vocab_size, vector_dimension))
    dict_index_tokenstring = dict()
    for token_string, index in zip(set_token_string, range(vocab_size)) :
        if nlp.vocab.has_vector(token_string):
            weight_matrix[index] = nlp.vocab.get_vector(token_string)
            dict_index_tokenstring[index] = token_string
    return weight_matrix,dict_index_tokenstring
#-------------------------------------------------------------------------------
