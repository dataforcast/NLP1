"""This file contains all utilities functions used in for project 
issue by Open Classromm in the context od Data Scientist master.
"""

import numpy as np
import pandas as pd
import re
import random

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

from scipy import sparse

from bs4 import BeautifulSoup


from sklearn.feature_extraction.text import CountVectorizer

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.decomposition import LatentDirichletAllocation

import p5_util


LIST_EMBEDDING_MODE=['tfidf','bow','ngram']

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_gscv_best_classifier(dict_param_grid, classifier, X_train, y_train\
                              , X_test,y_test,cv=None,iid=None):
                              

    true_classifier = OneVsRestClassifier(classifier)
    if cv is None :
        if iid is None :
            gscv_classifier  = GridSearchCV(true_classifier, dict_param_grid)
        else :
            gscv_classifier  = GridSearchCV(true_classifier, dict_param_grid\
            , iid=iid)
        
    else :
        if iid is None :
            gscv_classifier  = GridSearchCV(true_classifier, dict_param_grid\
            , cv=cv)
        else :
            gscv_classifier  = GridSearchCV(true_classifier, dict_param_grid\
            , cv=cv, iid=iid)
    
    gscv_classifier.fit(X_train, y_train)
    print (gscv_classifier.best_score_)
    print (gscv_classifier.best_params_)
    y_pred = gscv_classifier.best_estimator_.predict(X_test)
    return y_pred, gscv_classifier
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_gscv_best_classifier_result(dict_param_grid, classifier, X_train, y_train\
                              , cls_name, X_test,y_test, dict_cls_score):
    
    y_pred , gscv_classifier = p6_gscv_best_classifier(dict_param_grid\
    , classifier, X_train, y_train, X_test,y_test)
    
    cls_score = p6_util.p6_supervized_mean_accuracy_score(y_test, y_pred)
    print("Mean accuracy score for "+cls_name+" : {0:1.2F} %".format(cls_score*100))
    dict_cls_score[cls_name]=cls_score
    return dict_cls_score, gscv_classifier.best_estimator_
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_get_vectorized_doc(vectorizer, doc) :
    """Returns a vectorized document issued from vectorizer operator.
    Document given as function parameter is expressed in a natiral language.
    Vectorization process includes document standardization.
    
    Input : 
        * vectorizer : operator used for document vectorization.
        * doc : document to be vectorized.
    Output : 
        * X : vectorized document
    """
    ser_post_std = p6_str_standardization(doc)
    X = vectorizer.transform(ser_post_std)
    return X
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def clean_marker_text(text,leading_marker=None, trailing_marker=None):
    """ Remove trailer and leading markers from a given text.
    Returns a list of words that were encapsulated between markers.
    """
    if leading_marker is not None:
        list1=text.split(leading_marker)
        str2 = " ".join(list1)
    else :
        str2 = text
    if trailing_marker is not None: 
        list2=str2.split(trailing_marker)
    else :
        list2 = str2.split()

    list3 = [element.strip() for element in list2 if 0 < len(element)]
    return list3
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_df_standardization(ser, is_stemming=False, is_lem=True\
    , is_stopword=True, verbose=True, list_to_keep=list()
    , is_sentence_filter=False
    , is_lxml=True) :
    """ Applies all pre-processing actions over a Series ser given as 
    parameter.
    
    Returned Series is a cleaned one. 
    """

    if verbose is True :
        print("\nCleaning text in-between markers <code></code> markers...")
    ser = ser.apply(cb_remove_marker,args=('code',))

    if is_lxml is True :
        if verbose is True :
            print("\nCleaning LXML markers...")
        ser = ser.apply(cb_clean_lxml)

    if is_sentence_filter is True :
        if verbose is True :
            print("\nRemove non alpha-numeric words from sentences...")
        ser = ser.apply(cb_sentence_filter, args=(list_to_keep,))

    if verbose is True :
        print("\nRemove verbs from sentences...")
    ser = ser.apply(cb_remove_verb_from_sentence, args=(list_to_keep,))

    
    if is_stopword is True :
        if verbose is True :
            print("\nRemoving stopwords...")
        ser= ser.apply(cb_remove_stopwords, args=(list_to_keep,))

    if is_lem is True:
        if verbose is True :
            print("\nLemmatization ...")
        lemmatizer=WordNetLemmatizer()
        ser=ser.apply(p5_util.cb_lemmatizer,args=(lemmatizer,'lower'))

    if is_stemming is True:
        if verbose is True :
            print("\nEnglish stemming ...")
        stemmer=SnowballStemmer('english')
        ser=ser.apply(p5_util.cb_stemmer,args=(stemmer,'lower'))
    return ser
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_str_standardization(post, is_stemming=False, is_lem=True, is_stopword=True\
    ,is_stopverb=True, is_stopalfanum=False, list_to_keep=list(), verbose=False):
    """Apply to a given POST all transformations in order to clean and 
    standardize text model.
        Input : 
            * post : suite of words forming a POST
            * is_stemming : when True, stemming is applied on given post.
            * is_lem : when True, lemmatization is applied on given post.
            * is_stopword : when True, engish stopwords are filtered from post.
            * is_stopverb : when True, engish verbs are filtered from post.
            * is_stopalfanum : when True, non alpha-numeric characters are filtered 
            from given POST.
        Output :
            * dataframe with  standardized Body column.
    """
    df = pd.DataFrame({'Body':post}, index=[0,])
   
    #print("\nCleaning text in-between markers <code></code> markers...")
    df["Body"] = df.Body.apply(cb_remove_marker,args=('code',))

    ser = p6_df_standardization(df["Body"], is_stemming=is_stemming\
    , is_lem=is_lem, verbose=verbose, list_to_keep=list_to_keep\
    , is_sentence_filter=is_stopalfanum)
    return ser

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_df_from_FreqDist(freqDist, query=None):
    """Returns a dataframe from a given nltk.probability.FreqDist object class.
    The returned dataframe have a single column named Freq.
    """
    if freqDist is not None:
        df = pd.DataFrame.from_dict(freqDist, orient='index')
        if 'Freq' not in df.columns:
            df.rename(columns={0:'Freq'}, inplace=True)
            df.sort_values(by = 'Freq', inplace=True, ascending=False) 
        else:
            pass
    if query is not None:
        df=df.query(query)
    else :
        pass
    return df
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_tag_original(str_tag) :
    str_tag_original = re.sub("[^a-zA-Z0-9=++# ]", " ", str_tag )
    list_tag_original = [tag for tag in str_tag_original.split(' ') \
    if tag not in ['',]]
    return list_tag_original
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_tag_stat_tfidf(sentence, vectorizer, tag_original_count=0\
                            , tag_ratio=1.0) :
    """ Returns a dictionary with suggested TAGS from sentence given as 
    parameter.

    List of returned TAGs into dictionary is a ratio from sentence words count.
    
    This function appplies for TFIDF vectorization only. This mean, sentence is
    represented as a vector from which components values are TFIDF values of 
    terms from vocabulary.
             +---------+-----+---------+-----+---------+
             | vocab_1 | ... | vocab_i | ... | vocab_K |<--issue from vectorizer
    ---------+---------+-----+---------+-----+---------+
    sentence | tfidf_1 | ... | tfidf_i | ... | tfidf_K |<--issue from vectorizer
    ---------+---------+-----+---------+-----+---------+
    
    Those values are stored into csr_matrix given as parameter 
    while terms from vocabulary are stored into vectorizer.
    
    Selected tags belong to both, vocabulary and words from given sentence.

    Terms from vocabulary having greater occurence value are selected as TAGs.
    
    Input :
        * sentence : words separated by an empty space.
        * vectorizer : operator used for TF-IDF vectorization process and 
        containing words vocabulary.
        * tag_original_count : number of assigned TAG for the sentence.
        * tag_ratio : ratio of suggested TAG foro the number of words in the 
        sentence.
     Output : 
        * dictionary structured as following : 
            {word:TFIDF} where TFIDF are sorted values, reversed order.
    
    """
    df_value_name = 'TFIDF'
    # Sentence is splitted as alist of words.
    list_term_sentence = sentence.split(' ')
    
    #---------------------------------------------------------------------------
    # Using vectorized sentence in csrmatrix, get indexes from all features 
    # with frequencies >0 
    #---------------------------------------------------------------------------
    csrmatrix= vectorizer.transform([sentence])
    arr_index = np.where(csrmatrix.A>0)[1]

    #---------------------------------------------------------------------------
    # Using array of indexes from features contained in sentence, 
    # vocabulary terms are extracted and so forth, are suggested TAG.
    #---------------------------------------------------------------------------
    list_tag_sentence= [tag for tag, index in vectorizer.vocabulary_.items() \
    if index in arr_index]

    #---------------------------------------------------------------------------
    # Build dictionary {TAG:TFIDF} where TAG are issued from the given row.
    #---------------------------------------------------------------------------
    dict_tfidf = dict()
    for tag_sentence in list_tag_sentence : 
        index = vectorizer.vocabulary_[tag_sentence]
        dict_tfidf[tag_sentence]= vectorizer.idf_[index]

    #---------------------------------------------------------------------------
    # Get tag_count, from dict_index, with highest TF-IDF values.
    #---------------------------------------------------------------------------
    if tag_ratio is not None :
        tag_count = int(len(list_term_sentence)*tag_ratio)
    else :
        tag_count = tag_original_count
        
    df_row_tfidf = pd.DataFrame.from_dict(dict_tfidf, orient='index')

    if 0 < df_row_tfidf.shape[0] and 0< df_row_tfidf.shape[1]:
        df_row_tfidf = df_row_tfidf.rename(columns={0:'TFIDF'}, inplace=False)
        df_row_tfidf.sort_values(by=[df_value_name],  ascending=False, inplace=True)
    else : 
        return None

    #---------------------------------------------------------------------------
    # TFIDF tag_count greatest values are returned.
    #---------------------------------------------------------------------------
    return df_row_tfidf[:tag_count].to_dict()[df_value_name]
    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_dict_vocab_frequency(vectorizer, csr_matrix):
    """Returns a dictionary {index:frequency} where : 
         * index values are indexes from vocabulary
         * frequency is the number of occurences of a term from vocabulary

       csr_matrix is used in order to compute occurences for any feature.
       A feature from csr_matrix is a term in vocabulary.

       This fuction appplies for BOW vectorization only.
    """
    
    if type(vectorizer) is not CountVectorizer:
        print("\n*** ERROR : get_dict_frequency function should apply with \
        CountVectorizer type only!")
        return None
    else: 
        pass

    #-------------------------------------------------------------------------------
    # Get list of indexes for any features
    #-------------------------------------------------------------------------------
    list_feature_index = [index  for (tag, index) \
    in vectorizer.vocabulary_.items()]

    #-------------------------------------------------------------------------------
    # Total occurence is computated for any feature.
    #-------------------------------------------------------------------------------
    arr_total_occurence = csr_matrix.toarray().sum(axis=0)

    #-------------------------------------------------------------------------------
    # Build dictionary {feature_index:occurence} for the given row
    #-------------------------------------------------------------------------------
    dict_vocabularyIndex_occurence = dict()
    for feature_index, total_occurence_value \
    in zip(list_feature_index, arr_total_occurence) :
        dict_vocabularyIndex_occurence[feature_index]=total_occurence_value

    return dict_vocabularyIndex_occurence
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_tag_stat_ngram(sentence, vectorizer, csr_matrix\
    , tag_ratio=1.0, tag_original_count=0) :
    """
    """
    #---------------------------------------------------------------------------
    # Checking parameters
    #---------------------------------------------------------------------------
    if type(vectorizer) is not CountVectorizer:
        print("\n*** ERROR : get_dict_frequency function should apply with \
        CountVectorizer type only!")
        return None
    else: 
        pass

    df_value_name = 'Frequency'

    #---------------------------------------------------------------------------
    # Using vectorized sentence in csrmatrix, get indexes from all features 
    # with frequencies >0 
    #---------------------------------------------------------------------------
    csrmatrix= vectorizer.transform([sentence])
    arr_index = np.where(csrmatrix.A>0)[1]

    #---------------------------------------------------------------------------
    # Using array of indexes from features contained in sentence, 
    # vocabulary terms are extracted.
    #---------------------------------------------------------------------------
    list_tag= [tag for tag, index in vectorizer.vocabulary_.items() \
    if index in arr_index]

    #---------------------------------------------------------------------------
    # Build dictionary {TAG:TotalFreq} where TotalFreq is the sum over 
    # all rows from corpus matrix (csr_matrix) for any TAG belonging to list_tag.
    #---------------------------------------------------------------------------
    dict_frequency_row = dict()
    dict_vocab_frequency = get_dict_vocab_frequency(vectorizer, csr_matrix)
    for tag in list_tag : 
        index_tag = vectorizer.vocabulary_[tag]
        dict_frequency_row[tag] = dict_vocab_frequency[index_tag]

    #---------------------------------------------------------------------------
    # Dictionary is converted as a dataframe allowing values to be ordered.
    #---------------------------------------------------------------------------
    df_row_frequency \
    = pd.DataFrame.from_dict(dict_frequency_row, orient='index')
    
    df_row_frequency.rename(columns={0:df_value_name}, inplace=True)
    df_row_frequency.sort_values(by=[df_value_name],  ascending=False\
    , inplace=True)

    #---------------------------------------------------------------------------
    # Get tag_count computed from ratio.
    #---------------------------------------------------------------------------
    list_term_sentence = sentence.split(' ')
    if tag_ratio is not None :
        tag_count = int(len(list_term_sentence)*tag_ratio)
    else : 
        tag_count = tag_original_count

    return df_row_frequency[:tag_count].to_dict()[df_value_name]

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_tag_stat_bow(sentence, vectorizer, csr_matrix, tag_ratio=1.0) :
    """ Returns a list of TAGS from sentence given as parameter.
    List of returned TAGs count is a ratio from sentence words count.
    
    This fuction appplies for BOW vectorization only. This mean, documents are
    vectors from which components values are occurences or frequencies of 
    terms from vocabulary. 
    
    Terms may be N-gram type with N>=1.
    
    Those frequencies values are stored into csr_matrix given as parameter 
    while terms from vocabulary are stored into vectorizer.
    
    Selected tags belong both to vocabulary and words from given sentence.

    Terms from vocabulary having greater occurence value are selected as TAGs.
    
    """
    #---------------------------------------------------------------------------
    # Checking parameters
    #---------------------------------------------------------------------------
    if type(vectorizer) is not CountVectorizer:
        print("\n*** ERROR : get_dict_frequency function should apply with \
        CountVectorizer type only!")
        return None
    else: 
        pass

    #---------------------------------------------------------------------------
    # Get intersection between words in vocabulary and and words in list_term.
    # TAG will be assigned from vocabulary issued from vectorization.
    #---------------------------------------------------------------------------
    list_term_sentence = sentence.split(' ')
    list_tag_row = list(set(list_term_sentence) & set(vectorizer.vocabulary_))

    #---------------------------------------------------------------------------
    # Build dictionary {TAG:Freq}
    #---------------------------------------------------------------------------
    dict_frequency_row = dict()
    dict_frequency = get_dict_vocab_frequency(vectorizer, csr_matrix)
    for tag_row in list_tag_row : 
        index = vectorizer.vocabulary_[tag_row]
        dict_frequency_row[tag_row] = dict_frequency[index]

    df_row_frequency \
    = pd.DataFrame.from_dict(dict_frequency_row, orient='index')
    
    df_row_frequency.rename(columns={0:'value'}, inplace=True)
    df_row_frequency.sort_values(by=['value'],  ascending=False, inplace=True)

    #---------------------------------------------------------------------------
    # Get tag_count most greater value from dict_index
    #---------------------------------------------------------------------------
    tag_count = int(len(list_term_sentence)*tag_ratio)

    return sorted(list(df_row_frequency.index[:tag_count]))
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_clean_lxml(str_lxml, mode='lower'):
    """This function clean (remove) all LXML markers from str_lxml given as 
    parameter.
    
    BeautifulSoup library is used to process LXML strings.
    
    It returns a cleaned string in low case.
    
    If mode value given as parameter is neither lower nor upper, then cleaned 
    string is returned in lower mode.
    """

    if 'upper' == mode : 
        soup = BeautifulSoup(str_lxml.upper(),"lxml") 
    else : 
        soup = BeautifulSoup(str_lxml.lower(),"lxml") 
        
    return soup.get_text()

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_sentence_filter(sentence, list_to_keep, mode='lower'):
    """Remove all patterns that are not alpha-digital words, 
    also not '=' nor '++' characters; replace then with ' ' character.    
    """
    
    sentence_filtered = re.sub("[^a-zA-Z0-9=++#]", " ", sentence )
    if mode=="upper" :
        sentence_filtered = sentence_filtered.upper()
    else:
        sentence_filtered = sentence_filtered.lower()

    return sentence_filtered
    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_code_source_compare(item):
    """This function removes from item given as parameter source code 
    comparaison expression such as A=B or A>=cf,...
    """
    my_text = item.replace(' = ','=')
    my_text = my_text.replace(' == ','==')
    my_text = my_text.replace(' > ','>')
    my_text = my_text.replace(' < ','<')
    my_text = my_text.replace(' >= ','>=')
    my_text = my_text.replace(' <= ','<=')
    my_text = my_text.replace('  ',' ')
    
    
    my_text = re.sub("([A-Za-z0-9]+)(=+)([A-Za-z0-9]+)",' ',my_text)    
    my_text = re.sub("([A-Za-z0-9]+)(>)([A-Za-z0-9]+)",' ',my_text)    
    my_text = re.sub("([A-Za-z0-9]+)(<)([A-Za-z0-9]+)",' ',my_text)    
    my_text = re.sub("([A-Za-z0-9]+)(>=)([A-Za-z0-9]+)",' ',my_text)    
    my_text = re.sub("([A-Za-z0-9]+)(<=)([A-Za-z0-9]+)",' ',my_text)    
    return my_text
#-------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_sentence_from_df(df, feature) : 
    if feature not in df.columns : 
        print("*** ERROR : no column named "+str()+" into dataframe!")
        return None

    list_sentence = list()
    for sentence in df[feature].tolist():
        #tokenized_sentence = sentence.split(' ')
        tokenized_sentence=nltk.word_tokenize(sentence)
        list_sentence.append(tokenized_sentence)    
    return list_sentence
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_marker(str_marked,str_marker):
    """Removes text in between markers <str_marker> and </str_marker>
    
        * Input :         
            str_marked : marked string, mean, text in between markers
            str_marker : marker compounding removed text.
    """
    str_marked = str_marked.replace('\n','')
    marker_start='<'+str_marker+'>'
    marker_end='</'+str_marker+'>'
    str_cleaned = re.sub(marker_start+"(.*)"+marker_end,"", str_marked.lower())
    return str_cleaned
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_list_tag_from_post(post, ml_model, max_tag=5):
    """Returns a list of most appropriate tags from a post.
    Tags are computed from a M.L model.
    
    """
    list_tag_returned = list()
    tokenized_post = post.split()
    list_tag_pred = ml_model.predict_output_word(tokenized_post)
    tag_count = 0
    for tuple_word_percent in list_tag_pred :
        if tag_count < max_tag:
            tag = "<"+str(tuple_word_percent[0])+">"
            list_tag_returned.append(tag)
            tag_count += 1

    return list_tag_returned
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------

def cb_remove_verb_from_sentence(sentence, list_to_keep=list(), verbose=False) :
    tokenized_sentence = nltk.word_tokenize(sentence.lower())

    if verbose is True :
        print(tokenized_sentence)
        print()

    #Rebuild c# 
    i_offset=0
    tokens=None
    for i, token in enumerate(tokenized_sentence):
        i -= i_offset
        if token == '#' and i > 0:
            left = tokenized_sentence[:i-1]
            joined = [tokenized_sentence[i - 1] + token]
            right = tokenized_sentence[i + 1:]
            tokens = left + joined + right
            i_offset += 1
    
    if tokens is not None:
        tokenized_sentence = tokens
    if verbose is True :
        print(tokenized_sentence)
        print()


    list_tagged = nltk.pos_tag(tokenized_sentence)
    if verbose is True :
        print(list_tagged)
        print()



    list_pos_tag_excluded \
        = ['PRP','VB','VBP','VBN','CD','VBD','JJ','PRP','VBZ','DET','VBG','MD','NN$'\
           ,':',')','(','.','NNP','PDT',"``","''",'CC','POS',',','RB']

    list_token_excluded = ['i','s','m',',','question','this','can','using']
    list_token_filtered \
        = [token for (token, tag) in list_tagged \
        if str(tag) not in list_pos_tag_excluded \
        and token not in list_token_excluded  or token in list_to_keep \
        #or token in list_to_keep 
        ]
        
        
    if False :
        for token, tag in list_tagged :
            if tag == ':' :
                print(token,tag,tag in list_pos_tag_excluded,list_pos_tag_excluded)

    if verbose is True :
        print(list_token_filtered)
        print()
    sentence = " ".join(list_token_filtered)
    #sentence = nltk.Text(list_token_filtered)
    if True:
        tokenized_sentence \
                    = nltk.regexp_tokenize(sentence, pattern=r"\s|[\,;]:!?()<>", gaps=True)
        sentence = " ".join(tokenized_sentence)
    else :
        pass
    return sentence
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_verb_from_sentence_deprecated(sentence, list_to_keep=list()):
    """Remove verbs from sentence given as parameter and returns a sentence
    without any verb
    """
    
    if list_to_keep is None :
        list_to_keep = list()
    #---------------------------------------------------------------------------
    # All sentences are tokenized and words are tagged.
    #---------------------------------------------------------------------------
    list_tagged = list()
    
    #---------------------------------------------------------------------------
    # Remove special characters except `#`
    #---------------------------------------------------------------------------
    if False :
        if False :
            patterns = [
            (r'.*ing$', 'VBG'),               # gerunds
            (r'.*ed$', 'VBD'),                # simple past
            (r'.*es$', 'VBZ'),                # 3rd singular present
            (r'.*ould$', 'MD'),               # modals
            (r'.*\'s$', 'NN$'),               # possessive nouns
            (r'.*s$', 'NNS'),                 # plural nouns
            (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
            #(r'.*', 'NN')                     # nouns (default)
            ]

            regexp_tagger = nltk.RegexpTagger(patterns)
            tokenized_sentence = regexp_tagger.tag(sentence)
        else :        
            tokenized_sentence \
            = nltk.regexp_tokenize(sentence, pattern=r"\s|[\,;']:!?()<>", gaps=True)
            #sentence = " ".join(tokenized_sentence)
            #tokenized_sentence = nltk.word_tokenize(sentence)         
    else :
        tokenized_sentence = nltk.word_tokenize(sentence)
        #-----------------------------------------------------------------------
        # Rebuild c# 
        #-----------------------------------------------------------------------
        i_offset=0
        for i, token in enumerate(tokenized_sentence):
            i -= i_offset
            if token == '#' and i > 0:
                left = tokenized_sentence[:i-1]
                joined = [tokenized_sentence[i - 1] + token]
                right = tokenized_sentence[i + 1:]
                tokens = left + joined + right
                i_offset += 1
        tokenized_sentence = tokens

    list_tagged += nltk.pos_tag(tokenized_sentence)
    
    
    #---------------------------------------------------------------------------
    # List of words tagged as verbs are filtered
    #---------------------------------------------------------------------------
    if True :

        list_pos_tag_excluded \
        = ['PRP','VB','VBP','VBN','CD','VBD','JJ','PRP','VBZ','DET','VBG','MD'\
        ,'NN$',')','(','.','NNP','PDT']

        list_word_filtered \
            = [token for (token, tag) in list_tagged \
            if tag not in list_pos_tag_excluded \
            or token in list_to_keep \
            and token.isalpha()]
    
    else :
        list_word_filtered = list()    
        for tuple_word_tag in list_tagged:
            tag = tuple_word_tag[1]
            if 'V' == tag[0] and tuple_word_tag[0] not in list_to_keep:
                pass
            else:
                list_word_filtered.append(tuple_word_tag[0])
    #-----------------------------------------------------------------------
    # Tokenized words that have been filtered are compound into a sentence
    #-----------------------------------------------------------------------
    sentence = " ".join(list_word_filtered)
    if False :
        tokenized_sentence \
                = nltk.regexp_tokenize(sentence\
                , pattern=r"\s|[\,;]:!?()<>\\", gaps=True)
        sentence = " ".join(tokenized_sentence)
    return sentence

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_stopwords(item, list_to_keep, lang='english', mode='lower') :
    """This function removes some stopwords form item given as parameter.
    Removed stopwords are issued from 'nltk.corpus.stopwords'.
    
    """
    
    list_word=item.split()
    list_stop_words_lang = stopwords.words(lang)
    list_stop_words = [word for word in list_stop_words_lang if word not in list_to_keep]

    
#    item_no_stopwords_1=[ word for word in list_word if word.lower() not in list_stop_words ]

    item_no_stopwords_1=[ word for word in list_word \
    if word.lower() not in list_stop_words 
                          and not str(word).isdigit()]
               
               
    item_no_stopwords=[word for word in item_no_stopwords_1 \
    if word.lower() not in ['cnn','.','#','way','would','like']]
    
    item_no_stopwords=" ".join(item_no_stopwords)
    if mode == 'upper':
        return item_no_stopwords.upper()
    elif mode =='lower':
        return item_no_stopwords.lower()
    else:
        print("*** ERROR no mode defined for word! ***")
        return None
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def corpus_tokenization(dict_content, token_mode='split')   :
    """Tokenize each document from a corpus represented as a dictionary.

    Input :
        * dict_content : dictionary where values contain a document. The whole
        dictionary represents a corpus.
        * token_mode : split or nltk
    Output :
        dict_tokenized : dictionary of tokenized documents. Each value of 
        dictionary is a list of tokenized documents.
    """
    
    dict_tokenized = dict()
    if token_mode == 'split' :
        for root_name in dict_content.keys():
            content= dict_content[root_name]
            dict_tokenized[root_name] = content.split(' ')

    elif token_mode == 'nltk' :
        for root_name in dict_content.keys():
            content= dict_content[root_name]
            dict_tokenized[root_name] = nltk.word_tokenize(content)
    else :
        print("*** ERROR : unknown tokenization mode= "+str(token_mode))
        return None

    return dict_tokenized
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def compute_frequency_sentence(dict_content, token_mode='split'):
    """Computes each word frequency from a dictionary of contents.
    Contents from dict_content are tokenized using split() method as default 
    mode.
    
    Other mode option is nltk to tokenize contents.
    Input :
        * dict_content : dictionary where values contain a tokenized document. 
        The whole dictionary represents a corpus.
        * token_mode : split or nltk
    Output :
        freq_content : nltk.FreqDist type object mapping a token with its 
        frequency in the corpus. 
        dict_tokenized : dictionary of tokenized documents.
    """
    
    list_content = list()
    if token_mode == 'split' :
        for root_name in dict_content.keys():
            content= dict_content[root_name]
            tokenized_content = content.split(' ')
            list_content += tokenized_content

    elif token_mode == 'nltk' :
        for root_name in dict_content.keys():
            content= dict_content[root_name]
            tokenized_content = nltk.word_tokenize(content)
            list_content += tokenized_content

    else :
        print("*** ERROR : unknown tokenization mode= "+str(token_mode))
        return None

    freq_content = nltk.FreqDist(list_content)

    return freq_content
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_tag_count(str_tags):
    """Returns number of tags from a string str_tag, given as parameter.
    str_tags is expected having following format: <tag1><tag2>...<tagN>
    """
    tag_count = 0
    for char_marker in str_tags:
        if char_marker == '<':
            tag_count +=1
    return tag_count
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def taglist_stat_predict(df_corpus, document_index, embedding_mode, vectorizer\
                        , csr_matrix, p_tag_ratio):
    """Returns a predicted list of TAGS from a given row from df_corpus 
    dataframe.
    
    Prediction is based on statistics of terms over the whole corpus.

    The given dataframe, df_corpus, contains corpus of documents.

    Each row from df_corpus contains a document from which returned TAGs are 
    suggested.

    TAGs are extracted from vocabulary handled into vectorizer operator that 
    is given as parameter.
    
    Input : 
        * df_corpus : dataframe that is a corpus represented as a Bag Of Words .
        * document_index : document identifier from corpus (dataframe) from 
        which suggested TAGs will be extracted.
        * vectorizer : operator that has been built in order to embbed documents
         words
        * csr_matrix : digitalized corpus structured as following : 
            csr_matrix[doc_i] = [val_i1,...., val_iN] where : 
                doc_i : is the #i document in the corpus
                val_iX :are values issues from vectorization for N documents 
                in corpus.
                
        * embeding_mode : may be TF-IDF, BOW, COO(co-occurency); this express
        the way csr_matrix has been built.
        
        * p_tag_ratio : ratio of extracted TAGs=number_of_tags/number_of_words
        This allows to compute the number of TAGs to be returned.
    Output :
        * The list of ordered predicted TAGS
        * The list of original TAGS
        * The original document isuue from df_corpus at document_index position.
    
    """
    
    #---------------------------------------------------------------------------
    # Check parameters
    #---------------------------------------------------------------------------
    if 'Body' not in df_corpus.columns :
        print("\n*** ERROR : dataframe corpus must contain column named Body!")
        return None,None,None
    else : 
        pass

    if 'Tags' not in df_corpus.columns :
        print("\n*** ERROR : dataframe corpus must contain column named Tags!")
        return None,None,None
    else : 
        pass
        
    if embedding_mode not in LIST_EMBEDDING_MODE :
        print("\n *** ERROR : embedding mode not supported : "\
        +str(embedding_mode))
        return None,None,None
    else : 
        pass
    
    #---------------------------------------------------------------------------
    # Extract document from corpus thanks to row index.
    #---------------------------------------------------------------------------
    if document_index is not None:
        df_document = df_corpus[df_corpus.index==document_index]
    else : 
        df_document = df_corpus


    #---------------------------------------------------------------------------
    # Document is normalized
    #---------------------------------------------------------------------------
    ser_document_std = p6_df_standardization(df_document.Body, verbose=False)


    #---------------------------------------------------------------------------
    # Extract standardized sentence as a dictionary : {document_id:sentence}
    #---------------------------------------------------------------------------
    dict_sentence = ser_document_std.to_dict()
    sentence_std = dict_sentence[document_index]

    #---------------------------------------------------------------------------
    # Get formated list of original TAGs for this document.
    # Formating consists in changing TAGS under the form <tag1><tag2>...<tagN>
    # into a list under the form [tag1,tag2,...,tagN]
    #---------------------------------------------------------------------------
    str_original_tag=df_document.Tags[document_index]
    list_tag_original=get_list_tag_original(str_original_tag)

    #---------------------------------------------------------------------------
    # Number of TAGs assigned in this document (in this post)
    #---------------------------------------------------------------------------
    tag_original_count = len(list_tag_original)  
    
    #---------------------------------------------------------------------------
    # Extract TAGs from standardized document using vectorizer that contains
    # vocabulary and statistical values upon each document word.
    # Number of TAGs in list depends on tag_ratio criteria. This criteria 
    # applies on number of words in docuement.
    #---------------------------------------------------------------------------
    if embedding_mode == 'tfidf':
        dict_suggested_tag \
        = get_list_tag_stat_tfidf(sentence_std, vectorizer\
        , tag_ratio=p_tag_ratio, tag_original_count=tag_original_count)
    elif embedding_mode == 'bow' or embedding_mode == 'ngram':
        dict_suggested_tag \
        = get_list_tag_stat_ngram(sentence_std, vectorizer\
        , csr_matrix, tag_ratio=p_tag_ratio\
        , tag_original_count=tag_original_count)
    else :
        print("*** ERROR : not supported embedding mode= "+str(embedding_mode))
        return dict(), list_tag_original, str_original_document
    
    #---------------------------------------------------------------------------
    # Record original document as long as original TAG list
    #---------------------------------------------------------------------------
    str_original_document = df_corpus.iloc[document_index].Body
    
    return dict_suggested_tag, list_tag_original, str_original_document
    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_dict_cluster_tag(dict_corpus, vectorizer, cluster_labels\
, cluster_id, p_tag_ratio, is_log=True):
    """Returns a dictionary of suggested tags for a given cluster.
    Input : 
        * dict_corpus : dictionary : {index:document} for the whole corpus
        where document is represented as a standardized sentence composed by 
        words separated by an empty space.
        
        * vectorizer : operator used for corpus vectorization
        * cluster_labels : list of clusters labels
        * cluster_id : a cluster identifier
        * p_tag_ratio : ration of tags considering question lentgh.
    Output : 
        * dictionary for cluster, formated as follow :
            {suggested_tag : value} where value is issued from corpus 
            vectorization. 
        * number of documents in cluster.
    """
    #---------------------------------------------------------------------------
    # Get corpus indexes for any document assigned with the given cluster;
    # This allows to access all documents from dict_corpus belonging to cluster.
    #---------------------------------------------------------------------------
    arr_cluster_index = np.where(cluster_labels==cluster_id)[0]
    
    #---------------------------------------------------------------------------
    # Initialization of list of TAGs relative to the cluster.
    #---------------------------------------------------------------------------
    dict_cluster_tag = dict()

    #---------------------------------------------------------------------------
    # Process any document assigned with cluster_id
    #---------------------------------------------------------------------------
    document_count = len(arr_cluster_index)
    for index in arr_cluster_index :
    
        #-----------------------------------------------------------------------
        # Each document is represented as a sentence structured as a list a 
        # words separated by an empty character.
        #-----------------------------------------------------------------------
        doc_sentence = dict_corpus[index]
        
        #-----------------------------------------------------------------------
        # Get TAGs from document assigned with cluster_id; 
        # we get all N-grams from document.
        # Number of TAGs is limited with p_tag_ratio.
        # Suggested TAG in dictionary are sorted with TF-IDF descendant values.
        #-----------------------------------------------------------------------
        dict_suggested_tag = get_list_tag_stat_tfidf(doc_sentence\
        , vectorizer, tag_ratio=p_tag_ratio)
        
        #-----------------------------------------------------------------------
        # Aggregate into a single list all TAGs from cluster.
        # These aggragated TAGs will caracterized the cluster.
        #-----------------------------------------------------------------------
        if dict_suggested_tag is not None :
            dict_cluster_tag.update(dict_suggested_tag) 

    if is_log is True:
        print("Cluster #"+str(cluster_id)+" : Number of documents= "\
        +str(document_count)+" : Done!")

    return dict_cluster_tag, document_count
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_dict_list_cluster_tag( arr_cluster_label, dict_sof_document, vectorizer\
, p_tag_ratio):
    """Returns dictionaries issued from clusters analysis.
    Input : 
        * arr_cluster_label : array containing clusters identifiers for each row 
        of vectorized corpus.
        This array is structured as following : 
            arr_cluster_label[cluster_id]=[doc_i1, doc_i2,.., doc_iN]
            where #i is the cluster_id and {1,..N} the list of documents 
            belonging the the cluster #i
        
        * dict_sof_document : standardized corpus formated 
        as following : {index:document}
        
        * vectorizer : operator used for corpus vectorization. 
        Handled dictionary of words from which TAG are picked up.
        
        * p_tag_ratio : number of TAGS / number of document terms. 
        It is applied for each document belonging a cluster.
    Output : 
        * dict_list_cluster_tag : a dictionary formated as following :
            {cluster_id : str_cluster_tag_formated} where 
            str_cluster_tag_formated is a sentence with standardized words.
            
        * dict_cluster_stat : a dictionary formated as following: 
            {cluster_id : count} where count is the number of documents found 
            in the cluster.
                
        * dict_df_freq_cluster_tag : a dictionary formated as following :
            { cluster_id: dataframe} where dataframe is structured as follwing :
                dataframe.index : list of words in the cluster cluster_id
                dataframe.Freq : Series containing occurencies of each word 
                referenced from index.
    """
    #---------------------------------------------------------------------------
    # Get all cluster identifiers from arr_cluster_label
    #---------------------------------------------------------------------------
    arr_cluster_id = np.unique(arr_cluster_label)
    
    dict_cluster_stat = dict()
    
    #---------------------------------------------------------------------------
    # For any cluster, retrieve list of TAGs
    #---------------------------------------------------------------------------
    dict_list_cluster_tag = dict()
    dict_df_freq_cluster_tag = dict()
    dict_dict_cluster_tag = dict()
    for cluster_id in arr_cluster_id:
        #-----------------------------------------------------------------------
        # Get, from corpus, all indexes rows assigned with cluster_id
        # Those indexes are documents identifier in the corpus.
        #-----------------------------------------------------------------------
        arr_corpus_index = np.where(arr_cluster_label==cluster_id)[0]

        #-----------------------------------------------------------------------
        # Retrieve TAGs related to cluster_id as a dictionary.
        # The number of returned TAGs is constrained with p_tag_ratio.
        #-----------------------------------------------------------------------
        is_activated = (0==cluster_id%10)
        dict_dict_cluster_tag[cluster_id], dict_cluster_stat[cluster_id] \
        = get_dict_cluster_tag(dict_sof_document, vectorizer\
        , arr_cluster_label, cluster_id, p_tag_ratio, is_log=is_activated)


        #-----------------------------------------------------------------------
        # For each TAG in dict_cluster_tag, tags frequency is computed in order 
        # to filter those tags with greater frequency.
        #-----------------------------------------------------------------------
        list_cluster_tag = dict_dict_cluster_tag[cluster_id].keys()
        freq_cluster_tag = nltk.FreqDist(list_cluster_tag)

        df_freq_cluster_tag \
        = pd.DataFrame.from_dict(freq_cluster_tag, orient="index")
                
        df_freq_cluster_tag.rename(columns={0:'Freq'}, inplace=True)
        df_freq_cluster_tag.sort_values(by=['Freq'], axis=0, ascending=False\
        , inplace=True)

        #-----------------------------------------------------------------------
        # For word cloud display over a cluster
        #-----------------------------------------------------------------------
        dict_df_freq_cluster_tag[cluster_id] = df_freq_cluster_tag

        #-----------------------------------------------------------------------
        # Lets format n-gram for having a more convenenient data to be displayed as 
        # a wordcloud set: any bigram word is aggregated into a single word with a 
        # separation character.
        #-----------------------------------------------------------------------
        list_cluster_tag_formated \
        = [re.sub("[' ']", "_", ngram_tag ) \
        for ngram_tag in df_freq_cluster_tag.index]
        
        str_cluster_tag_formated = ' '.join(list_cluster_tag_formated)
        
        dict_list_cluster_tag[cluster_id] = str_cluster_tag_formated
                
        
    return dict_list_cluster_tag, dict_cluster_stat, dict_df_freq_cluster_tag\
    ,dict_dict_cluster_tag
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_tag_intersect_cluster_list_tag(list_question_key_words\
, dict_cluster_list_tag) :
    """Returns a dictionary formated as following : {cluster_id, tag_count} 
    where tag_count is issued from intersection between list_question_key_words 
    and dict_cluster_list_tag[clster_id], for any cluster_id.
    
    Input : 
        * list_question_key_words : vectorized question
        * dict_list_cluster_tag : dictionary {cluster_id, list_tag} issue from
        a clustering process.
    Output : 
        * A dictionary {cluster_id, tag_count} where,for each cluster_id,  
        tag_count is the number of elements issued from intersection between 
        list_question_key_words and list_tag.
    """

    dict_cluster_intersect_count=dict()
    for cluster_id, list_cluster_tag in dict_cluster_list_tag.items() :
        intersect_count = len(set(list_cluster_tag)&set(list_question_key_words))
        dict_cluster_intersect_count[cluster_id] = intersect_count

    return dict_cluster_intersect_count
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------
# 
#-------------------------------------------------------------------------
def p6_build_dict_dict_index_filter(list_ref, list_list_tag_target):
    index_row=0
    
    dict_filter=dict()
    for list_tag_target in list_list_tag_target:
        #--------------------------------------------------------------
        # list_tag_target is a list of target tags over a given row.
        #--------------------------------------------------------------
        row_tag_filter = np.zeros(len(list_ref), dtype=bool)
        for tag_target in list_tag_target:
            # ------------------------------------------------------------------
            # For a given tag from a row, a filter is built.
            # This filer is issued from condition (np.array(list_ref)==tag_target
            # row_tag_filter is then a list of booleans which size is size of list_ref.
            # This filter is aggregated with all tags from the row, using bitwise operator "|".
            # ------------------------------------------------------------------
            row_tag_filter = (row_tag_filter) | (np.array(list_ref)==tag_target)
        #---------------------------------------------------------------------
        # Filter dictionay for each row is built.
        #---------------------------------------------------------------------        
        dict_filter[index_row] = list(row_tag_filter)
        index_row +=1
    return dict_filter
#-------------------------------------------------------------------------



#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def p6_encode_target(list_tags_ref, list_list_tags):
    """One hot encode the given list_list_tags into values from list_tags_ref.
    Returns a list of encoded values.
    
    """
    dict_index_filter \
    = p6_build_dict_dict_index_filter(list_tags_ref,list_list_tags)
    
    list_all_encoded_row=list()
    for row, index_filter in dict_index_filter.items():
        encoded_row = np.zeros(len(list_tags_ref), dtype=int)
        #-----------------------------------------------------------------------
        # np.where(condition) will return an array of indexes values.
        #-----------------------------------------------------------------------
        index_row_array = np.where(dict_index_filter[row])
        
        #-----------------------------------------------------------------------
        # Row is encoded with value 1 for index values from index_row_array
        #-----------------------------------------------------------------------
        for i in index_row_array:
            encoded_row[i]=1
            
        #-----------------------------------------------------------------------
        # Encoded row is added to list of encoded rows 
        #-----------------------------------------------------------------------
        list_all_encoded_row.insert(row,list(encoded_row))
    return list_all_encoded_row
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def p6_get_tags_from_dict(row, dict_filter, list_all_tags):
    index_array_tag = np.where(np.array(dict_filter[row])==1)
    return np.array(list_all_tags)[index_array_tag]
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def p6_get_list_tag_from_encoded_row(list_list_encoded_row,row,list_all_tags):
    list_encoded_row = list_list_encoded_row[row]
    arr_encoded_index = np.where(np.array(list_encoded_row)==1)
    return np.array(list_all_tags)[arr_encoded_index]
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#
#-------------------------------------------------------------------------
def p6_get_list_all_tag(ser_tag):
    """Returns a list of unique tags from a given Series.
    """
    list_all_tags=list()
    #---------------------------------------------------------------------
    # All tags lists from any row are aggregated as a unique list of tags.
    # Tags may be duplicated onto this list.
    #---------------------------------------------------------------------
    for index, list_tags in ser_tag.items():
        list_all_tags =   list_all_tags+list_tags    

    #---------------------------------------------------------------------
    # Tags are made unique thanks to set()
    #---------------------------------------------------------------------
    list_all_tags = list(set(list_all_tags))
    return list_all_tags
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_encode_target_in_csr_matrix(list_ref_tags, list_tags_to_be_encoded) :
    """Returns a list of list of encoded TAGs that is compressed into a 
    CSR matrix.
    Input : 
        *  list_ref_tags : list of TAGs used as reference for encoding
        *  list_tags_to_be_encoded : list of TAGs to be encoded.
    Output:
        * CSR matrix of encoded TAGs
        Format of returned value is a CSR matrix with expanded list of lists  :
                  ref_1     ref_2         ref_K
            [
       Row 1--->[tag_1.1, tag_1.2, ..... tag_1.K],
                    .
                    .
                    .                
       Row p--->[tag_p.1, tag_p.2, ..... tag_p.K],
                    .
                    .
                    .                
       Row N--->[tag_N.1, tag_N.2, ..... tag_N.K],            
            ]
            
    where : 
        * ref_1,..., ref_K are TAGs named (string type) used as references 
          for encoding process.
        * tag_p_j value is 1 if row p contains TAG ref_j, 0 otherwise. 
    
    """
    #---------------------------------------------------------------------------
    # For each row, TAGs represented as a list of elementaries TAG are encoded.
    # Each row that is a string of TAGs is splitted into a list of elementary 
    # TAGs all rows are aggregated into a list
    #---------------------------------------------------------------------------
    list_list_encoded_row = p6_encode_target(list_ref_tags\
    , list_tags_to_be_encoded)
    
    #---------------------------------------------------------------------------
    # Conversion into CSR matrix for memory optimisation
    #---------------------------------------------------------------------------
    csr_matrix_encoded_tag=sparse.csr_matrix(np.array(list_list_encoded_row))
    
    return csr_matrix_encoded_tag
#-------------------------------------------------------------------------------    
         
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_encode_ser_tag_2_csrmatrix(ser_tag, list_ref_tags, leading_marker='<'\
, trailing_marker='>'):
    """One hot encode the Series named ser_tag that is given as parameter.
    Input : 
        * ser_tag : Series with rows formated as following : 
                    <tag1><tag2>...<tagN>
        * list_ref_tags : list of unique TAGSs to be encoded in.
        * leading_marker : marker format for any TAG
        * trailing_marker : marker format for any TAG
 
    Output :
        * Returned value is a CSR matrix with expanded list  : 
            [
       Row 1   [tag_1.1, tag_1.2, ..... tag_1.K],
                    .
                    .
                    .                
       Row p   [tag_p.1, tag_p.2, ..... tag_p.K],
                    .
                    .
                    .                
       Row N   [tag_N.1, tag_N.2, ..... tag_N.K],            
            ]
            
            where tag_i_j has values 0 or 1.
    """
    #---------------------------------------------------------------------------
    # Markers '<' and '>' are removed from tags.
    #---------------------------------------------------------------------------
    ser_tag = ser_tag.apply(clean_marker_text, leading_marker=leading_marker\
    , trailing_marker=trailing_marker)

    csr_matrix_encoded_tag \
    = p6_encode_target_in_csr_matrix(list_ref_tags, ser_tag.tolist())    
      

    return csr_matrix_encoded_tag, list_ref_tags
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_encode_ser_tag_2_csrmatrix_deprecated(ser_tag, leading_marker='<'\
, trailing_marker='>'):
    """One hot encode the Series named ser_tag that is given as parameter.
    
    As input, each one of the rows from Series content has following 
    format : <tag1><tag2>...<tagN>
    
    Returned value is a CSR matrix with expanded list of lists format : 
        [
   Row 1   [tag_1.1, tag_1.2, ..... tag_1.K],
                .
                .
                .                
   Row p   [tag_p.1, tag_p.2, ..... tag_p.K],
                .
                .
                .                
   Row N   [tag_N.1, tag_N.2, ..... tag_N.K],            
        ]
        
        where tag_i_j has values 0 or 1.
    """
    #---------------------------------------------------------------------------
    # Markers '<' and '>' are removed from tags.
    #---------------------------------------------------------------------------
    ser_tag = ser_tag.apply(clean_marker_text, leading_marker=leading_marker\
    , trailing_marker=trailing_marker)

    #---------------------------------------------------------------------------
    # A unique list of all TAGs is built : this is the vocabulary for TAGs
    # This list is supposed to be completed enough for covering test tags dataset.
    #---------------------------------------------------------------------------
    list_ref_tags = p6_get_list_all_tag(ser_tag)

    csr_matrix_encoded_tag \
    = p6_encode_target_in_csr_matrix(list_ref_tags, ser_tag.tolist())      
    #---------------------------------------------------------------------------
    # For each row, TAGs represented as a list of elementaries TAG are encoded
    # Each row that is a string of TAGs is splitted into a list of elementary TAGs
    # All rows are aggregated into a list
    #---------------------------------------------------------------------------
    # list_list_encoded_row = p6_encode_target(list_all_tags, ser_tag.tolist())

    #---------------------------------------------------------------------------
    # Conversion into CSR matrix fro easyness of computation
    #---------------------------------------------------------------------------
    #csr_matrix_encoded_tag=sparse.csr_matrix(np.array(list_list_encoded_row))

    return csr_matrix_encoded_tag, list_ref_tags
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_extract_list_tag_from_sof_tag(list_sof_tag_1, list_tag_suggested_1 \
    ,score_cutoff=0) :

    """Returns a list of TAGs extracted from a list of referenced TAGs 
    provided by Stack Over FLow.
    Extracted list of TAGs are TAGs from a suggested list of TAGs with greater 
    proximity with TAGs from Stack Over FLow list.
    """
    #---------------------------------------------------------------------------
    # Initialization for avoiding None return.
    #---------------------------------------------------------------------------
    list_extracted_tag = list()

    #---------------------------------------------------------------------------
    # Extract Tags from SOF tags list given the suggested tag list.
    #---------------------------------------------------------------------------
    for tag_suggested in list_tag_suggested_1 :
        tuple_extracted_tag = process.extractOne(tag_suggested, list_sof_tag_1\
        , score_cutoff=score_cutoff)
        if tuple_extracted_tag is not None :
            list_extracted_tag.append(tuple_extracted_tag[0])
        
    return list_extracted_tag
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_score_mean_string_simlarity(nb_test, df_corpus_test, list_sof_tag\
    , vectorizer, csr_matrix,p_tag_ratio=0.1, embeding_mode='bow' ) :
    """Returns mean similarity score for suggested tags.
    
    Suggested TAGs are those suggested from calling a prediction TAG 
    method such as `taglist_stat_predict`. Predictions are performed using 
    vectorizer and csr_matrix parameters.


    Input :
        * nb_test : number of tests to be evaluated
        * df_corpus_test : corpus of documents to be tested
        * vectorizer : operator used for documents vectorization
        * csr_matrix : vectorized documents as a TF_IDF matrix
        * p_tag_ratio : ratio of tags to be suggested and issued from TF-IDF 
          vectorization
        * embeding_mode : 
    Output : 
        * dictionary with, for each document evaluated, under the format:
            -> the mean similarity ratio
               If similarity ratio > 0 : they are more TAGs suggested 
               then original list.
            -> the percentage of matching suggested TAGs considerung original 
                tags.
            -> a dictionary formated as following : 
                {list_original_ag:list_suggested_tag}
            TAGs
        
    """
    idoc=0
    dict_match_result = dict()
    modulo = int(nb_test/10)
    print("\nTest mode {} covering {} documents\n"\
    .format(embeding_mode, nb_test))

    for index_document in range(0, nb_test):
        #-----------------------------------------------------------------------
        # Document is extracted from corpus
        #-----------------------------------------------------------------------
        df_document = df_corpus_test[df_corpus_test.index==index_document]

        #-----------------------------------------------------------------------
        # List of suggested TAGs is returned from TF-IDF matrix; weights from 
        # predicators variables with higher values are selected and related 
        # TAGs are returned. The number of returned TAG depends of p_tag_ratio.
        #-----------------------------------------------------------------------
        list_tag_suggested, list_tag_original, str_original_document \
        = taglist_stat_predict(df_corpus_test, index_document\
                                    , embeding_mode\
                                   , vectorizer, csr_matrix, p_tag_ratio)
            
        #-----------------------------------------------------------------------
        # Similarity of list of suggested TAGs with Stack Over Flow TAGs is 
        # computed using taglist_stat_predict Fuzzy module.
        # Similarity value range from 0 to 100.
        # When similarity value is 100, then evaluated TAGs are considered as 
        # matching. If similarity value is >= 90, then 2 TAGs are considered 
        # as similar.
        #
        # Result is returned in a list of extracted TAGs.
        #-----------------------------------------------------------------------
        if list_tag_suggested is not None:
            if False :
                list_tag_suggested \
                = p6_extract_list_tag_from_sof_tag(list_sof_tag\
                , list_tag_suggested, score_cutoff=90) 
            else :
                #---------------------------------------------------------------
                # Extracted TAGs are filtered against list of SOF TAGs.
                #---------------------------------------------------------------
                list_tag_suggested \
                = list(set(list_tag_suggested).intersection(list_sof_tag))
            
            #-------------------------------------------------------------------
            # For each document, a mean similarity ratio is computed dividing 
            # number of extracted TAGs with number of original TAGs.
            #-------------------------------------------------------------------
            mean_similarity_ratio \
            = (len(list_tag_suggested)/len(list_tag_original))*100
            
            #-------------------------------------------------------------------
            # For each document, the matching count between extracted TAGs 
            # and original TAGs is computed.
            #-------------------------------------------------------------------
            match_count = \
            len(set(list_tag_suggested).intersection(set(list_tag_original)))
        else : 
            mean_similarity_ratio =0.
            match_count=0
            list_tag_suggested=list()

        #-----------------------------------------------------------------------
        # Result is stored into a dictionary along with list of extracted TAGs
        #  and list of original TAGs.
        #-----------------------------------------------------------------------
        dict_element \
        = {'original':list_tag_original,'suggested':list_tag_suggested}
        
        dict_match_result[index_document] = (mean_similarity_ratio\
        , match_count/len(list_tag_original), dict_element)

        #-----------------------------------------------------------------------
        # Tracking computation...
        #-----------------------------------------------------------------------
        idoc += 1
        if modulo > 0 :
            if idoc %modulo == 0 :
                print("Processed documents : {}/{}".format(idoc,nb_test ))
        else:
            print("Processed documents : {}/{}".format(idoc,nb_test ))
                
            
    return dict_match_result
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_stat_compute_result(dict_match_result, verbose=False):

    list_similarity_result = [value[0] for value in dict_match_result.values()]
    arr_similarity_result = np.array(list(list_similarity_result))

    list_matching_result = [value[1] for value in dict_match_result.values()]
    arr_matching_result = np.array(list(list_matching_result))

    #arr_result[0]=1

    if False :
        percent_result \
        = (np.where(arr_similarity_result>=100)[0].shape[0]\
        /len(list_similarity_result))*100
        print("\n*** Mean similarity indice >100: {0:1.2F} %".format(percent_result))

        percent_result \
        = (np.where(arr_similarity_result==0)[0].shape[0]\
        /len(list_similarity_result))*100
        print("\n*** Mean similarity indice = 0: {0:1.2F} %".format(percent_result))


        percent_result = (np.where(arr_matching_result>0)[0].shape[0]\
        /len(list_matching_result))*100
    else :
        percent_result = arr_matching_result.sum()/len(arr_matching_result)


    print("\n*** Matching results : {0:1.2F} %".format(percent_result))
    return arr_similarity_result, arr_matching_result, percent_result
    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_lda_display_topics(lda_model, feature_names, no_top_words, verbose=False):
    """Display topics issued from LDA model.
    Most weighted topics among feature_names list are selected from LDA model.
    
    Input : 
        * lda_model : model issued from LDA .
        * no_top_words : number of words defining a topic 
        * verbose : when activated to True, then words modelizing atopic are 
        displayed.
    Output : 
        * dict_topic : dictionary formated as {topic_id:[list_of_features]}
        where list_of_features is a subset of feature_names selected among the 
        most no_top_words weighted values from lda_model 
        
    """
    #---------------------------------------------------------------------------
    # Length of LDA words representing topics is used to limit 
    #---------------------------------------------------------------------------
    len_feature_names = len(feature_names)
    dict_topic = dict()
    
    
    for topic_idx, topic in enumerate(lda_model.components_):
        if verbose is True :
            message = "Topic %d: " % (topic_idx)
            message += " / ".join([lda_feature_names[i] \
                               for i in topic.argsort()[:-no_top_words - 1:-1] \
                               if i<len_feature_names])
            print(message)
        else :
            pass
            
        dict_topic[topic_idx] = [feature_names[i] \
        for i in topic.argsort()[:-no_top_words - 1:-1] if i<len_feature_names]
    return dict_topic
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_lda_get_topic_from_list_word(dict_lda_topic, list_of_word):
    """Returns a dictionary containing with following format :
    {topic_id:(count,list_matching_word)} where : 
    * topic_id : is the key from dict_lda_topic identifying a topic.
    * count : number of elements from intersection.
    * list_matching_word : is issued from intersection of a given 
    list_of_word and words contained into dict_lda_topic

    Input : 
        * dict_lda_topic : dictionary with format : {topic_id:[list_of_word_topic]}
        * list_of_word : list of words to be matched with list_of_word_topic
    Output : 
        * dict_topic_result : a subset of dict_lda_topic (see above)
    
    """
    dict_topic_result = dict()
    for topic_id, list_lda_word in dict_lda_topic.items():
        list_intersection = list(set(list_of_word).intersection(list_lda_word))
        intersection_count = len(list_intersection)
        dict_topic_result[topic_id] = (intersection_count,list_intersection)
    return dict_topic_result
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_lda_mean_score_post(post, tags, dict_lda_topic, list_sof_tags\
, verbose=False, dict_expert_tag=None,list_train_tags=None):
    """ For a given POST assigned with given TAGa, this function returns mean 
    accuracy score for suggested TAGs issued from LDA process.
    Mean accuracy score is computed by dividing :
        the number of elements from intersection of TAGs issued from a POST and LDA
        with the number of TAGs from a given POST.
    
    """
    dict_result = dict()
    mean_score = 0.
    
    #---------------------------------------------------------------------------
    # POST is standadardized
    #---------------------------------------------------------------------------
    ser_post = p6_str_standardization(post)

    #---------------------------------------------------------------------------
    # Words from any LDA topic matching with list of words 
    # issued from POST are extracted from dict_lda_topic
    #---------------------------------------------------------------------------
    list_post_word = ser_post.iloc[0].split()
    dict_topic_result \
    = p6_lda_get_topic_from_list_word(dict_lda_topic, list_post_word)

    #---------------------------------------------------------------------------
    # Dictionary is converted into a dataframe with 2 columns : Count and Words.
    #---------------------------------------------------------------------------
    df_topic_result = pd.DataFrame.from_dict( dict_topic_result, orient='index')
    df_topic_result.rename(columns={0:'Count', 1:'Words'}, inplace=True)
    
    #---------------------------------------------------------------------------
    # Get topics having count >= 1
    #---------------------------------------------------------------------------
    df_topic_result_pos = df_topic_result[df_topic_result.Count>=1]

    #---------------------------------------------------------------------------
    # Building the list of unique words belonging to LDA topics.
    #---------------------------------------------------------------------------
    list_all_word = list()
    for list_word in df_topic_result_pos.Words.tolist() :
        list_all_word += [word for word in list_word]
        
    list_word_result = list(set(list_all_word))

    #---------------------------------------------------------------------------
    # TAGs are remapped over referenced TAGr using fuzzy-wuzzy measurement
    # Referenced TAG are list of assigned TAG from train dataset.
    #---------------------------------------------------------------------------
    if list_train_tags is not None :
        list_tag_pred = list()
        for word in list_word_result :
            list_tuple_score = process.extract(word, list_train_tags)
            list_tag_pred += [tuple_score[0] for tuple_score \
            in list_tuple_score if tuple_score[1] >= 90]
        list_word_result = list_tag_pred.copy()

    #---------------------------------------------------------------------------
    # Assigned TAGs from POST are converted into a list of TAGs.
    #---------------------------------------------------------------------------
    list_assigned_tag \
    = clean_marker_text(tags,leading_marker='<', trailing_marker='>')

    #---------------------------------------------------------------------------
    # TAGs issued from LDA are filtered with SOF TAGs list.
    #---------------------------------------------------------------------------
    if list_train_tags is None :
        list_intersection_sof \
        = list(set(list_word_result).intersection(list_sof_tags))
    else :
        list_intersection_sof = list_word_result.copy()

    #---------------------------------------------------------------------------
    # Building intersection between list of TAGs issued from LDA model and 
    # list of TAGs from POST.
    #---------------------------------------------------------------------------
    if False :
        if len(list_intersection_sof) > 0 :
            list_intersection_result \
            = list(set(list_assigned_tag).intersection(list_intersection_sof))    
        else :
            list_intersection_result \
            = list(set(list_assigned_tag).intersection(list_word_result))
    else : 
        list_intersection_result \
            = list(set(list_assigned_tag).intersection(list_intersection_sof))
    
    score_accuracy = len(list_intersection_result)/len(list_assigned_tag)
    
    if verbose is True :
        print("\nList of assigned TAG from POST : "+str(list_assigned_tag))
        print("\nList of suggested TAG : "+str(list_word_result))

        print("\nList intersection SOF : "+str(list_intersection_sof))
        print("\nList intersection result : "+str(list_intersection_result))
        print("\nAccuracy = "+str(dict_result[post_id]))
        print()
        print(post)
    return score_accuracy    
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_lda_build_range(range_topic,embedding_type, csr_matrix, rangeName=None):
    for nb_topic in range_topic :
        if rangeName is None :                
            print("Building LDA model with "+str(nb_topic)+" topics")
            file_name \
            = "./data/lda_"+embedding_type+"_"+str(nb_topic)+"topics.dump"
        else : 
            print("Building LDA model with "+str(nb_topic)+" topics / "+rangeName)
            file_name \
            = "./data/lda_"+embedding_type+"_"+str(nb_topic)+"topics_"\
            +rangeName+".dump"
        # Training LDA model 
        lda = LatentDirichletAllocation(n_topics=nb_topic, max_iter=5\
                                        , learning_method='online'\
                                        , learning_offset=50.\
                                        ,random_state=0).fit(csr_matrix)
        p5_util.object_dump(lda,file_name)

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_lda_range_mean_score(nb_test, range_topic, embedding_type, df_sof_test\
                        ,list_sof_tags, vectorizer, nb_top_words = 10\
                        , rangeName = None,dict_expert_tag=None,is_dumped=True\
                        ,list_train_tags=None) :
    """
        Input : 
            * nb_test : for each LDA model, number of POSTs used for accuracy.
            * range_topic :  range of topics used to build LDA models.
            This mean, LDA models with 10, 100... topics each.
            * embedding_type :  BOW, TF-IDF, ... : embedding algorithm.
            * df_sof_test :  test data-set
            * list_sof_tags : list of the whole Stack Over Flow (SOF) TAGs 
            issued from SOF database.
            * vectorizer : operator used for words embedding. It allows to 
            access predictive features list.
            * nb_top_words : max number of features for each topic that will be 
            selected from LDA words characterizing a topic.
        Output : 
            A dictionary that is dumped into a file named 
            `dict_score_lda_<embedding_type>_<nb_topic>` where `nb_topic` 
            belongs to `range_topic`.
    """                        
                        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------    
    feature_names = vectorizer.get_feature_names()
    
    for nb_topic in range_topic :
        #-----------------------------------------------------------------------
        # Load, from dumped file, LDA model built for each topic range.
        # Such models have been built and dumped from within function 
        # `p6_lda_build_range`
        #-----------------------------------------------------------------------
        if rangeName is None :
            file_name = \
            "./data/lda_"+str(embedding_type)+"_"+str(nb_topic)+"topics.dump"
        else :
            file_name \
            = "./data/lda_"+embedding_type+"_"+str(nb_topic)+"topics_"\
            +rangeName+".dump"
            
        lda = p5_util.object_load(file_name)

        #-----------------------------------------------------------------------
        # Accuracy is computed by randomly selecting nb_test POSTs from 
        # Test dataset.
        #-----------------------------------------------------------------------
        dict_score_lda = dict()
        for i_test in range(0,nb_test):
            post_id = random.choice(range(0, df_sof_test.shape[0]))

            #-------------------------------------------------------------------
            # Build a POST from Body and Title, extract assigned TAGs.
            #-------------------------------------------------------------------
            body  = df_sof_test.Body.iloc[post_id]
            title = df_sof_test.Title.iloc[post_id]
            tags  = df_sof_test.Tags.iloc[post_id]
            post  = body+title

            #-------------------------------------------------------------------
            # LDA model (that handles weights) is used in order to extract, 
            # for each topic,  the most weighted features from feature_names.
            # By the way, nb_top_words are extracted from each topic.
            #-------------------------------------------------------------------
            dict_lda_topic \
            = p6_lda_display_topics(lda, feature_names, nb_top_words)

            mean_score = p6_lda_mean_score_post(post, tags, dict_lda_topic\
            , list_sof_tags, verbose=False, dict_expert_tag=dict_expert_tag\
            ,list_train_tags=list_train_tags)
            dict_score_lda[post_id] = mean_score
        
        #-----------------------------------------------------------------------
        # Dump dictionary into a file containing mean score for any POST.
        #-----------------------------------------------------------------------
        if is_dumped is True :
            if rangeName is None :
                fileName \
                = "./data/dict_score_lda_"+str(embedding_type)\
                +"_"+str(nb_topic)+".dump"
            else :
                fileName \
                = "./data/dict_score_lda_"+str(embedding_type)\
                +"_"+str(nb_topic)+"_"+rangeName+".dump"
            p5_util.object_dump(dict_score_lda,fileName)    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_w2vec_mean_score(w2vec_model, df_sof_test, list_sof_tags, vectorizer\
                        , nb_top_words = 10) :
    for i_post in range(0,len(df_sof_test)) :
    
        #-------------------------------------------------------------------
        # Build a POST from Body and Title, extract assigned TAGs.
        #-------------------------------------------------------------------
        body  = df_sof_test.Body.iloc[post_id]
        title = df_sof_test.Title.iloc[post_id]
        tags  = df_sof_test.Tags.iloc[post_id]
        post  = body+title
        
        #-------------------------------------------------------------------
        # Get top words 
        #-------------------------------------------------------------------
        

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_lda_build_accuracy_result(embedding_type, range_topic ) :
    """This function loads dictionaries of results that have been dumped 
    on mass storage per LDA model.
    Once loaded, a mean accuracy value is computed per LDA model.
    """
    dict_lda_mean_accuracy = dict()
    
    if str(embedding_type) == 'bow' :
        pass
    elif str(embedding_type) == 'tfidf' :
        pass
    else :
        print("\n*** ERROR : unknow embredding type = "+str(embedding_type))
        return None

    for nb_topic in range_topic:
        fileName \
        = "./data/dict_score_lda_"+str(embedding_type)+"_"+str(nb_topic)+".dump"
        dict_score_lda = p5_util.object_load(fileName)

        sumScore=0.
        for key in dict_score_lda.keys():
            sumScore += dict_score_lda[key]
        lda_mean_accuracy = 100*sumScore/len(dict_score_lda)

        dict_lda_mean_accuracy[nb_topic] = lda_mean_accuracy

    return dict_lda_mean_accuracy
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_get_dict_row_col_from_csrmatrix(csrmatrix) :
    ''' Get a dictionary formated as following : {row:list_col}
    where :
        * row is the row number of CSR matrix given as parameter function
        * list_col : is the list of values in the row of CSR matrix 
        given as parameter function.
    '''
    tuple_index = np.where(csrmatrix.A >0)
    #print('p6_get_dict_row_col_from_csrmatrix')
    dict_row_col = dict()
    for row,col in zip(tuple_index[0],tuple_index[1]):
        if row in dict_row_col.keys() :
            dict_row_col[row].append(col)
        else :
            dict_row_col[row] = list()
            dict_row_col[row].append(col)
        
    return dict_row_col
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_supervized_mean_accuracy_score(y_true, y_pred\
    , mode_match='intersection_matching'\
    , encoder=None, arr_encoded_filter=None):
    ''' Computes, for supervized models,mean accuracy score as following : 
    for each row from y_true vector, number of TAGs issue from 
    intersection between y_true and y_pred is cumulated.
    The total sum is devided by the cumulated sum of TAGs issued from 
    y_true.
    Input : 
        * y_true : true labels
        * y_pred : predicted labels.
        * mode_match : intersection_matching or fuzzy_matching
        In the first case, TAG issued from prediction matche with assigned 
        TAG if TAG are exactly same.
        In the second case, TAG issued from prediction matche with assigned
        TAG if fuzzy similarity score between TAG if >= 95.
        * encoder : labelizer used if matching mode is fuzzy_matching
        * arr_encoded_filter : allows to match values with string inssued from 
        encoder

        Note that intersection_matching is much more coercitive then other mode.
    '''
    #---------------------------------------------------------------------------
    # For y_true and y_pred, dictionaries results are computed under the format :
    # {row:[col1,...,colK]} where colX is the column number in the referenced 
    # list of TAGs issued from one hot encoding operation.
    #---------------------------------------------------------------------------
    dict_row_col_true = p6_get_dict_row_col_from_csrmatrix(y_true)
    
    dict_row_col_pred = p6_get_dict_row_col_from_csrmatrix(y_pred)

    count_tag_match = 0
    tag_row_count = 0
    if(mode_match == 'intersection_matching') :
        #-----------------------------------------------------------------------
        # Compute, from each row, intersection of TAGs.
        #-----------------------------------------------------------------------
        for row, list_col_true in dict_row_col_true.items() :
            if row in dict_row_col_pred.keys():
                # Row still contains a list; 
                list_col_pred = dict_row_col_pred[row]
                count_tag_match \
                += len(set(list_col_true).intersection(list_col_pred))
            else : 
                pass
            tag_row_count += len(list_col_true)
        #print(tag_row_count)

    
    elif mode_match == 'fuzzy_matching' :
        for row, list_col_true in dict_row_col_true.items() :
            if row in dict_row_col_pred.keys() :
                list_col_pred = dict_row_col_pred[row]
                for token_pred in list_col_pred :
                    row = np.where(arr_encoded_filter==token_pred)[0][0]
                    str_token_pred = encoder.classes_[row]
                    list_str_col_true=list()
                    for col_true in list_col_true :
                        row = np.where(arr_encoded_filter==col_true)[0][0]
                        list_str_col_true.append(encoder.classes_[row])

                    list_tuple_score = process.extract(str_token_pred\
                    , list_str_col_true)

                    for tuple_score in list_tuple_score :
                        if 95<= tuple_score[0] :
                            count_tag_match += 1
                        else :
                            pass
                            
            else :
                pass
            tag_row_count += len(list_col_true)
    else:
        print("*** ERROR : matching mode= "+mode_match+" NOT SUPPORTED!")
    
    mean_score = count_tag_match/tag_row_count 
    return mean_score      
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p6_w2vec_mean_accuracy(word2vec_model, df_sof_test, ratio=0.1):
    accuracy = 0.0
    if ratio is None :
        nb_test = len(df_sof_test)
    else :
        ratio = min(ratio, 1.)
        nb_test = int( len(df_sof_test)*ratio )
        if nb_test == 0 :
            ratio = 0.1
            nb_test = int(len(df_sof_test)*ratio)
            print("*** WARNING : ratio for total test is fixed to value= "+str(ratio))
        else :
            pass
    print("\n*** INFO : NB test= "+str(nb_test))
    for post_id in range(0,nb_test) :
        #post_id = random.choice(range(0, nb_test))
        #print(post_id)

        body  = df_sof_test.Body.iloc[post_id]
        title = df_sof_test.Title.iloc[post_id]
        tags  = df_sof_test.Tags.iloc[post_id]
        post  = body+title

        #-----------------------------------------------------------------------
        # POST is standadardized
        #-----------------------------------------------------------------------
        ser_post = p6_str_standardization(post)
        list_post_word = ser_post.tolist()[0].split()

        list_computed_tag \
        = word2vec_model.predict_output_word(list_post_word, topn=10)

        if list_computed_tag is not None :
            list_suggested_tag = [tuple_computed_tag[0] \
            for tuple_computed_tag in list_computed_tag]

            list_tags \
            = clean_marker_text(tags,leading_marker='<', trailing_marker='>')

            list_match = list(set(list_tags).intersection(list_suggested_tag))
            accuracy \
            += len(set(list_tags).intersection(list_suggested_tag))/len(list_tags)

    return accuracy/nb_test
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def ser_corpus_2_df_word_count(ser_corpus, token_mode='nltk'):
    """Converts a Series containing sentences from corpus into a Dataframe 
    containing words from sentences and number of words.
    Input : 
        * ser_corpus Series from which, each index contains a list of tokens 
        named as a sentence.
        * token_mode : tokenization type; sentences are splitted as list of 
        words.
    Output : 
        * DataFrame with 2 columns : Word, Count.
    """
    
    dict_corpus = ser_corpus.to_dict()
    freq_token\
    = compute_frequency_sentence(dict_corpus, token_mode=token_mode)

    dict_word = dict()
    dict_count = dict()
    index =0
    for tuple_item in freq_token.items():
        dict_word[index] = tuple_item[0]
        dict_count[index] = tuple_item[1]
        index += 1

    df_word = pd.DataFrame.from_dict( dict_word, orient='index')
    df_count = pd.DataFrame.from_dict( dict_count, orient='index')

    df_word_count = pd.DataFrame({'Word':df_word[0],"Count":df_count[0]})
    return df_word_count

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def w2vec_build_dict_post_tag(w2vec_model , df_corpus, nb_test, topn=10):
    """ Build a dicitonary formated as following : {doc_id:[list_word_suggested]}
    
    Input :
        * w2vec : Word2Vec model used to build a list of suggested words
        * df_corpus : dataframe containing documents from which W2VEC model will
        compute suggested words.
        * topn : number of words picked in w2vec model for each sentence.
    Output
        * dictionary formated as  {doc_id:[list_word_suggested]}
        
    """
    dict_doc_w2vec_word = dict()
    nb_fail = 0
    for i_test in range(0,nb_test):
        post_id = random.choice(range(0, df_corpus.shape[0]))

        #-------------------------------------------------------------------
        # Build a POST from Body and Title, extract assigned TAGs.
        #-------------------------------------------------------------------
        body  = df_corpus.Body.iloc[post_id]
        title = df_corpus.Title.iloc[post_id]
        post  = body+title

        #-----------------------------------------------------------------------
        # POST is standadardized
        #-----------------------------------------------------------------------
        ser_post = p6_str_standardization(post)

        #-----------------------------------------------------------------------
        #
        #-----------------------------------------------------------------------
        list_post_word = ser_post.iloc[0].split()

        #-----------------------------------------------------------------------
        # Words with highest weights are picked up from model
        #-----------------------------------------------------------------------
        list_tag_weight_suggested \
        = w2vec_model.predict_output_word(list_post_word, topn=topn)

        #-----------------------------------------------------------------------
        # Words list dictionay is built foe each document.
        #-----------------------------------------------------------------------
        if(list_tag_weight_suggested is None):
            print("POST= "+str(post_id)+" Nb suggested TAG= None")
            nb_fail +=1
        else : 
            dict_doc_w2vec_word[post_id] \
            =[tuple_result[0] for tuple_result \
            in list_tag_weight_suggested if list_tag_weight_suggested is not None]


    return dict_doc_w2vec_word, nb_fail
#-------------------------------------------------------------------------------

