import numpy as np
import pandas as pd
import numpy.ma as ma
import os

import glob
import dill

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn import tree

from sklearn import  cluster

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from scipy import sparse
from scipy.sparse.csr import csr_matrix
import scipy.sparse

from sklearn.metrics import silhouette_score
from sklearn import metrics

import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk import text
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA

import string


from datetime import datetime
import time
import pickle 
import zlib

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p5_get_pod_from_timestamp(timestamp):
   """Extract and returns period of day from timestamp value that is given as 
   input parameter.

   Period of day got 2 values : 
      * 1 :  7:00 AM < Period <= 19:00 AM
      * 2 : 19:00 PM < Period <= 07:00 AM

   Input : 
      * timestamp time formated as : %Y-%m-%d %H:%M:%S (2011-07-06 13:14:00)
   Output :
      * day as integer
   """
   
   str_timestamp=str(timestamp)
    
   my_date=datetime.strptime(str_timestamp, "%Y-%m-%d %H:%M:%S")
   hour=my_date.timetuple().tm_hour
   mn  =my_date.timetuple().tm_min
   sec =my_date.timetuple().tm_sec
   
   hourmin=hour*60+mn
   
   if 7.0*60 < hourmin <= 19.0*60:
      pod=1
   else : 
      pod=2
   return pod
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p5_get_dow_from_timestamp(timestamp):
   """Extract and returns day from timestamp value that is given as 
   input parameter.
   
   Input : 
      * timestamp time formated as : %Y-%m-%d %H:%M:%S (2011-07-06 13:14:00)
   Output :
      * day as integer
   """
   
   str_timestamp=str(timestamp)
    
   my_date=datetime.strptime(str_timestamp, "%Y-%m-%d %H:%M:%S")
   wday=my_date.timetuple().tm_wday
   return wday
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p5_get_day_from_timestamp(timestamp):
   """Extract and returns day from timestamp value that is given as 
   input parameter.
   
   Input : 
      * timestamp time formated as : %Y-%m-%d %H:%M:%S (2011-07-06 13:14:00)
   Output :
      * day as integer
   """
   str_timestamp=str(timestamp)
    
   my_date=datetime.strptime(str_timestamp, "%Y-%m-%d %H:%M:%S")
   day=my_date.timetuple().tm_mday
   return day
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p5_get_month_from_timestamp(timestamp):
   """Extract and returns month from timestamp value that is given as 
   input parameter.
   
   Input : 
      * timestamp time formated as : %Y-%m-%d %H:%M:%S (2011-07-06 13:14:00)
   Output :
      * month as integer
   """
   str_timestamp=str(timestamp)
    
   my_date=datetime.strptime(str_timestamp, "%Y-%m-%d %H:%M:%S")
   month=my_date.timetuple().tm_mon
   return month
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p5_get_year_from_timestamp(timestamp) :
   """Extract and returns year from timestamp value that is given as 
   input parameter.
   
   Input : 
      * timestamp time formated as : %Y-%m-%d %H:%M:%S (2011-07-06 13:14:00)
   Output :
      * year as integer
   """
   str_timestamp=str(timestamp)
    
   my_date=datetime.strptime(str_timestamp, "%Y-%m-%d %H:%M:%S")
   year=my_date.timetuple().tm_year
   return year
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p5_convert_timestamp_to_sec(timestamp) :
   str_timestamp=str(timestamp)
   my_date=datetime.strptime(str_timestamp, "%Y-%m-%d %H:%M:%S")
   my_seconds=time.mktime(my_date.timetuple())
   return my_seconds
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p5_convert_timestamp_to_days(timestamp):
   """Returns number of days since Unix epoch time 
      Input : 
      * timestamp : Timestamp format
      Output :
      * Number of days from Unix EPOCH 
   """
   my_seconds=p5_convert_timestamp_to_sec(timestamp)
   my_days=my_seconds/(3600*24)
   return round(my_days)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p5_score_r2_linear_regression(df, feature,model_name='LinearRegression') :


    df_droped=df.drop([feature],inplace=False,axis=1)
    df_y=pd.DataFrame(df[feature], index= df_droped.index)

    df_droped=df_droped.astype(float)
    X_std=df_droped.values
    std_scale=preprocessing.StandardScaler().fit(X_std)
    X_std=std_scale.transform(X_std)
    y=df_y.values

    X_train_std, X_test_std, y_train, y_test=\
    model_selection.train_test_split(X_std, y, test_size=0.25,random_state=42)
    
    if model_name  == 'LinearRegression':
        regresion_model=linear_model.LinearRegression()
    else :
        model_name='DecisionTreeRegressor'
        regresion_model =tree.DecisionTreeRegressor(random_state=42)
        
    regresion_model.fit(X_train_std, y_train.ravel())

    r2_score=regresion_model.score(X_test_std, y_test)
    print("Model=\'{0:s}\' : R2 score for variable \'{1:s}\' \
   ={2:0.3f}".format(model_name, feature,r2_score))    
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def func_rfm_scoring(rfm_value,rfm_threshold,rfm_column) :
   """Compute RFM classification from a given value, depending on 
   rfm_threshold matrix.
   
   RFM score from a value is computed as following : 
   
      +------------------+-----+                  
      |      rfm_value   | RFM |
      +------------------+-----+                 
      |      value <= Q1 | 4   |
      +------------------+-----+
      | Q1 < value <= Q2 | 3   |
      +------------------+-----+
      | Q2 < value <= Q3 | 2   |
      +------------------+-----+
      | Q3 < value       | 1   |
      +------------------+-----+
   
   Input : 
      * rfm_value : value to be classified into a RFM score against 
      rfm_threshold
      * rfm_threshold : threshold matrix 
      * rfm_column : column from rfm_threshold from which threshold are 
      extracted.
   Output : 
      * rfm_class : RFM class value belongs to.
      
   """
   
   sorted_list=sorted(rfm_threshold[rfm_column])
   Q1=sorted_list[0]
   Q2=sorted_list[1]
   Q3=sorted_list[2]
   
   if rfm_value <= Q1 :
     rfm_class=4
   elif Q1 < rfm_value <= Q2 :
     rfm_class=3
   elif Q2 < rfm_value <= Q3 :
     rfm_class=2
   else:
     rfm_class=1
   
   return rfm_class
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def tsne_X_dim_process_perplexity(X, tsne_iter=3000\
    , dim = 2\
   ,list_tsne_perplexity=[var for var in range(5,55,5)]) :
   
   """Builds data points reduction in X dimensions set using tSNE algorihtm.
   This set is leaded by perplexity values ranged from 5 to 50.
   
   Input : 
    *   X_std_sample standardized array of N rows and C columns.
    *   dim : numbers of dimensions issued from t-SNE processing.
    *   list_tsne_perplexity : list of t-SNE perplexities values.
    
   Return : 
    *   dictionary with perplexity as keys and 2D reduced points as values.
   """

   dict_tsne_result=dict()

   for tsne_perplexity in list_tsne_perplexity :
      print("tSNE perplexity : "+str(tsne_perplexity)+"/50 ...")
      manifold_embedd=TSNE(n_components=dim\
      ,n_iter=tsne_iter,perplexity=tsne_perplexity)
      
      dict_tsne_result[tsne_perplexity]=manifold_embedd.fit_transform(X)

   return dict_tsne_result 
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def tsne_2D_process_perplexity(X, tsne_iter=3000\
   ,list_tsne_perplexity=[var for var in range(5,55,5)]) :
   
   """Builds data points reduction set thanks to tSNE algorihtm.
   This set is leaded by perplexity values ranged from 5 to 50.
   
   Input : 
    *   X_std_sample array of N rows and C columns
   Return : 
    *   dictionary with perplexity as keys and 2D reduced points as values.
   """
   nb_components=2
   dict_tsne_result=dict()

   for tsne_perplexity in list_tsne_perplexity :
      print("tSNE perplexity : "+str(tsne_perplexity)+"/50 ...")
      manifold_embedd=TSNE(n_components=nb_components\
      ,n_iter=tsne_iter,perplexity=tsne_perplexity)
      
      dict_tsne_result[tsne_perplexity]=manifold_embedd.fit_transform(X)

   return dict_tsne_result 
#-------------------------------------------------------------------------------
   
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def kmeans_scan_silhouette(df, n_cluster_start, n_cluster_end) : 
   """ Scan and compute silhouettte score for a range of Kmeans clusters 
   over given dataframe values.
   Input : 
      df : given dataframe containing values on which clustering applies
      n_cluster_start : minimum number of clusters
      n_cluster_end : maximum number odf clusters
   Output :
      dict_cluster_scoring : a dictionary with keys as number of clusters and
      values as silhouette scoring.
   """
   dict_cluster_scoring=dict()
   csr_matrix=sparse.csr_matrix(df.values)

   if 1 == n_cluster_start :
      n_cluster_start=2

   for n_cluster in range(n_cluster_start, n_cluster_end, 1):
      # Apply your clustering algorithm 
      cluster_kmean=KMeans(n_clusters=n_cluster).fit(csr_matrix)

      # Predict the cluster for each data point
      preds_kmean=cluster_kmean.predict(csr_matrix)

      # Find the cluster centers
      centers_kmean=cluster_kmean.cluster_centers_

      # Calculate the mean silhouette coefficient for each cluster
      dict_cluster_scoring[n_cluster] \
     =silhouette_score(csr_matrix, preds_kmean)
      
      print("Cluster "+str(n_cluster)+" done!")
   return dict_cluster_scoring
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def gmm_scan_score(X, n_cluster_start, n_cluster_end\
                  , p_covariance_type='full') : 
   """ Scan and compute silhouettte score for GaussianMixture models 
   applied over datapoints.

   Defaut hyper-parameter used : covariance_type='full'; axis orientation for 
   each cluster is arbitrary.
   
   Input : 
      X : given datapoints on which GMM models scan is applied.
      n_cluster_start : minimum number of clusters
      n_cluster_end : maximum number odf clusters

   Output :
      dict_cluster_scoring : a dictionary with keys as number of clusters and
      values as silhouette scoring.
   """
   dict_cluster_scoring=dict()
   for n_cluster in range(n_cluster_start,n_cluster_end,1):

      # Apply your clustering algorithm
      cluster_gmm=GaussianMixture(n_components=n_cluster\
      , covariance_type=p_covariance_type).fit(X)

      # Predict the cluster for each data point
      preds_gmm=cluster_gmm.predict(X)

      # Calculate the mean silhouette coefficient for the number 
      # of clusters chosen
      dict_cluster_scoring[n_cluster]=silhouette_score(X, preds_gmm)
   return dict_cluster_scoring
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def scan_best_kmean_cluster_from_tsne_result(dict_tsne_result, \
    n_cluster_start, n_cluster_end):
    """Returns best number of clusters from Kmeans algorithm and best 
    perplexity value from tSNE algorithm. 
    Input :
      dict_tsne_result : dictionary containing perplexity values as keys and 
      data points reduced with t-SNE as values.
      n_cluster_start : start point for cluster scanning process.
      n_cluster_end : end point for cluster scanning process.
    Output : 
      n_cluster_optimum : number of Kmeans clusters providing the best \
      silhouette score.
      best_perplexity : perplexity key value from dict_tsne_result leading to 
      the best score.
      score_max : the best silouhette score for any perplexity value.
    """
    dict_perplexity_cluster_scoring=dict()
    for perplexity, X_std_projected in dict_tsne_result.items():
        print("Perplexity= "+str(perplexity)+\
        " : Kmeans clustering from "+str(n_cluster_start)+" to "\
        +str(n_cluster_end))
        dict_perplexity_cluster_scoring[perplexity]=\
        kmean_scan_score(X_std_projected, n_cluster_start, n_cluster_end)
    
    score_max=0
    cluster_optimum=0
    best_perplexity=-1
    
    for perplexity, dict_cluster_scoring \
    in dict_perplexity_cluster_scoring.items():
    
        for n_cluster in dict_cluster_scoring.keys() :
            score=dict_cluster_scoring[n_cluster]
            if score > score_max : 
                score_max=score
                n_cluster_optimum=n_cluster
                best_perplexity=perplexity
    
    return n_cluster_optimum,best_perplexity,score_max
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def scan_best_gmm_cluster_from_tsne_result(dict_tsne_result, \
    n_cluster_start, n_cluster_end):
    """Returns best number of clusters from GMM algorithm and best 
    perplexity value from tSNE algorithm. 
    Input :
      dict_tsne_result : dictionary containing perplexity values as keys and 
      data points reduced with t-SNE as values.
      n_cluster_start : start point for cluster scanning process.
      n_cluster_end : end point for cluster scanning process.
    Output : 
      n_cluster_optimum : number of GMM clusters providing the best \
      silhouette score.
      best_perplexity : perplexity key value from dict_tsne_result leading to 
      the best score.
      score_max : the best silouhette score for any perplexity value.
    """
    
    dict_perplexity_cluster_scoring=dict()
    for perplexity, X_std_projected in dict_tsne_result.items():
        print("Perplexity= "+str(perplexity)+\
        " : GMM clustering from "+str(n_cluster_start)\
        +" to "+str(n_cluster_end))
        
        dict_perplexity_cluster_scoring[perplexity]=\
        gmm_scan_score(X_std_projected, n_cluster_start, n_cluster_end)
    
    score_max=0
    cluster_optimum=0
    best_perplexity=-1
    
    for perplexity, dict_cluster_scoring \
    in dict_perplexity_cluster_scoring.items():
    
        for n_cluster in dict_cluster_scoring.keys() :
            score=dict_cluster_scoring[n_cluster]
            if score > score_max : 
                score_max=score
                n_cluster_optimum=n_cluster
                best_perplexity=perplexity
    
    return n_cluster_optimum,best_perplexity,score_max
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_my_punctuation():
    my_punctuation=str()
    list_punctuation_excluded=['-','_',' ']
    for char in string.punctuation:
        if char in list_punctuation_excluded:
            pass
        else:
            my_punctuation += char
    return my_punctuation
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_digit(item_description) :
   """ Remove digits from item_str given as paraemeter.
   This callback is called when building new features from items descriptions.
   Input : 
      * item_description : description of an item from an invoice line.
   Output :
      * item description free od digits
   """
   
   list_item_no_digit \
  =[item for item in item_description if not item.isdigit()]
   
   item_free_digit=''.join(list_item_no_digit)
   return item_free_digit.upper()
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_punctuation(item, list_char_remove) :
    item_no_punctuation=[ char for char in item.lower() \
    if char not in list_char_remove ]

    item_no_punctuation="".join(item_no_punctuation)
    return item_no_punctuation.upper()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_remove_stopwords(item) :
    list_word=item.split()
    item_no_stopwords=[ word for word in list_word if word.lower() \
    not in stopwords.words('english') ]

    item_no_stopwords=" ".join(item_no_stopwords)
    return item_no_stopwords.upper()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_stemmer(item, stemmer, mode='lower'):
    list_word=item.split()
    stemmed_item=[ stemmer.stem(word.lower()) for word in list_word ]
    stemmed_item=" ".join(stemmed_item)
    if 'upper' == mode : 
        stemmed_item = stemmed_item.upper()
    elif 'lower' == mode : 
        stemmed_item = stemmed_item.lower()
    else :
        stemmed_item = stemmed_item.lower()
    return stemmed_item
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_lemmatizer(item, lemmatizer, mode='lower'):
    list_item=item.split()
    lemmatized_item=[ lemmatizer.lemmatize(word.lower()) \
    for word in list_item ]
    
    lemmatized_item=" ".join(lemmatized_item)
    if 'upper' == mode : 
        lemmatized_item = lemmatized_item.upper()
    elif 'lower' == mode : 
        lemmatized_item = lemmatized_item.lower()
    else :
        lemmatized_item = lemmatized_item.lower()
    
    return lemmatized_item
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_clean_numeric_word_in_item(item) :
    list_word=item.split()
    list_cleaned=[ word for word in list_word if not (word.isnumeric())]
    new_item=' '.join(list_cleaned)
    return new_item
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_clean_list_word(item, list_no_word):
    list_word=item.lower().split()
    #list_no_word is converted as lower characters
    no_word_lower=' '.join(list_no_word).lower()
    list_no_word=no_word_lower.split()
    list_cleaned=[ word for word in list_word if word.lower() \
    not in list_no_word ]
    
    new_item=' '.join(list_cleaned)
    return new_item.upper()
#-------------------------------------------------------------------------------
    
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_ser_set_len(ser):
    return len(set([ item for item in ser ]))
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p5_df_rfm_build(df, day_now, df_RFM_threshold=None):
   """This function builds RFM features from a given dataframe.

   Dataframe need to have followings columns : 
      * CustomerID , InvoiceNo, InvoiceDate, Total
   Input : 
      * df : pandas dataframe with required columns.
      * day_now : the most recent day from df.
      * df_RFM_threshold : when None value, it iw built in this function from
      df. 
   Output : 
      * df : orginal dataframe in which units column InvoiceDate has been 
      changed into days rather then timestamp.
      * df_RFM : pandas dataframe with 2 columns : 'CustomerID' and 'RFM'
      * df_RFM_threshold : dataframe with threshold for RFM classification
      * day_now : the most recent day from df. An arbitrary value may be given.
      When value is None, then value is the most recent date from dataframe.
   """
   
   is_first_quantiles_build=False
   
   if df_RFM_threshold is None:
      day_now=None
      is_first_quantiles_build=True
   
   #----------------------------------------------------------------------------
   # Total feaure is added
   #----------------------------------------------------------------------------
   if 'Total' in df.columns :
      pass
   else:
      df.loc[:,'Total']=df.Quantity * df.UnitPrice
      #df['Total']=df.Quantity * df.UnitPrice
      df=df[df['Total']>0]
   
   #----------------------------------------------------------------------------
   # Invoice date values are converted into days
   #----------------------------------------------------------------------------
   if 'InvoiceDate' in df.columns:
      try :
         df.loc[:,'InvoiceDate']=df['InvoiceDate']\
            .apply(p5_convert_timestamp_to_days)
      except ValueError as valueError:
         print("*** WARNING : "+str(valueError)) 
   else :
      print("\n*** ERROR : no feature \'InvoiceDate\' into given dataframe")
      return None,None ,  df_RFM_threshold, day_now
   
   
   #----------------------------------------------------------------------------
   # Frequency is computed from InvoiceNo feature.
   # Rows matching to invoice lines have to be gathered as invoices.
   # Once done, then Frequency is deduced from gathered invoices lines.
   #----------------------------------------------------------------------------
   df_RFM=get_invoiceFreq_from_invoiceLine(df)
   
   
   #----------------------------------------------------------------------------
   # Quantiles has not been built; day_now is extracted from dataframe given 
   # as parameter.
   # Otherwise, day_now given as parameter is used.
   #----------------------------------------------------------------------------
   if day_now is None:
      day_now=df.InvoiceDate.max()
   else:
      pass

   df_RFM['Recency']=\
   df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (day_now - x.max())})

   df_RFM['Monatary']=\
   df.groupby('CustomerID').agg({'Total': lambda x: x.sum()})
   
   #----------------------------------------------------------------------------
   # RFM threshold is built from RFM dataframe (df_RFM) quantiles.
   #----------------------------------------------------------------------------
   if df_RFM_threshold is None:
      is_first_quantiles_build=True
      df_RFM_threshold=df_RFM.quantile(q=[0.25 ,0.5, 0.75])
      df_RFM_threshold.reset_index(inplace=True)
      df_RFM_threshold.rename(index={0:'Q1',1:'Q2',2:'Q3'},inplace=True)
      del(df_RFM_threshold['index'])
   else:
      pass
   #----------------------------------------------------------------------------
   # For applying same scoring function, Recency values 
   # are applied with opposite sign.
   #----------------------------------------------------------------------------
   df_RFM['neg_recency']=df_RFM['Recency'].apply(lambda x: -x)
   df_RFM['neg_recency']=sorted(df_RFM['neg_recency'])
   
   if is_first_quantiles_build is True:
      df_RFM_threshold['neg_recency']=\
      df_RFM_threshold['Recency'].apply(lambda x: -x)

   df_RFM_threshold['neg_recency']=sorted(df_RFM_threshold['neg_recency'])

   #----------------------------------------------------------------------------
   # Classification des clients en score RFM
   #----------------------------------------------------------------------------
   df_RFM['R_score']=df_RFM.neg_recency.apply(func_rfm_scoring, \
   args=(df_RFM_threshold,'neg_recency'))

   df_RFM['F_score']=df_RFM.Frequency.apply(func_rfm_scoring, \
   args=(df_RFM_threshold,'Frequency'))

   df_RFM['M_score']=df_RFM.Monatary.apply(func_rfm_scoring, \
   args=(df_RFM_threshold,'Monatary'))

   #----------------------------------------------------------------------------
   #  R, F and M features are aggregated into feature RFM
   #----------------------------------------------------------------------------
   df_RFM['RFM']=df_RFM['R_score'].map(str) + df_RFM['F_score'].map(str) + \
   df_RFM['M_score'].map(str)


   #----------------------------------------------------------------------------
   #  Indexes issued from original dataframes are reset.
   #----------------------------------------------------------------------------
   df_RFM.reset_index(inplace=True)
   
   if is_first_quantiles_build is False:
      #----------------------------------------------------------------------------
      #  RFM score for customerID from dataframe is updated
      #----------------------------------------------------------------------------
      df['RFM']=df_RFM['RFM'].copy()

   return df, df_RFM ,  df_RFM_threshold, day_now
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_feature_one_hot_encode(df, column_name
                              , categorical_features='all'
                              , encoder=None):
   """This function aims to encode feature from a given dataframe.
   If encoder is None, then encoder is built from data in df and then data are
   encoded.
   Otehrwise, data from df are encoded wiith encoder.
   Input : 
      * df : pandas dataframe 
      * column_name : name of columnn containing feature values
   Output : 
      * df_encoded_feature :  features values encoded as a pandas dataframe
      * encoder : encoder structure used to encode feature.
   """
   
   encoded_feature=None
   
   if column_name not in df.columns or 'CustomerID' not in df.columns:
      print("\n*** ERROR : either no column name= \'"+str(column_name)+"\' \
      or no column name= CustomerID into given dataframe ")
      return None, None      
   else:
      pass

   if encoder is None:
      encoder=preprocessing.OneHotEncoder(categorical_features\
      =categorical_features) 
      try :
          encoded_feature \
         =encoder.fit_transform(df[column_name].values.reshape(-1,1))
      except ValueError as valueError :
          print("\n*** df_feature_one_hot_encode() : Erreur encodage : {}"\
          .format(valueError)) 
   else:
      try :
         encoded_feature \
        =encoder.transform(df[column_name].values.reshape(-1,1))
      except ValueError as valueError :
          print("\n*** df_feature_one_hot_encode() : Erreur encodage : {}"\
          .format(valueError)) 
   # ---------------------------------------------------------------------------
   # Encoded values are returned into dataframe with CustomerID as index array.
   # ---------------------------------------------------------------------------
   
   return encoder, pd.DataFrame(encoded_feature.toarray(), index=df.CustomerID)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_rfm_one_hot_encode(df, rfm_column_name, encoder=None):
   """This function aims to encode RFM feature from a given dataframe.
   Dataframe need to have followings columns : 
      * RFM
   Input : 
      * df : pandas dataframe 
      * rfm_column_name : name of columnn containing RFM values
      * encoder : RFM encoder
   Output : 
      * df_encoded_RFM : pandas dataframe encoded RFM values and indexes are
                        CustomerID.
   """

   if rfm_column_name not in df.columns or 'CustomerID' not in df.columns:
      print("\n*** ERROR : either no column name= \'"+str(rfm_column_name)+"\' \
      or no column name= CustomerID into given dataframe ")
      return encoder, None      
   else:
      pass
   
   if encoder is None:
      #-------------------------------------------------------------------------
      # Data model is in built step
      #-------------------------------------------------------------------------
      encoder=preprocessing.OneHotEncoder() 
      try :
          encoded_RFM \
         =encoder.fit_transform(df[rfm_column_name].values.reshape(-1,1))
      except ValueError as valueError :
          print("\n*** Erreur encodage : {}".format(valueError))
   else:
      #-------------------------------------------------------------------------
      # Data model is already built; 
      #-------------------------------------------------------------------------
      encoded_RFM=encoder.transform(df[rfm_column_name].values.reshape(-1,1))
      
   # ---------------------------------------------------------------------------
   # Encoded values are returned into dataframe with CustomerID as index array.
   # ---------------------------------------------------------------------------
   return encoder, pd.DataFrame(encoded_RFM.toarray(), index=df.CustomerID)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_normalize(df,column):
   """Apply a normalization function over a dataframe column values.
   Input : 
      * df : pandas dataframe 
      * column : name of columnn to be normalized
   Output : 
      * df: pandas dataframe with normalized values for column name
   """
   if column not in df.columns:
      print("\n*** ERROR : no column name \'"+str(column)+"\' \
      into given dataframe")
      return None      
   else:
      pass
   mean_v=df[column].mean()
   min_v=df[column].min()
   max_v=df[column].max()
   df[column]=df[column].apply(lambda x: ((x-mean_v)/(max_v-min_v)))
   return df
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def bunch_dump(bunch_object, row_packet, data_path, core_name):
    """Dump a bunch object in a splited manner.
    Each data matching with bunch key is dumped with row_packet rows.
    Input : 
        * data_path : path to access file (director name)
        * core_name : part of file name handling dumped data.
        File name is built as following : <core_name><key_name>_<index>.dump
    Output : none
    """
    list_key=[key for key in  bunch_object.keys() if key not in ['DESCR',]]
    for key in list_key:
        key_core_name = core_name+'_'+str(key)
        
        object_dump_split(bunch_object[key], bunch_object[key].shape[0]\
        , row_packet,data_path,key_core_name)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def bunch_load(list_key, data_len, row_packet, data_path, core_name):
    """Load a bunch object that has been dumped in a splited manner.
    See bunch_dump.
    
    Input : 
        * row_packet : max packet of rows contained into data handles into 
        each dumped file. 
        * data_path : path to access file (director name)
        * core_name : part of file name handling dumped data.
        
        File name is built as following : <core_name><key_name>_<index>.dump
        
    Output : none
    """
    dict_bunch = dict()
    for key in list_key:
        key_core_name = core_name+'_'+str(key)
        
        dict_bunch[key] \
        = object_load_split(data_len, row_packet, data_path, key_core_name)
    return dict_bunch
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def object_dump_split(data,  data_len, step, data_path, core_name):
    """Dump on harddisk a data in a splited manner.
    Dumped file is created with a generic name : <core_name>_<index>.dump
    <index> is the incremented after each dumped data packet.
    Input : 
        * data : data to be dumped. data is supposed having attribute shape.
        * data_len : total size od data rows to be dumped.
        * step : packet size of data rows to be splitted for dump.
        * data_path : path to access file (director name)
        * core_name : part of file name handling dumped data.
    Output : none
    """
    start_len=0
    stop_len=-1
    file_index=0
    for part_len in range(0,data_len):
        if 0 ==  part_len%step :
            start_len=stop_len+1
            stop_len = start_len+step
            file_name = data_path+'/'+core_name+'_'+str(file_index)+'.dump'
            file_index +=1
            if 1 == len(data.shape):
                object_dump(data[start_len:stop_len], file_name)
                print(file_name, data[start_len:stop_len].shape)            
            elif 1 < len(data.shape):
                object_dump(data[start_len:stop_len,:], file_name)
                print(file_name, data[start_len:stop_len,:].shape)
            else:
                print("\n*** ERROR : unconsistent data shape")
                return
            
    print("\nDumping done!\n")

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def object_load_split(data_len, step, data_path, core_name):
    """Load from harddisk data dupmed in a splited manner.
    Dumped file name : <core_name>_<index>.dump
    <index> is the incremented after each loaded file.
    Input : 
        * data_len : total size of data rows to be dumped.
        * step : packet size of data rows to be splitted for dump.
        * data_path : path to access file (director name)
        * core_name : part of file name handling dumped data.
    Output : none
        * data : loaded data
    """
    file_index=0
    start_len=0
    stop_len=-1
    bunch_len=data_len
    bunch_data = None
    list_file_name = sorted(glob.glob(data_path+'/'+core_name+'_*'+'.dump'))

    # First file from list is used for reference
    len_ref = list_file_name[0].rfind('_')
    name_ref= list_file_name[0][:len_ref]
    len_ref = len(name_ref)
    for file_name in list_file_name:
        # File name are filtered in order to avoid to mixture 
        # files with same root name.
        # File lengh before last _ character is tested against a reference 
        # lenght.
        len_file_ref = file_name.rfind('_')
        name_file_ref= file_name[:len_file_ref]
        len_file_ref = len(name_file_ref)
        if len_file_ref != len_ref:
            continue
        else:
            pass
            
        if bunch_data is None:
            bunch_data = object_load(file_name)
        else:
            if 1 == len(bunch_data.shape) :
                try :
                    object_loaded = object_load(file_name)
                    bunch_data \
                    = np.hstack((bunch_data,object_loaded))
                except TypeError :                
                    bunch_data \
                    = np.hstack((bunch_data,object_load(file_name)))
                except ValueError :
                    print("\n*** ERROR on file= "+str(file_name)+" Bunch shape= "+str(bunch_data.shape))
                    return None
            elif 1 < len(bunch_data.shape) : 
                try :
                    bunch_data \
                    = scipy.sparse.vstack((bunch_data,object_load(file_name)))
                except TypeError :                
                    bunch_data = np.vstack((bunch_data,object_load(file_name)))
                                  
            else :
                print("\n*** ERROR : unconsistent data shape\n")
                return None
        print(file_name, bunch_data.shape)

    if False:    
        for part_len in range(0,bunch_len):
            if 0 ==  part_len%step :
                start_len=stop_len+1
                stop_len = start_len+step
                file_name = data_path+'/'+core_name+'_'+str(file_index)+'.dump'
                file_index +=1
                if bunch_data is None:
                    bunch_data = object_load(file_name)
                else:
                    if 1 == len(bunch_data.shape) :
                        print(type(bunch_data))
                        break
                        bunch_data \
                        = scipy.sparse.hstack((bunch_data,object_load(file_name)))
                    elif 1 < len(bunch_data.shape) : 
                        bunch_data \
                        = scipy.sparse.vstack((bunch_data,object_load(file_name)))
                    else :
                        print("\n*** ERROR : unconsistent data shape\n")
                        return None
                print(file_name, bunch_data.shape)
    print("\n Loading splited file done!\n")
    return bunch_data
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def object_dump(object_to_dump, fileName, is_verbose=False):
    """ This function dump (serialize), into a file name, an object given as 
    parameter using pickle package.
    Input : 
      * object_to_dump : object to be dumped
      * fileName : name of the file in which object has been dumped
    Output : none
      
    """
    if None is fileName :
        print("*** ERROR : no name provided for object dumping")   
        return
    else:
        pass

    if 0 == len(fileName) :
        print("*** ERROR : no file name for object dumping")   
        return
    else:
        pass
  
    #---------------------------------------------------------------------------
    # Get file extension; this will lead to type of dump 
    #---------------------------------------------------------------------------
    if is_verbose :
        print("object_dump: file name= {}".format(fileName))
    else :
        pass

    extension = fileName.split(".")[-1]
    if len(extension) >0 :
        try :
            if extension == "dill" :
                with open(fileName,'wb') as fp:
                    dill.dump(object_to_dump,fp)     
            elif extension == "dump" :
                with open(fileName, 'wb') as (dumpedFile):
                    oPickler=pickle.Pickler(dumpedFile)
                    oPickler.dump(object_to_dump)
            else : 
                print("\n*** ERROR : object_dump() : extension not supported for dump : ".format(extension))      
        except pickle.PicklingError as picklingError :
            print("*** ERROR : dumping into "+str(fileName)\
            +" Error= "+str(picklingError))
    else :
        print("\n*** ERROR : object_dump() : No extension for file name= ".format(fileName))      
    return
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def object_load(fileName, is_verbose=True):
    """ This function load a dumped (serialized) object from file name given as 
    parameter. It uses pickle package.
    Input : 
      * file_name : name of the file to access dumped object
      * is_verbose : when True, file name to be loaded is printed.
    Output :
      * dumped (deserialized) object      
    """
    dumped_object = None
    
    if not os.path.exists(fileName)  :
        print("\n*** ERROR : unknown file = {}".format(fileName))
    else :
        if is_verbose :
            print("p5_util.object_load : fileName= "+fileName)
        #-----------------------------------------------------------------------------
        # Get file extension; this will lead to type of dump 
        #-----------------------------------------------------------------------------
        extension = fileName.split(".")[-1]
        if len(extension) >0 :
            if extension == "dill" :
                #-------------------------------------------------------------------
                # Load object dumped in dill format
                #-------------------------------------------------------------------
                with open(fileName,'rb') as fp:
                    try :
                        dumped_object = dill.load(fp)
                    except EOFError as eOFError :
                        print("\n*** ERROR : Loading file failed due to : {}".format(eOFError))
                    except FileNotFoundError:
                        print('\n*** ERROR : file not found : ' + fileName)
                    except ModuleNotFoundError as moduleNotFoundError:
                        print('\n*** ERROR : no module found : ' + str(moduleNotFoundError))
                        
            elif extension == "dump" :
                #-------------------------------------------------------------------
                # Load object dumped in dump format
                #-------------------------------------------------------------------
                try:
                    with open(fileName, 'rb') as (dataFile):
                        oUnpickler=pickle.Unpickler(dataFile)
                        dumped_object=oUnpickler.load()
                except FileNotFoundError:
                    print('\n*** ERROR : file not found : ' + fileName)
                except ModuleNotFoundError as moduleNotFoundError:
                    print('\n*** ERROR : no module found : ' + str(moduleNotFoundError))
            else :
                print("\n*** ERROR : object_load() : extension not supported for dump : ".format(extension))      
            
        else :
            print("\n*** ERROR : object_load() : No extension for file name= ".format(fileName))      


    return dumped_object
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def object_compress_dump(object_to_dump, fileName):
   """ This function compress and dump (serialize) a given object into a 
   file name using pickle package.
   Input : 
      * object_to_dump : object to be compressed then dumped
      * fileName : name of the file in which object has been dumped
   Output : none
      
   """

   if None is fileName :
      print("*** ERROR : no name provided for object dumping")   
      return
   else:
      pass

   if 0 == len(fileName) :
      print("*** ERROR : no file name for object dumping")   
      return
   else:
      pass
      
   
   try :
      with open(fileName, 'wb') as (dumpedFile):
          oPickler=pickle.Pickler(dumpedFile)
          oPickler.dump(zlib.compress(pickle.dumps(object_to_dump)))
   except pickle.PicklingError as picklingError :
      print("*** ERROR : dumping into "+str(fileName)\
      +" Error= "+str(picklingError))   
   return
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def compute_precision_per_segment(y_test, y_pred, list_cluster):
   """Computes and display global accurency predictions and per segment 
      accurency predictionss.
      
      Function used for accurency is metrics.accuracy_score
      Input : 
         * y_test : vector to be tested
         * y_pred : vector issues from prediction model
         * list_cluster : list of market segments found with unsupervised M.L.
         algorithm.
      Output : none
   """

   #----------------------------------------------------------
   # Global accuracy is computed
   #----------------------------------------------------------
   score_global=metrics.accuracy_score(y_test, y_pred)


   dict_score_segment=dict()
   for i_segment in list_cluster :
       #----------------------------------------------------------
       # Get tuple of array indexes matching with targeted segment
       #----------------------------------------------------------
       index_tuple=np.where( y_pred==i_segment )

       #----------------------------------------------------------
       # Extract values thanks to array of indexes 
       #----------------------------------------------------------
       y_test_segment=y_test[index_tuple[0]]
       y_pred_segment=y_pred[index_tuple[0]]
       
       nb_elt_segment=len(y_test_segment)
       
       #----------------------------------------------------------
       # Accuracy is computed and displayed
       #----------------------------------------------------------
       score_segment=metrics.accuracy_score(y_test_segment, y_pred_segment)
       dict_score_segment[i_segment]=score_segment
       #print("Segment "+str(i_segment)+" : "+str(nb_elt_segment)\
       #+" elts / Random forest / Précision: {0:1.2F}".format(score))
   return score_global,dict_score_segment
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def nlp_process(df, feature, vectorizer=None, list_no_words=None\
               , is_verbose=True):
   """This function uses NLTK package in order to expand the given feature 
   from dataframe (mean expanse df.feature Series).
   
   A sequence of transformations is applied to all values from feature.
   Such N.L.P. transformations result in assigning a weight to any word from
   feature values. Each value from feature is expanded with a weight.
   
   Weight value for a word depends on word frequency and 
   word inverse frequency into all values. 
   
   The whole values from feature is regarded as a single corpus of texts.
   
   * Words in at least 1 document (mean row from column Description) are 
      regarded as relevant.
   * Words in more then 80% of the corpus are regarded as irrelevant/
   * Each separated word are considered in vectorization process 
      (ngram_range=(1,1)).


   Input  :
      * df : dataframe containing feature 
      * feature : feature name matching to data to be processed.
      * vectorizer : TFIDF NLTK vectorizer
      * list_no_words : list of words to be removed
      * is_berbose : if fixed to True, then print function is activated.
      
   Output :
      * df : dataframe with additionals features issue from N.L.P. 
      transformations.
      * vectorizer : object of class TfidfVectorizer
      
      * matrix_weights :  matrix of weights linked to any word from N.L.P. 
      transformation.
   """
   
   is_build_step=False
   if vectorizer is None:
      is_build_step=True
      
   if feature not in df.columns:
      print("\n*** ERROR : feature= "+str(feature)+" is not into dataframe\n")

   #----------------------------------------------------------------------------
   # Return additional punctuation 
   #----------------------------------------------------------------------------
   my_punctuation=get_my_punctuation()

   #----------------------------------------------------------------------------
   # NLTK sequence of transformations
   #----------------------------------------------------------------------------
   if is_verbose is True:
      print(get_ser_set_len(df[feature]))
   
   # Digits are removed from item
   df[feature]=df[feature].apply(cb_remove_digit)
   if is_verbose is True:
      print(get_ser_set_len(df[feature]))

   # Punctuations characters are removed from item
   df[feature]=df[feature].apply(cb_remove_punctuation,args=(my_punctuation,))   
   if is_verbose is True:
      print(get_ser_set_len(df[feature]))

   #----------------------------------------------------------------------------
   # We remove stopwords in orde to extract words with most information.
   #----------------------------------------------------------------------------
   df[feature]=df[feature].apply(cb_remove_stopwords)                       
   if is_verbose is True:
      print(get_ser_set_len(df[feature]))

   #----------------------------------------------------------------------------
   # Clean repetitives keyword from item Description 
   #----------------------------------------------------------------------------
   use_idf=True
   if list_no_words is not None:
      use_idf=False
      df[feature]=df[feature].apply(cb_clean_list_word, args=(list_no_words,))
   if is_verbose is True:
      print(get_ser_set_len(df[feature]))

   #----------------------------------------------------------------------------
   # Clean all numeric word from item Description 
   #----------------------------------------------------------------------------
   df[feature]=df[feature].apply(cb_clean_numeric_word_in_item)
   if is_verbose is True:
      print(get_ser_set_len(df[feature]))

   #----------------------------------------------------------------------------
   # Lemmatization of feature values
   #----------------------------------------------------------------------------
   lemmatizer=WordNetLemmatizer()
   df[feature]=df[feature].apply(cb_lemmatizer,args=(lemmatizer,))
   if is_verbose is True:
      print(get_ser_set_len(df[feature]))
   

   #----------------------------------------------------------------------------
   # Stemming of feature values
   #----------------------------------------------------------------------------
   stemmer=nltk.stem.SnowballStemmer('english')
   df[feature]=df[feature].apply(cb_stemmer,args=(stemmer,))
   if is_verbose is True:
      print(get_ser_set_len(df[feature]))
   
   #----------------------------------------------------------------------------
   # Apply vectorization with Text Freq. Inv. Doc. Freq. algorithm.
   #----------------------------------------------------------------------------
   if vectorizer is None:
      #-------------------------------------------------------------------------
      # Data-model building is in progress.
      #-------------------------------------------------------------------------
      vectorizer=TfidfVectorizer( min_df=1, max_df=.3, ngram_range=(1,1))
      csr_matrix_weights=vectorizer.fit_transform(df[feature])
   else:
      #-------------------------------------------------------------------------
      # Data-model is already built. This is a prediction process.
      #-------------------------------------------------------------------------
      csr_matrix_weights=vectorizer.transform(df[feature])

   #----------------------------------------------------------------------------
   # Feature from original dataframe is droped
   #----------------------------------------------------------------------------
   del(df[feature])
   
   #----------------------------------------------------------------------------
   # Data-model building : backup of CSR matrix into dumped file.
   #----------------------------------------------------------------------------
   if is_build_step is True:
      if is_verbose is True:
         print(csr_matrix_weights.shape)
      fileName="./data/matrix_weights_NLP.dump"

      if is_verbose is True:
         print("Dumping matrix_weights into file= "+str(fileName))
      object_dump(csr_matrix_weights, fileName)

      if is_verbose is True:
         print("Done!")
   else:
      pass
   
   return df, csr_matrix_weights, vectorizer
#-------------------------------------------------------------------------------
   
      
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_features_standardize(df, list_feature, p_std_scale=None\
, root_name='STD_'):
   """ Standardizes list of features values given as parameter.
   Columns names issued from standardization are renamed with root_name
   given as parameter.
   Input :
      * df : DataFrame in which list of features to be standardized stands.
      * list_feature : list of quantitative features to be standardized
      * p_std_scale : standardizer class; if None, then standard scaler 
      issued from class p_std_scale is created.
   Return :
      * std_scale : standardizer used to standardize features in list
      * df_quant_std : dataframe of standardized values with same indexes as df
   """
   #----------------------------------------------------------------------------
   # Checking parameters
   #----------------------------------------------------------------------------
   if list_feature is None:
      print("\n*** ERROR : emply list of features")
      return None, None

   for feature in list_feature :
      if feature not in df.columns:
         print("\n*** ERROR : feature= "+str(feature)+" not in dataframe")
         return None, None

   #----------------------------------------------------------------------------
   # Features are aggregated per customer
   #----------------------------------------------------------------------------
   df_quant_cust= pd.DataFrame()

   for col in df.columns:
      df_quant_cust[col]=df.groupby('CustomerID')\
      .agg({col: lambda x: sum(x)})

   #----------------------------------------------------------------------------
   # Data scaling and dataframe handling standardized values is created
   #----------------------------------------------------------------------------
   X_quantitative_std=df_quant_cust.values
   X_quantitative_std=X_quantitative_std.astype(float)
   
   if p_std_scale is None:
      std_scale=preprocessing.MinMaxScaler().fit(X_quantitative_std)
   else:
      std_scale=p_std_scale().fit(X_quantitative_std)

   X_quantitative_std=std_scale.transform(X_quantitative_std)  

   df_quant_std=pd.DataFrame(X_quantitative_std, index=df.index)   

   #----------------------------------------------------------------------------
   # Columns issued from standardization are renamed
   #----------------------------------------------------------------------------
   if root_name is not None:
      dict_rename=dict()
      for col, feature in zip(df_quant_std.columns, list_feature):
          dict_rename[col]=root_name+str(feature)
      df_quant_std.rename(columns=dict_rename,inplace=True)
   

   
   return std_scale, df_quant_std
   
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_rename_columns(df, list_colums, root_name):
   """Change name of columns listed into parameter by adding root_name
   before each column name
   
   Input :
      *  df : dataframe for which columns have to be changed.
      *  list_colums : list of columns names to be renamed
      *  root_name : root name for each column
   Output :
      * df : dataframe with renamed columns names
   """
   
   #----------------------------------------------------------------------------
   # Build dictionary of names to be renamed.
   #----------------------------------------------------------------------------
   dict_matching_name=dict()
   list_col_unchanged=list()
   for col in list_colums:
      if col in df.columns:
         new_col_name=root_name+str(col)
         dict_matching_name[col]=new_col_name
      else:
         print("*** WARNING : column name="+str(col)+" not into dataframe!")
         list_col_unchanged.append(col)
         
   #----------------------------------------------------------------------------
   # Rename columns
   #----------------------------------------------------------------------------
   df.rename(columns=dict_matching_name, inplace=True)
   return df, list_col_unchanged
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def drop_duplicated_lines(df, list_col_dupl):
   """Drop duplicated rows from a given dataframe who matches columns that are 
   in given list of columns.
   
   Invoice lines are considered as duplicated if, for a given row, values 
   from columns 'list_col_dupl' are same.
   
   Input : 
      * df : dataframe where duplicated rows need to be removed
      * list_col_dupl : list of features having to duplicated rows.
   Output :
      * df : dataframe with removed duplicated rows.
   """
   #----------------------------------------------------------------------------
   # Checking input parameters
   #----------------------------------------------------------------------------
   if 0 == len(list_col_dupl):
      return df
      
   #----------------------------------------------------------------------------
   # Drop duplicated rows from given columns
   #----------------------------------------------------------------------------
   df_=df[list_col_dupl].drop_duplicates()

   #----------------------------------------------------------------------------
   # Drop columns from original dataframe
   #----------------------------------------------------------------------------
   list_col_keep \
  =[col for col in df.columns if col not in list_col_dupl]
   df=df[list_col_keep]
   
   #----------------------------------------------------------------------------
   # Aggregate dropped columns into original dataframe
   #----------------------------------------------------------------------------
   df_invoice=pd.concat([df_, df], axis=1, join='inner')

   return df_invoice
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def kmeans_scan_inter_inertia(df, cluster_start, cluster_end):
   """Build a range of clusters and store 
   inter-inertia and clusters model.
   Given dataframe is converted as a CSR matrix on which scan applies.
   
   Input  : 
      * df  : dataframe from which values are clusterized
      * cluster_start : start point for cluster range
      * cluster_end : sendtart point for cluster range
   Output : 
      * dict_kmeans : a dictionary containing kmeans cluster model structured 
        as following : {cluster_id: kmeans_model}
   """
   #arr_inter_inertia=[]

   csr_matrix=sparse.csr_matrix(df.values)

   dict_kmeans=dict()
   for i in range(cluster_start,cluster_end):
       kmeans=cluster.KMeans(n_clusters=i) 
       kmeans.fit(csr_matrix) 
       #arr_inter_inertia.append(kmeans.inertia_)
       dict_kmeans[i]=kmeans
       print("Clustering : {0} clusters".format(i))
   return dict_kmeans
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def gmm_scan(df, cluster_start, cluster_end, p_covariance_type='full'):
   """Build a range of GaussianMixture models and store them into a dictionary.
   Input  : 
      * df  : dataframe from which values are clusterized
      * cluster_start : start point for cluster range
      * cluster_end : sendtart point for cluster range
   Output : 
      * dict_gmm : a dictionary containing GMM cluster model
   """
   #arr_inter_inertia=[]


   dict_gmm=dict()
   for i in range(cluster_start,cluster_end):
       gmm=GaussianMixture(n_components=i, covariance_type=p_covariance_type) 
       gmm.fit(df.values) 
       dict_gmm[i]=gmm
       print("Clustering : {0} clusters".format(i))
   return dict_gmm
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_df_from_features(df, cluster, RFM):
   """Returns a customerID from given cluster value and RFM value.
   Input  : 
      * df : dataframe containing both customerID, cluster and RFM features.
      Index must be customers identifiers.
      
      * cluster : cluster value
      * RFM : RFM value
   Output :
      * arr_customer_id : array of customers identifiers from df dataframe 
      matching with given input values.
   """
   if 'cluster' not in df.columns:
      print("\n*** ERROR : ")
   df1=df[df['cluster']==cluster]
   df2=df1[df1['RFM']==RFM]
   arr_customer_id=df2.index

   return arr_customer_id
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_ratio_uniqueItem_over_invoiceLines(df, customers_index):
   rat=0.0
   for customer_id in customers_index:
       rat += len(df[df['CustomerID']==customer_id]['StockCode'].unique())/\
       len(df[df['CustomerID']==customer_id])
   rat /=len(customers_index)    
   return rat
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_invoiceFreq_from_invoiceLine(df) :
   """Returns dataframe with a column matching with invoice count for each 
   customer.
   Dataframe indexes are customers identifiers.
   
   Input :
      * df : dataframe from which each raw is an invoice line.
   Output :
      * df_freq : a dataframe with :
         --> CustomerID as index
         --> Frequency as coluumn, matching with invoice counts.      
   """
   df_freq=df.groupby(['CustomerID','InvoiceNo']).agg({'InvoiceNo': lambda x:len(x)})
   df_freq.rename(columns={'InvoiceNo':'Frequency'}, inplace=True)

   df_freq=df_freq.groupby(['CustomerID']).agg({'Frequency': lambda x:len(x)})   
   return df_freq
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def feature_dow_new(df):
   """Created the new feature, Day of Week, named dow. 
   This feature is encoded in a new dataframe. 
   Then a dataframe is created from encoded feature and condensed per customer 
   by summing all encoded values per customer.
   
   Input : 
      * df : dataframe containing invoice lines informations.
   Output : 
      * df_dow_encoded_cust : dataframe with sum of encoded dow per customer.
      * dow_encoder : encoder used for dow encoding
   """
   #----------------------------------------------------------------------------
   # Checking dataframe
   #----------------------------------------------------------------------------
   if 'InvoiceDate' not in df.columns:
      print("\n*** ERROR : no column InvoiceDate into dataframe!")
      return None, None
   else:
      pass
   
   #----------------------------------------------------------------------------
   # dow is created per invoice line raw.
   #----------------------------------------------------------------------------
   df['dow']=df['InvoiceDate'].apply(p5_get_dayOfWeek_from_timestamp)

   #----------------------------------------------------------------------------
   # dow 5 is missing : dataframe is reworked for adding this feature
   #----------------------------------------------------------------------------
   list_dow=[(dow,df[df.dow==dow].shape[0]) for dow in range(0,7,1)]

   list_dow_missing=list()
   for dow_tuple in list_dow:
      dow=dow_tuple[0]
      dow_count=dow_tuple[1]
      if 0 == dow_count:
         list_dow_missing.append(dow)

   if 0 < len(list_dow_missing):
      pass      
   else:
      pass   

   #----------------------------------------------------------------------------
   # dow is encoded
   #----------------------------------------------------------------------------
   dow_encoder, df_dow_encoded=df_feature_one_hot_encode(df, 'dow')

   #----------------------------------------------------------------------------
   # Columns are renamed with root name w_dow_
   #----------------------------------------------------------------------------
   list_colums=df_dow_encoded.columns
   df_dow_encoded, list_col_unchanged \
  =df_rename_columns(df_dow_encoded, list_colums, 'w_dow_')   
   
   #----------------------------------------------------------------------------
   # df_dow_encoded columns are condensed per customer
   #----------------------------------------------------------------------------

   if 'CustomerID' in df_dow_encoded.columns:
      del(df_dow_encoded['CustomerID'])
   df_dow_encoded['CustID']=df_dow_encoded.index

   list_col=[col for col in df_dow_encoded.columns if col not in ['CustID']]    

   df_dow_encoded_cust=pd.DataFrame()

   for col in list_col:
       df_=pd.DataFrame(df_dow_encoded.groupby('CustomerID')\
       .agg({col: lambda x: sum(x)}))
       df_dow_encoded_cust=pd.concat([df_dow_encoded_cust,df_], axis=1)

   return df_dow_encoded_cust, dow_encoder
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def cb_sum_feature_binary_flag(sum_feature):
   if sum_feature >0:
      return 1
   else:
      return 0
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def time_feature_encoded_new(df, feature, encoder=None):
   """Create a new feature from name given as parameter using 
   'InvoiceDate' column.
   
   This new feature is encoded in a new dataframe.

   Feature values are condensed per customer by summing all encoded values 
   per customer.

   Input : 
      * df : dataframe containing InvoiceDate column.
      * feature : feature name to be created; supported values : 
      year, month, day, dow (day of week), pod (Period of Day)
   Output : 
      * df_feature_encoded_cust : dataframe with sum of encoded feature 
      per customer.
      * feature_encoder : encoder used for encoding new feature
   """
   list_features_supported=['year','month', 'day','dow','pod']
   #----------------------------------------------------------------------------
   # Checking dataframe
   #----------------------------------------------------------------------------
   if 'InvoiceDate' not in df.columns:
      print("\n*** ERROR : no column InvoiceDate into dataframe!")
      return None, None
   else:
      pass

   #----------------------------------------------------------------------------
   # feature is created per invoice line row.
   #----------------------------------------------------------------------------
   if feature not in list_features_supported:
      print("\n*** ERROR : new feature="+str(feature)+" is not supported!")
      return None, None
   try:
      if 'year' == feature:
         df[feature]=df['InvoiceDate'].apply(p5_get_year_from_timestamp)
      elif 'month' == feature:
         df[feature]=df['InvoiceDate'].apply(p5_get_month_from_timestamp)
      elif 'day' == feature:
         df[feature]=df['InvoiceDate'].apply(p5_get_day_from_timestamp)
      elif 'dow' == feature:
         df[feature]=df['InvoiceDate'].apply(p5_get_dow_from_timestamp)
      elif 'pod' == feature:
         df[feature]=df['InvoiceDate'].apply(p5_get_pod_from_timestamp)
      else :
         print("\n*** ERROR : new feature="+str(feature)+" is not supported!")
         return None, None
   except ValueError as valueError:
      print("*** WARNING : "+str(feature)+" : "+str(valueError)) 
      return None, None
   #----------------------------------------------------------------------------
   # feature given as parameter is encoded
   #----------------------------------------------------------------------------
   feature_encoder, df_feature_encoded \
  =df_feature_one_hot_encode(df, feature, encoder=encoder)

   #----------------------------------------------------------------------------
   # Columns issued form encoding process are renamed with root name w_feature_
   #----------------------------------------------------------------------------
   list_colums=df_feature_encoded.columns
   root_name='w_'+str(feature)+'_'
   
   df_feature_encoded, list_col_unchanged \
  =df_rename_columns(df_feature_encoded, list_colums, root_name)   

   #----------------------------------------------------------------------------
   # df_feature_encoded columns are condensed per customer
   # 
   # Using CustID rather then CustomerID as column name allows to avoid warning 
   # to be triggered from Python; using same name for index and column is not 
   # recomended.
   #----------------------------------------------------------------------------
   if 'CustomerID' in df_feature_encoded.columns:
      del(df_feature_encoded['CustomerID'])
   df_feature_encoded['CustID']=df_feature_encoded.index

   list_col \
  =[col for col in df_feature_encoded.columns if col not in ['CustID']]    

   df_feature_encoded_cust=pd.DataFrame()

   #----------------------------------------------------------------------------
   # Encoded features are summarized for each customer ID
   #----------------------------------------------------------------------------
   for col in list_col:
       df_=pd.DataFrame(df_feature_encoded.groupby('CustID')\
       .agg({col: lambda x: sum(x)}))
       df_feature_encoded_cust \
      =pd.concat([df_feature_encoded_cust,df_], axis=1)
       
   #----------------------------------------------------------------------------
   # Depending of new feature, a post-processing is applied
   #----------------------------------------------------------------------------
   if False:
      if feature in ['year','month','day','dow','pod']:
         #----------------------------------------------------------------------
         # A flag 0 or 1 is applied if sum is 0 or > 0
         #----------------------------------------------------------------------
         for new_col in df_feature_encoded_cust.columns:
            df_feature_encoded_cust[new_col] \
           =df_feature_encoded_cust[new_col].apply(cb_sum_feature_binary_flag)
      else:
         pass
    

   #----------------------------------------------------------------------------
   # Index is renamed as CustomerID
   #----------------------------------------------------------------------------
   df_feature_encoded_cust.index.rename('CustomerID', inplace=True)

   return df_feature_encoded_cust, feature_encoder


#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def time_list_feature_build(df, list_feature, dict_encoder=None\
, is_verbose=True):
   """Create dataframes handling new features from InvoiceDate feature.
   One dataframe per new feature is created then aggregated into a single 
   dataframe.
   
   List of new features extracted from InvoiceDate are contained into 
   list_feature.
   
   New created features are encoded then summarized per customer.
   
   When encoder given as parameter is None, then data-model is in built step.
   During built step, each new dataframe (matching with a new feature) is 
   dumped into a separate file.

   Otherwise, data-model is already built and this method is called in the
   segmentation prediction process.   
   During this step, each new dataframe (matching with a new feature) 
   is aggregated into a new dataframe.
   
   Input : 
      * df : dataframe with feature InvoiceDate
      * list_feature : list of new features names to be created

   Output : 
      * One file per new feature containing a dataframe is dumped under the 
      name : ./data/df_customers_<new_feature>.dump where <new_feature> 
      is an element from list_feature
      * endocer used to encode new features is returned
   """

   df_aggregated=None
   is_built_step=False
   
   if dict_encoder is None:
      is_built_step=True
      dict_encoder=dict()
   else:
      pass
      
   for new_feature in list_feature :
      if is_verbose is True:
         print("is_built_step : "+str(is_built_step))
      if is_built_step is True:
         df_feature_encoded_cust, feature_encoder \
        =time_feature_encoded_new(df, new_feature, encoder=None)
         dict_encoder[new_feature]=feature_encoder
      else:
         df_feature_encoded_cust, feature_encoder \
        =time_feature_encoded_new(df, new_feature\
         , encoder=dict_encoder[new_feature])
         
      
      if df_feature_encoded_cust is not None:
         if is_verbose is True:
            print("Time feature : "+str(new_feature)+" --> "\
            +str(df_feature_encoded_cust.shape))
         if is_built_step is True:
            #-------------------------------------------------------------------
            # Data-model is in built step.
            #-------------------------------------------------------------------
            fileName='./data/df_customers_'+new_feature+'.dump'
            object_dump(df_feature_encoded_cust,fileName)
         else:
            #-------------------------------------------------------------------
            # Data-model is already built; this is a prediction process
            # Each dataframe produced from new time fature is aggregated
            #-------------------------------------------------------------------
            if df_aggregated is None:
               df_aggregated=df_feature_encoded_cust.copy()
            else:
               df_aggregated \
              =pd.concat([df_aggregated, df_feature_encoded_cust], axis=1\
               ,join='inner')
      else :
         print("*** ERROR : creating new feature= "+str(new_feature)+" Failed!")

   if is_built_step is True:
      #----------------------------------------------------------------------
      # Data-model is in built step.
      #----------------------------------------------------------------------
      return dict_encoder, pd.DataFrame()
   else:
      if is_verbose is True:
         print("Aggregated time features df : "+str(df_aggregated.shape))
      return dict_encoder, df_aggregated
   
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def time_list_feature_restore(list_feature\
                              , std_scale=None, df_timeFeature=None\
                              , is_verbose=True):
   """Read dataframe files issued from time features created from InvoiceDate 
   with function time_list_feature_build().

   When std_scale value is None, then data are loaded from dumped files.
   Features are aggregated in a single dataframe and are standardized.
   
   Input : 
      * list_feature : list of features to be restored from dumped files.
      * std_scale : data scaler; when value is None, then data-model 
      is in built step. Otherwise, this method is called in the prediction 
      process.
   Output :
      * df_timeFeature : standardized dataframe.
      * std_scale : data scaler.
   """
   if df_timeFeature is not None:
      if is_verbose is True:
         print("df_timeFeature= "+str(df_timeFeature.shape))
   else:
      pass
      
      
   if std_scale is None:
      list_df=list()
      for new_feature in list_feature :
         fileName='./data/df_customers_'+new_feature+'.dump'
         df=object_load(fileName)
         list_df.append(df)

      #-------------------------------------------------------------------------
      # Aggregation : 
      #-------------------------------------------------------------------------

      # Initialization
      df_timeFeature=list_df[0].copy()
      
      # Aggregation for all remaining elts of list
      for i in range(1,len(list_df)):
         df_timeFeature=pd.concat([df_timeFeature,list_df[i]]\
         , axis=1, join='inner')
   else:
      #-------------------------------------------------------------------------
      # data is issued from df_timeFeature given as parameter function.
      #-------------------------------------------------------------------------
      pass
   #----------------------------------------------------------------------------
   # Standardization
   #----------------------------------------------------------------------------
   X=df_timeFeature.values
   if std_scale is None:
      std_scale=preprocessing.StandardScaler().fit(X)
   else:
      pass

   X_std=std_scale.transform(X)
   df_timeFeature_std=pd.DataFrame(X_std, index= df_timeFeature.index\
   ,columns= df_timeFeature.columns)
   df_timeFeature=df_timeFeature_std.copy()
   del(df_timeFeature_std)
   return df_timeFeature, std_scale
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_add_cluster(df,arr_clusters):
   """Insert array labels as a new column into dataframe df.
   Input :
      * df : dataframe
      * arr_clusters : array of numbers issued from clustering
      
   Output :
      * df_customers : new dataframe added with cluster column
   """

   if 'cluster' in df.columns:
       del(df['cluster'])

   #-------------------------------------------------------------------    
   # Checking for Nan values    
   #-------------------------------------------------------------------    
   is_nan_in_arr=np.isnan(df.values)

   df_segment=pd.DataFrame(arr_clusters, index=df.index, columns=['cluster'])
   df_=pd.concat([df_segment,df], axis=1, join='inner')

   
   return df_

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_pca_reduce(df, n_dim, root_name, p_is_scale=True, pca=None):
   """Proceed to dataframe dimention reduction with PCA.

   If n_dim is greater or equal to df dimension, then df is returned.
   
   A reduction dimension is applied and reduced data are handled 
   in a new dataframe. Columns from this new dataframe are renamed. 

   Index from this new dataframe have same values then CustomerID column from 
   original dataframe. This allows to concatenate this dataframe to other 
   one with 'join' attribute.
   
   Input :
      * df : dataframe on which dimension reduction applies
      * n_dim : new dimension after reduction
      * root_name : root column name after dimension reduction.
      * p_scale : standar scaler is appied to dataframe value if True
      * pca : PCA decomposer; when value is None, then PCA is built. Otherwise 
      pca is applied to decrease dimension.
   Output :
      * df_pca_reduced : dataframe with reduced dimension
      * pca : PCA reducer fitted with data values from input dataframe
   """
   #----------------------------------------------------------------------------
   # Checking if dimension reduction applies
   #----------------------------------------------------------------------------
   if df.shape[1] <= n_dim:
      print("*** WARNING : dataframe dimention too low for reduction : "\
      +str(df.shape[1]))
      return df, pca

   #----------------------------------------------------------------------------
   # Get standardized data
   #----------------------------------------------------------------------------
   list_col=[col for col in df.columns if col not in ['CustomerID']]

   X=df[list_col].values
   if p_is_scale is True:
      std_scale=preprocessing.StandardScaler().fit(X)
      X_std=std_scale.transform(X)
   else:
      X_std=X.copy()

   #----------------------------------------------------------------------------
   # Reduction of dimension is applied
   #----------------------------------------------------------------------------
   if pca is None:
      pca=PCA(n_components=n_dim)
      X_pca=pca.fit_transform(X_std)
   else:
      if n_dim != pca.n_components:
         print("*** WARNING : Using PCA with components= "\
         +str(pca.n_components)+" Expected components= "+str(n_dim))
      else:
         pass
      X_pca=pca.transform(X_std)

   if 'CustomerID' in df.columns:
      df_pca=pd.DataFrame(X_pca, index=df.CustomerID)
   else :
      df_pca=pd.DataFrame(X_pca, index=df.index)
   

   #----------------------------------------------------------------------------
   # Reduced dataframe columns are renamed
   #----------------------------------------------------------------------------
   dict_rename=dict()
   for col in  df_pca.columns:
      dict_rename[col]=root_name+str(col)

   df_pca.rename(columns=dict_rename,inplace=True)

   return df_pca , pca 
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def compress(my_object) :
   ''' Compress object given as parameter using zlib object.'''
   # Compress:
   compressed=zlib.compress(pickle.dumps(my_object))
   return compressed
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def gmm_hyper_parameter_cv(df, t_cluster, dict_hyper_parameter):
   """Builds GMM models ranging from cluster_start to cluster_end, for each 
   GMM hyper-parameter from dict_hyper_parameter.
   
   Input :
      * df : dataframe containing values from which GMM models are built
      * t_cluster : tuple of (cluster_start, cluster_end)
      * dict_hyper_parameter : each key is an hyper-parameter for GMM adressing
      a list of hyper-parameters values.
   Output :
      * dict_list_gmm_model : dictonary of GMM models, each key is an 
      hyper-parameter adressing a list of GMM models ranking from cluster_start
      to cluster_end.
   """

   X=df.values
   
   cluster_start=t_cluster[0]
   cluster_end=t_cluster[1]
   
   n_components=np.arange(cluster_start, cluster_end+1)
 
   print("Clustering from clusters range from : "\
   +str(cluster_start)+" --> "+str(cluster_end))
 
   dict_covariance_type={1:'diag', 2:'spherical', 3:'full'}
   dict_list_gmm_model=dict()
   for hyper_parameter, list_hyper_parameter in dict_hyper_parameter.items():
      print("GMM Hyper-parameter type= "+str(hyper_parameter))
      if hyper_parameter == 'covariance_type':
         for hyper_param_value in list_hyper_parameter:
            print("Hyper parameter value : "+str(hyper_param_value))
            #------------------------------------------------------------------
            # For a given hyper-parameter type, GMM models ranging from 
            # cluster_start to cluster_end are built
            #------------------------------------------------------------------
            list_gmm_model \
           =[ GaussianMixture(n, covariance_type=hyper_param_value\
            , random_state=0).fit(X) for n in n_components]
            dict_list_gmm_model[hyper_param_value]=list_gmm_model
      else:
         pass
    
   return dict_list_gmm_model
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def gmm_silhouette_compute(df, dict_list_gmm_model):
   """Compute silhouette coefficient for a list of GMM models matching with 
   a list of GMM hyper-parameters.
   Input : 
      * df : dataframe containing values from which GMM models are built
      * dict_list_gmm_model : dictonary of GMM models; each key is an 
      hyper-parameter adressing a list of GMM models.
   
   Output : 
      * dict_dict_silhouette_score : dictionary with : 
         -> keys that are hyper_parameter 
         -> values are dictionary of silhouette score. For such dictionary,
            keys are clusters and values are mean silhouette scores.
   """
   dict_dict_silhouette_score=dict()
   X =df.values
   
   for hyper_parameter, list_gmm_model in dict_list_gmm_model.items():
      #----------------------------------------------------------------------
      # For each one of the GMM models hyper-parameters, dictionary to be 
      # returned is filled with dictionary of silhouette scores computed for 
      # all GMM models.
      #----------------------------------------------------------------------
      dict_silhouette_score=dict()
      print("GMM Silhouette score: Hyper-parameter="+str(hyper_parameter))
      for gmm_model in list_gmm_model:
         #----------------------------------------------------------------------
         # For each one of the GMM models matching with an hyper-parameter, 
         # silhouette scores are computed and stored in a dictionary.
         #----------------------------------------------------------------------
         if 1 == gmm_model.n_components:
            continue
         else:
            cluster=gmm_model.n_components
            preds_gmm=gmm_model.predict(X)
            dict_silhouette_score[cluster]=silhouette_score(X, preds_gmm)
            print("GMM Silhouette score: Cluster= "+str(cluster))
      dict_dict_silhouette_score[hyper_parameter]=dict_silhouette_score
      print("")
   return dict_dict_silhouette_score
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------

def df_sampling(df, col, ratio, mode='random'):
    """ Returns a sampled dataframe from an original one given as parameter.

    Sampling process is based on filtering values from the dataframe column.
    
    Input : 
      * df : original dataframe from whichs sample dataframe is extracted
      * col : column from df from which values are filtered
      * ratio : percent of values in sampled dataframe 
      * mode : mode on which filter is applied to get sampling.
    Output : 
      * df_sampling : sampled dataframe
    """

    if col not in df:
      print("***ERROR : col= "+str(col)+" not in dataframe!")
      return None

    #--------------------------------------------------------------
    # Getting all values from col
    #--------------------------------------------------------------
    arr_col_values=df[col].unique()

    #--------------------------------------------------------------
    # Extracting a number of array values considering ratio given 
    # as parameter.
    #--------------------------------------------------------------
    nb_sampling=int(len(arr_col_values)*ratio)
    
    #--------------------------------------------------------------
    # Filtering in a random manner nb_sampling elements from array
    # of values
    #--------------------------------------------------------------
    if mode == 'random' :
      arr_sampling_col=np.random.choice(arr_col_values, size=nb_sampling)
    else:
      print("***ERROR : filtering mode= "+str(mode)+" is not supported!")
      return None
    
    #--------------------------------------------------------------
    # Boolean mask creation from arr_sampling_col values
    #--------------------------------------------------------------    
    ser_mask=df[col].apply(lambda x : x in arr_sampling_col)

    #--------------------------------------------------------------
    # Boolean mask is applied over original dataframe in order to 
    # to get a new dataframe.
    #--------------------------------------------------------------    
    arr_index_true=np.where(ser_mask.values == True)[0]
    #df_sampling=df.loc[arr_index_true]
    df_sampling=df.reindex(arr_index_true)
    
    #--------------------------------------------------------------
    # United Kingdom is filtered
    #--------------------------------------------------------------    
    if 'Country' in df_sampling.columns:
       df_sampling=df_sampling[df_sampling.Country == 'United Kingdom']
    return df_sampling
#-------------------------------------------------------------------------------

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def df_nlp_sum_per_customer(df, csr_matrix_weights, root_name):
   """Returns a dataframe in which NLP weights are summerized for each 
   customer.
   
   Input : 
      * df : dataframe with weights to be summerized 
      per customer.
      * csr_matrix_weights : matrix of NLP weights in CSR format
      * root_name : root name for columns with weights to be summerized.
   Output : 
      * df_w_nlp : dataframe with NLP weights summerized per customer.
   """   
   
   #csr_matrix_df=sparse.csr_matrix(df[['CustomerID']].values)
   csr_matrix_df=sparse.csr_matrix(df[['CustomerID']].astype(int))
   csr_hstack_matrix \
  =sparse.hstack((csr_matrix_df, csr_matrix_weights),format='csr')

   #-------------------------------------------------------------------------
   # Get customerID array from CSR matrix
   #-------------------------------------------------------------------------
   arr_customerID=np.unique(csr_hstack_matrix[:,0].A)
   arr_customerID.shape

   #-------------------------------------------------------------------------
   # Create an array with summerized columns per customerID : initialization
   #-------------------------------------------------------------------------
   customerID=arr_customerID[0]
   my_filter=csr_hstack_matrix[:,0].A==customerID
   index_customerID=np.where(my_filter)[0]

   arr_hstack_matrix_sum \
  =csr_hstack_matrix[index_customerID,1:].A.sum(axis=0)

   arr_hstack_matrix_sum=arr_hstack_matrix_sum.reshape(1,-1)
   arr_all= np.c_[ customerID, arr_hstack_matrix_sum ] 

   #-------------------------------------------------------------------------
   # Create an array with summerized columns per customerID 
   #-------------------------------------------------------------------------
   for customerID in arr_customerID[1:]:
      #----------------------------------------------------------------------
      # Create a boolean rows filter based on customerID value 
      # customerID value is first colun from CSR matrix
      #----------------------------------------------------------------------
      my_filter=csr_hstack_matrix[:,0].A==customerID

      #----------------------------------------------------------------------
      # Get indexes rows from boolean filter
      #----------------------------------------------------------------------
      index_customerID=np.where(my_filter)[0]

      #----------------------------------------------------------------------
      # Each column matching with filtered rows is summerized; 
      # fist column is ignored.
      #----------------------------------------------------------------------
      arr_hstack_matrix_sum \
     =csr_hstack_matrix[index_customerID,1:].A.sum(axis=0)

      #----------------------------------------------------------------------
      # Array with summurized values in columns is reshape allowing to add 
      # customerID in first column
      #----------------------------------------------------------------------
      arr_hstack_matrix_sum=arr_hstack_matrix_sum.reshape(1,-1)

      #----------------------------------------------------------------------
      # CustomerID is added in 1st column
      #----------------------------------------------------------------------
      arr_hstack_matrix_sum= np.c_[ customerID, arr_hstack_matrix_sum ] 

      #----------------------------------------------------------------------
      # Rows per customer are aggregated
      #----------------------------------------------------------------------
      arr_all=np.r_[arr_all, arr_hstack_matrix_sum ]    

   #-------------------------------------------------------------------------
   # A datarame is created from aggregated rows array
   #-------------------------------------------------------------------------
   df_w_nlp=pd.DataFrame(arr_all)

   del(arr_all)

   #-------------------------------------------------------------------------
   # Dataframe columns are renamed, except fisrt column that is matching with 
   # CustomerID values.
   #-------------------------------------------------------------------------
   root_name='w_nlp_'

   list_col=[col for col in df_w_nlp.columns \
   if str(col).isdigit() and col >0]

   df_w_nlp,list_col_unchanged= df_rename_columns(df_w_nlp, list_col, root_name)    

   #-------------------------------------------------------------------------
   # First column from dataframe is renamed 
   #-------------------------------------------------------------------------
   df_w_nlp.rename(columns={0:'CustomerID'}, inplace=True)

   #-------------------------------------------------------------------------
   # CustomerID column is casted as an integer
   #-------------------------------------------------------------------------
   df_w_nlp.CustomerID=df_w_nlp.CustomerID.map(int)
   
   return df_w_nlp
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def p5_reload_data_sample(mode='random'):
   """Load from dumped file a sampling of original dataset.
   
      Input : 
         * mode : sampling mode dumped file is issued from.
      output :
         * dataframe containing data sampling.
   """
   if mode == 'random':
      file_name='./data/df_invoice_line_sample_random.dump'
   else:
      print("*** ERROR : sampling mode= {} not supported!")
      return None
   
   df_invoice_line=object_load(file_name)
   print(df_invoice_line.shape)
   if 'Country' in df_invoice_line.columns:
      df_invoice_line.query("Country == 'United Kingdom'", inplace=True)
      list_col \
     =[col for col in df_invoice_line.columns if col not in ['Country']]
      df_invoice_line=df_invoice_line[list_col]
    
   return df_invoice_line
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def p5_build_df_cluster(df, df_customers, agg_col_name
, new_col_name, tuple_cluster, agg_func):
   
   df_cluster \
  =df.groupby(['CustomerID',agg_col_name])\
   .agg({agg_col_name: lambda x:agg_func(x)})
   
   df_cluster.rename(columns={agg_col_name:new_col_name}, inplace=True)
   df_cluster.reset_index(inplace=True)
   if 'CustomerID' not in df_customers:
       df_customers['CustomerID']=df_customers.index
   df_cluster \
  =pd.merge(df_cluster, df_customers[['cluster','CustomerID']], on='CustomerID')
   if 0 < len(tuple_cluster) :
      select=str()
      for pos_tuple in range(0, len(tuple_cluster)):
         cluster=tuple_cluster[pos_tuple]
         select += 'cluster == '+str(cluster)+" or "
      #-------------------------------------------------------------------------
      # Remove trailers characters 'or '
      #-------------------------------------------------------------------------
      select=select[:-3]

   #-------------------------------------------------------------------------
   # Apply cluster selection
   #-------------------------------------------------------------------------
   if 0 < len(tuple_cluster) :
      df_cluster.query(select, inplace=True)

   return df_cluster
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_cluster_list_add(df, df_customers, list_cluster):
   """ Add clusters values to a given dataframe thanks to df_customers 
   dataframe.
   
   Cluster values are issued from list_cluster.
   
   Input : 
      * df : dataframe for which cluster cluster column is added.
      * df_customers : dataframe containing clusters values per customer.
      * list_cluster : list of cluster values to be added.
      
   Output  : 
      * df_invl_cluster : dataframe with added clusters.
      
   """
   if 'cluster' in df.columns:
      print("*** WARNING : cluster column already in dataframe !")
      return df
   else:
      pass
   
   #----------------------------------------------------------------------------
   # Index reset allows to have CustomerID column, allowing merge operation
   #----------------------------------------------------------------------------
   if 'CustomerID' not in df_customers:
      df_customers.reset_index(inplace=True)

   #----------------------------------------------------------------------------
   # df is imerged with df_customers to get clusters values from df_customers.
   #----------------------------------------------------------------------------
   df=pd.merge(df, df_customers[['cluster','CustomerID']], on='CustomerID')
   
   #----------------------------------------------------------------------------
   # Build select condition for clusters selection issue from list_custer.
   #----------------------------------------------------------------------------
   if 0 < len(list_cluster) :
      select=str()
      for cluster in list_cluster:
         select += 'cluster == '+str(cluster)+" or "
      #-------------------------------------------------------------------------
      # Remove trailers characters 'or '
      #-------------------------------------------------------------------------
      select=select[:-3]
   else : 
      pass
   
   #-------------------------------------------------------------------------
   # Apply cluster selection
   #-------------------------------------------------------------------------
   if 0 < len(list_cluster) :
      df.query(select, inplace=True)
   
   
   return df
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_cluster_filter(df, list_cluster):
   """Returns a dataframe with cluster columns filtered with values from 
   list_cluster.
   Input :
      * df : dataframe to be filtered 
      * list_cluster : list of clusters values to be filtered
   Output:
      * dataframe with filtered cluster values.
   """
   df_=pd.DataFrame()
   for cluster in list_cluster:
       df_=pd.concat([df_, df[df.cluster==cluster]], axis=0)
       
   
   return df_
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_drop_customers_from_clusterList(df, y, list_cluster):
   """ Returns dataframe and array given as parameters with elements 
   excluded from clusters list values, list_cluster.

   Index of returned dataframe is based on customerID.
   """ 
   
   #----------------------------------------------------------------------------
   # Checking inputs.
   #----------------------------------------------------------------------------
   if len(list_cluster) == 0:
      print("*** WARNING : No cluster value in list! ")
      return df,y

   #----------------------------------------------------------------------------
   # Build list of customerID to be excluded
   #----------------------------------------------------------------------------
   list_customerID=list()
   #for cluster in list_cluster:
   #    list_customerID.append(y[y==cluster].index[0])
   for cluster in list_cluster:
       list_customerID.append(df[df.cluster==cluster].index)

   if len(list_customerID) == 0:
      print("*** WARNING : No element excluded from array Y! ")
      return df,y
   
   #----------------------------------------------------------------------------
   # Exclude from array y all elements that are matching with clusters 
   # in list_cluster.
   #----------------------------------------------------------------------------
   for cluster in list_cluster:
       mask_y=ma.masked_array(y, mask=( y==cluster ))
       y=mask_y[~mask_y.mask].data

   #----------------------------------------------------------------------------
   # Exclude from given dataframe rows that are not in built list list_customerID
   #----------------------------------------------------------------------------
   df.reset_index(inplace=True)
   for customerID in list_customerID:
       df=df[df.CustomerID != customerID]


   df.index=df.CustomerID
   return df, y
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_monthly_incomes_build(df, df_customers, dict_month_name):
   """Build dataframe of monthly incomes with such structure : 
      --> Rows : customer identifier
      --> 1st colum : cluster
      --> Other columns : months, identified by month name.
   Input : 
      * df : dataframe containing invoice lines.
      * df_customers : dataframe containing cluster assigned to customer.     
      * dict_month_name : dictonary matching month value with onth name.
   """
   #----------------------------------------------------------------------------
   # Checking inputs
   #----------------------------------------------------------------------------
   if 'Quantity' not in df.columns:
      print("*** ERROR : column= Quantity not into dataframe!")
      return
      
   if 'UnitPrice' not in df.columns:
      print("*** ERROR : column= UnitPrice not into dataframe!")
      return

   #----------------------------------------------------------------------------
   # Total per row is added as a new column.
   #----------------------------------------------------------------------------
   df['Total']=df.Quantity *  df.UnitPrice

   #----------------------------------------------------------------------------
   # Month column is added and is issued from InvoiceDate
   #----------------------------------------------------------------------------
   df['Month']=df['InvoiceDate'].apply(p5_get_month_from_timestamp)

   #----------------------------------------------------------------------
   # Monthly incomes dataframe is computed.
   #----------------------------------------------------------------------
   df_monthly_incomes=pd.DataFrame()

   min_month=1
   max_month=12


   #----------------------------------------------------------------------
   # Initialization stage
   #----------------------------------------------------------------------
   month=min_month

   #----------------------------------------------------------------------
   # Month where sum takes place is filtered from dataframe
   #----------------------------------------------------------------------
   df_month=df[df.Month==month]

   #----------------------------------------------------------------------
   # Total for this month is computed per customer in a new column
   #----------------------------------------------------------------------
   df_month_total \
  =df_month.groupby('CustomerID').agg({'Total': lambda x: sum(x)})

   month_name=dict_month_name[month]
   df_month_total[month_name]=df_month_total.loc[:,'Total']

   #----------------------------------------------------------------------
   # Drop Total column 
   #----------------------------------------------------------------------
   df_month_total.drop(['Total'],inplace=True,axis=1)

   #----------------------------------------------------------------------
   # Month columns are aggregated 
   #----------------------------------------------------------------------
   df_monthly_incomes=df_month_total

   #----------------------------------------------------------------------
   # Loop on all months except first one used for initialization
   #----------------------------------------------------------------------
   for month in range(min_month+1, max_month+1):

      #----------------------------------------------------------------------
      # Month where sum takes place is filtered from dataframe
      #----------------------------------------------------------------------
      df_month=df[df.Month==month]

      #----------------------------------------------------------------------
      # Total for this month is computed per customer in a new column
      #----------------------------------------------------------------------
      df_month_total \
     =df_month.groupby('CustomerID').agg({'Total': lambda x: sum(x)})

      month_name=dict_month_name[month]
      df_month_total[month_name]=df_month_total.loc[:,'Total']

      #----------------------------------------------------------------------
      # Drop Total column 
      #----------------------------------------------------------------------
      df_month_total.drop(['Total'],inplace=True,axis=1)

      #----------------------------------------------------------------------
      # Month columns are aggregated 
      #----------------------------------------------------------------------
      df_monthly_incomes=pd.concat([df_monthly_incomes,df_month_total]\
      , join='outer', axis=1)


   df_monthly_incomes.fillna(0, inplace=True)
   df_monthly_incomes.sample()
   df_monthly_incomes=df_add_cluster(df_monthly_incomes, df_customers.cluster)
   return df_monthly_incomes
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def df_customers_cluster_filter(df, list_cluster):
   """Returns a dataframe with points from which assigned cluster value 
   are into the list of clusters given as parameter function.
      Inputs :
         * df : dataframe with points assigned a cluster
         * list_cluster : list of clusters used to filter points from dataframe.
      Output :
         * dataframe without points that are not assigned a cluster from 
         list_cluster.
   """

   list_all_cluster=np.unique(df.cluster.values)

   print(df.shape)

   for cluster in list_all_cluster:
       if cluster not in list_cluster:
           df=df[df.cluster != cluster ]

   print(df.shape)
   y_sample=df.cluster.values
   return df
#-------------------------------------------------------------------------------


   
