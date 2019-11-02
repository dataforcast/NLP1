#-*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import preprocessing
from sklearn import neighbors

from p3_util import *

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def p4_week_of_month(value) :
   '''Assigns a number for each one of the raw as following : 
      For DAY_OF_MONTH in [ 1,  7] : 1
      For DAY_OF_MONTH in ] 7, 14] : 2
      For DAY_OF_MONTH in ]14, 21] : 3
      For DAY_OF_MONTH in ]21, 28] : 4
      For DAY_OF_MONTH in ]28, ...]: 5
   '''
   if value <= 7 :
      return 1
   elif 7< value and value <= 14 :
      return 2
   elif 14< value and value <= 21 :
      return 3
   elif 21< value and value <= 28 :
      return 4
   elif 28< value :
      return 5
   else :
      return -1
# ------------------------------------------------------------------------------
   

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def p4_mark_none_digit(value) :
   ''' Mark value given as parameter to -1 if this value is not a digit string.
   '''
   if isinstance(value,int) :
      return value
   if isinstance(value,float) :
      return value

   if value.isdigit() :
      return value
   else :
      return '-1'
   # Anyway...
   return value
# ------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p4_convert_cos_to_floathour(cos_value, max_value=1439) :

    hourmn = ( max_value * np.arccos(cos_value) ) / ( 2*3.1416 )

    # Conversion des minutes en heure et minutes
    hour = int(hourmn/60)
    mn = hourmn -hour*60
    
    # Formatage HHMM
    hourmn= float(str(hour)+str(mn))
    return hourmn
#-------------------------------------------------------------------------------      
   
   

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p4_convert_floathour_to_cos(float_value, max_value=1439) :
    ''' Convert float given as parameter into a value issue from cosinus.
    Input :
        dep_time : 1 to 4 digits values in such format : x.0 --> xxxx.0
    Output :
        cosinus value
    '''
    str_value = str(int(float_value))
    str_len = len(str_value)
    
    hour_delimiter = str_len-2
    #print(hour_delimiter)
    if 0  >= hour_delimiter :
        str_min = str_value[:str_len]
        str_hour ='0'
    else :
        str_min = str_value[hour_delimiter:str_len]
        str_hour = str_value[:hour_delimiter]
    #print(str_hour,str_min)
    try :
      mn = int(str_hour)*60+int(str_min)
    except ValueError as valueError :
      print("*** ERROR : p4_convert_floathour_to_cos() : input value can't be converted : "+str(float_value))
    teta = (mn*2*3.1416)/max_value
    return np.cos(teta)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p4_convert_cos_to_integer(cos_value,min_value, max_value) :
   max_value -= min_value
   int_value = int((max_value * np.arccos(cos_value)) /(2*3.1416))
   int_value += min_value
   
   return int_value
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def p4_convert_integer_to_cos(int_value, **kwargs) :
    ''' p4_convert_integer_to_cos(int_value, min_value=minvalue, max_value=maxvalue)
    
    Convert float given as parameter into a value issue from cosinus.
    Input :
        int_value : integer to be converted
    Output :
        cosinus value
    '''
    # Values are shifted from min_value
    PI = np.pi
    min_value = kwargs['min_value']
    max_value = kwargs['max_value']
    int_value -= min_value
    max_value -= min_value

    teta = (int_value*2*PI)/max_value
    return np.cos(teta)
#-------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def p4_train_test_split(df) :
   df_droped = df.copy()
   
   #----------------------------------------------------------------------------
   # Calcul des etiquettes
   #----------------------------------------------------------------------------
   df_droped['LABEL'] = df['CRS_ARR_TIME'] - df['ARR_TIME']
   
   
   #----------------------------------------------------------------------------
   # Purge des colonnes ayant servies a calculer les étiquettes
   #----------------------------------------------------------------------------
   df_droped = df_droped.drop(labels='CRS_ARR_TIME', axis=1, inplace=False)
   df_droped = df_droped.drop(labels='ARR_TIME', axis=1, inplace=False)
   
   
   #----------------------------------------------------------------------------
   # Calcul des données des jeux d'entraînement et de test
   #----------------------------------------------------------------------------
   X_train = df_droped[df_droped['DAY_OF_MONTH']<=20]
   y_train = df_droped[df_droped['DAY_OF_MONTH']<=20]['LABEL']

   X_test = df_droped[df_droped['DAY_OF_MONTH']>20]
   y_test = df_droped[df_droped['DAY_OF_MONTH']>20]['LABEL']
   
   
   #----------------------------------------------------------------------------
   # Purge des colonnes des etiquettes 
   #----------------------------------------------------------------------------
   X_train = X_train.drop(labels='LABEL', axis=1, inplace=False)
   X_test = X_test.drop(labels='LABEL', axis=1, inplace=False)
   
   return X_train,X_test,y_train,y_test

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def p4_train_test_split_from_column(df, separator_column, label_column,train_limit) :
   '''Returns  X_train,X_test,y_train,y_test from dataframe given as parameter.
   Input : 
      df : will allow  to build X_train and X_test data.
      separator_column values are used in order to split train and test data.
      train_limit is the separator criteria.
      y : column containing values for labels
   
   '''

   #----------------------------------------------------------------------------
   # Calcul des données des jeux d'entraînement et de test
   #----------------------------------------------------------------------------
   X_train = df[df[separator_column]<=train_limit]
   X_test = df[df[separator_column]>train_limit]

   y_train = df[df[separator_column]<=train_limit][label_column]
   y_test = df[df[separator_column]>train_limit][label_column]
   if False :
      #----------------------------------------------------------------------------
      # Purge de la colonne ayant servie comme étiquette
      #----------------------------------------------------------------------------
      X_train = df_drop_column(X_train,label_column)
      X_test  = df_drop_column(X_test,label_column)
      
      #----------------------------------------------------------------------------
      # Standardisation des données sur la base des données dans X_train
      #----------------------------------------------------------------------------
      std_scale = preprocessing.StandardScaler().fit(X_train)
      X_train_std = std_scale.transform(X_train)
      X_test_std = std_scale.transform(X_test)

   return df, X_train, X_test, y_train, y_test

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def p4_df_read_from_list_month(path_name, month, list_skip_rows=None) :
   ''' Read data files matching with list_month and return 
   a prepared dataframe.
   
   Columns that are not digit are excluded from dataframe.
   Last 9 columns are excluded from dataframe.
   '''
   #----------------------------------------------------------------------------
   # Check parameters
   #----------------------------------------------------------------------------
   df_dealays = pd.DataFrame()

   path_name_month = path_name+"_"+str(month)+".csv"
   
   try :
      df_dealays = pd.read_csv(path_name_month, delimiter=','\
      ,low_memory=False,skiprows=list_skip_rows)
      #----------------------------------------------------------------------------
      # Drop last 9 columns
      #----------------------------------------------------------------------------
      df_dealays = df_dealays.iloc[:,:-9]
      
   except pd.errors.ParserError as parserError:
      print("*** ParserError raised : "+str(parserError))
      return None, None
   except FileNotFoundError as fileNotFoundError :
      print("*** FileNotFoundError raised : "+str(fileNotFoundError))
      return None, None
   print("Month "+month+" loaded!")

   #----------------------------------------------------------------------------
   # Data preparation
   #----------------------------------------------------------------------------
   df_digit,list_col_notdigit = df_get_digital_columns(df_dealays) 

   #----------------------------------------------------------------------------
   # Drop last 9 columns
   #----------------------------------------------------------------------------
   #df_digit_restricted = df_digit.iloc[:,:-9]
   
   
   return df_dealays, list_col_notdigit
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def p4_df_read_from_list_month_deprecated(path_name, list_month) :
   ''' Read data files matching with list_month and return 
   a prepared dataframe.
   
   Columns that are not digit are excluded from dataframe.
   Last 9 columns are excluded from dataframe.
   '''
   #----------------------------------------------------------------------------
   # Check parameters
   #----------------------------------------------------------------------------
   df_dealays = pd.DataFrame()
   for month in list_month :
      path_name_month = path_name+"_"+str(month)+".csv"
      try :
         df_dealays = pd.read_csv(path_name_month, delimiter=','\
         ,low_memory=False, skiprows= list_rows_skip)
         #----------------------------------------------------------------------------
         # Drop last 9 columns
         #----------------------------------------------------------------------------
         df_dealays = df_dealays.iloc[:,:-9]
         
      except pd.errors.ParserError as parserError:
         print("*** ParserError raised : "+str(parserError))
         continue
      print("Month "+month+" loaded!")

   #----------------------------------------------------------------------------
   # Data preparation
   #----------------------------------------------------------------------------
   df_digit,list_col_notdigit = df_get_digital_columns(df_dealays) 

   #----------------------------------------------------------------------------
   # Drop last 9 columns
   #----------------------------------------------------------------------------
   #df_digit_restricted = df_digit.iloc[:,:-9]
   
   
   return df_dealays, list_col_notdigit
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def knn_cv_search(X_train, y_train, list_neighbors=None, cv_parameter=5\
   , scoring_parameter='accuracy', limit_list=(3,11) ):
   '''Search for best neighbours count for KNN classifier.
   Best number is provided from best MSE score over all cross-validations
   '''
   #----------------------------------------------------------------------------
   # Creation d'une liste de nombre de voisins impairs
   #----------------------------------------------------------------------------
   if list_neighbors is None :
      myList = list(range(limit_list[0],limit_list[1]))

      filtered_neighbors = filter(lambda x: x % 2 != 0, myList)
      list_neighbors = list(filtered_neighbors)
   else :
      pass

   #----------------------------------------------------------------------------
   # Liste contenant les scores moyens de la recherche croisée (CV)
   #----------------------------------------------------------------------------
   list_cv_mean_scores = list()

   min_index = 0
   scores_mean = 0.0
   import time
   t0 = time.time()

   #----------------------------------------------------------------------------
   # Search for best neighbors count over folds
   #----------------------------------------------------------------------------
   for neighbor in list_neighbors:
       knn_clf = neighbors.KNeighborsRegressor(n_neighbors=neighbor)

       
       # knn_clf = KNeighborsClassifier(n_neighbors=neighbor)
       # -----------------------------------------------------------------------
       # Get all scores over all cross validations folds
       # -----------------------------------------------------------------------

       scores = cross_val_score(knn_clf\
       ,X_train, y_train, cv=cv_parameter, scoring = scoring_parameter)

       # -----------------------------------------------------------------------
       #Get mean of this scores for the given neighbor
       # -----------------------------------------------------------------------
       list_cv_mean_scores.append(scores.mean())
   
   print("KNN classifier: Elapsed time for searching best neighbors number= %0.3fs" % (time.time()-t0))

   #----------------------------------------------------------------------------
   # Erreur de classification minimale
   #----------------------------------------------------------------------------
   if scoring_parameter=='accuracy' or scoring_parameter == 'r2':
      #-------------------------------------------------------------------------
      # Le meilleur score va a la valeur la plus proche de 1, signant ainsi 
      # une plus grande précision.
      #-------------------------------------------------------------------------
      list_score = [1 - x for x in list_cv_mean_scores]
   else :
      #-------------------------------------------------------------------------
      # Le meilleur score va a la valeur la plus faible, signant une moindre 
      # perte.
      #-------------------------------------------------------------------------
      list_score = list_cv_mean_scores
      
   min_index = list_score.index(min(list_score))
   
   #----------------------------------------------------------------------------
   # Extraction du meilleur nombre de voisins
   #----------------------------------------------------------------------------
   best_neighbors = list_neighbors[min_index]
   print( "Optimal number for neighbors= %d" % best_neighbors)
   return best_neighbors, list_neighbors, list_score
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def p4_df_plot_delay_route_parameter(df,route,digit_parameter, parameter_rotation=90) :
    ser_to_carrier = df[df['DEST_AIRPORT_ID']==route[0]][digit_parameter]
    ser_from_carrier = df[df['ORIGIN_AIRPORT_ID']==route[1]][digit_parameter]

    # Concatenation des dataframes d'origine et de destination
    df1 = pd.concat([ser_from_carrier, ser_to_carrier], axis=1, join_axes=[ser_to_carrier.index])

    # Concatenation des dataframes d'origine de destination et de retard
    df2 = pd.concat([df1, df['ARR_DELAY']], axis=1, join_axes=[df1.index])

    df2.dropna(axis=1, inplace=True)    

    # Calcul de la moyenne des retards par carrier
    df_plot  = df2.groupby(digit_parameter).mean()
    plt.figure(figsize=(10,10))

    x = df_plot.index.values
    y = df_plot['ARR_DELAY'].values
    plt.bar(x, height= y)
    plt.ylabel("Retards en mn")
    plt.xticks(x,rotation=parameter_rotation);
    origin = p4_df_get_listName_from_listCode(df,[route[0]],ref_code='ORIGIN_AIRPORT_ID')
    dest   = p4_df_get_listName_from_listCode(df,[route[1]],ref_code='DEST_AIRPORT_ID')
    z_=plt.title("Retards moyens entre "+origin[0]+" / "+dest[0]+" par type= "+str(digit_parameter))    
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def p4_df_get_listName_from_listCode(df,list_code, ref_code='ORIGIN', ref_name='ORIGIN_CITY_NAME') :
   list_name = list()
   for code in list_code :
      arr_name = df[df[ref_code]==code][ref_name].unique()
      list_name.append(arr_name[0])
   return list_name
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def p4_df_get_city_item_list(df) :
   df2 = df['ORIGIN','ORIGIN_CITY_NAME']

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def p4_df_get_carrier_from_code(df,list_carrier_code) :
   #list_city_name = list()
   #for carrier_code in list_carrier_code :
   #   city_name = df[df['UNIQUE_CARRIER']==carrier_code]['ORIGIN_CITY_NAME'].unique()
   #   list_city_name.append(city_name[0])
   return list_carrier_code
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def p4_df_get_feature_count_from_threshold(df,column,parameter_threshold, direction=1):
   ''' Returns a pandas Series structure issued from column in df dataframe.

   Features with count values greater then parameter_threshold are kept in 
   pandas Serie to be returned.
   
   direction parameter allows to change criteria direction : '>', '<' or '=='
   
   '''
   ser_result = None
   if column in df : 
      # ------------------------------------------------------------------------
      # Get values from column > parameter_threshold
      # ------------------------------------------------------------------------
      if direction == 1 :
         ser_filter = df[column].value_counts()>parameter_threshold
      elif direction == -1 :
         ser_filter = df[column].value_counts()<parameter_threshold
      elif direction == 0 :
         ser_filter = df[column].value_counts()==parameter_threshold
      else :
         print("*** ERROR : invalid value for direction = "+str(direction))

      # ------------------------------------------------------------------------
      # For each one of the indexes, get values count from column 
      # ------------------------------------------------------------------------
      ser  = df[column].value_counts()

      # ------------------------------------------------------------------------
      # Keep only index mathcing with parameter_threshold criteria, using 
      # ser_filter
      # ------------------------------------------------------------------------
      ser_result = ser[ser_filter]
      
   else :
      print("*** ERROR : no column= "+str(column)+" in dataframe!")
      
   return ser_result
# ------------------------------------------------------------------------------

