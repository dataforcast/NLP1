#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn.metrics

from sklearn import decomposition
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
from sklearn.manifold import MDS

from sklearn import preprocessing

from sklearn import manifold

from sklearn.metrics import pairwise
from sklearn.metrics import silhouette_samples 
from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import pairwise_distances

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

list_scoring_save = [
  'director_facebook_likes'
, 'actor_1_facebook_likes'
, 'actor_2_facebook_likes'
, 'actor_3_facebook_likes'
, 'num_critic_for_reviews'
, 'num_voted_users'
, 'cast_total_facebook_likes'
, 'num_user_for_reviews'
, 'imdb_score'
, 'movie_facebook_likes']

list_scoring_actors = ['director_facebook_likes'
, 'actor_1_facebook_likes'
, 'actor_2_facebook_likes'
, 'actor_3_facebook_likes']



# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_boxplot_min_max(df , nutrient) :
    """Retourne les valeurs extremes des moustaches d'une variable.
    """
    z = pd.DataFrame(df[nutrient]).describe()
    
    q1 = z.loc['25%',nutrient]
    q3 = z.loc['75%',nutrient]

    # Calcul des moustaches
    zmin1 = z.loc['min',nutrient]
    zmin2 = q1-(q3-q1)*1.5
    zmin = max(zmin1,zmin2)

    zmax1 = z.loc['max',nutrient]
    zmax2 = q3+(q3-q1)*1.5
    zmax = min(zmax1,zmax2)

    return zmin,zmax
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_boxplot_limits(df , col) :
    """Retourne les valeurs des moustaches d'une variable, 
    ainsi que les quantiles q1 et     q3.
    """
    z = pd.DataFrame(df[col]).describe()
    
    q1 = z.loc['25%',col]
    q3 = z.loc['75%',col]

    # Calcul des moustaches
    zmin1 = z.loc['min',col]
    zmin2 = q1-(q3-q1)*1.5
    zmin = max(zmin1,zmin2)

    zmax1 = z.loc['max',col]
    zmax2 = q3+(q3-q1)*1.5
    zmax = min(zmax1,zmax2)

    return q1,q3,zmin,zmax
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_del_list_column(df_param, list_column) :
    df = df_param.copy()    
    for column in list_column :
        if column in df.columns :
            del(df[column])
    return df
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def ser_get_list_genre(ser_genre) :
    list_genre = list()
    listAllGenres = list()
    for  genre in ser_genre.values :
        list_genre = genre.split('|')
        listAllGenres = list(set(listAllGenres + list_genre))
        #list_all_genre.extend(list_genre)
    listAllGenres.sort()
    return listAllGenres
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_clean_list_column(df, list_column):
    for column in list_column :
        if column in df.columns :
            del(df[column])
    return df
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_get_digital_columns(df) :
    ser_types = df.dtypes
    list_col_notdigit = list()
    list_col_digit = list()

    for col_name, item in ser_types.iteritems() :
        if item != 'float64' and item != 'int64':
            list_col_notdigit.append(col_name)
        else:
            list_col_digit.append(col_name)

    # Filtrage des colonnes numériques    
    df = df.loc[:,list_col_digit]
    return df,list_col_notdigit
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_one_hot_encode(df,list_col, dict_filtered_value=None):
    ''' La liste des valeurs dans la colonne passée en paramètre sont utilisées 
    comme nouvelles colonnes.
    Le nombre de colonnes du dataframe augmente.
    L'encodage one-hot  est ensuite appliqué à toutes les lignes du 
    dataframe comme suit : 
        * La valeur est 1 si la colonne avait une valeur correspondant 
        a une nouvelle colonne, 0 sinon.
        * La colonne qui a fait l'objet de l'encodage est purgée du dataframe.

    Quand dict_filtered_value est != None, alors l'encodage des colonnes definies 
    comme clé du dictonnaire est realisé uniquement sur les valeurs de la liste 
    associée a cette clé.
    '''
    dict_list_new_column = dict()
    print("Size of incoming list = {}".format(len(list_col)))
    for column in list_col :
        # Les colonnes de la liste passée en parametre sont extraites du 
        # dataframe passé en parametre.
        if column in df.columns :
            ser_list_col = df.loc[:,column]
        
            # La liste des valeurs qui composent la colonne est rendue unique
            list_new_column = list()
            list_value_unique2 = ser_list_col.unique().tolist()
            print("Before filtering : nb of values to encode = {}".format(len(list_value_unique2)))

            if dict_filtered_value is None :
               # No filter exists for values to be encoded
               list_value_to_encode = list_value_unique2.copy()
            else :
               if column in dict_filtered_value.keys() :
                  # Values to be encoded belong to list of values to be filtered 
                  list_filter_value = dict_filtered_value[column]
                  list_value_to_encode = \
                  [val for val in list_value_unique2 if val in list_filter_value]
               else :
                  # Values to be encoded do not belong to list of values 
                  # to be filtered 
                  list_value_to_encode = list_value_unique2.copy()

            #print("After filtering : nb of values to encode = {}".format(len(list_value_to_encode)))
            
            # New columns are renamed with white spare repaced with '_' character.
            for val in list_value_to_encode :
               try : 
                  list_new_column.append(val.strip().replace(' ','_'))
               except AttributeError as attributeError:
                  print("Value triggering AttributeError = {}".format(val))

            # La liste des colonnes est ajoutée au dataframe
            df = df_add_list_column(df, list_new_column)

            #print("After adding columns: nb of encoded columns= {}".format(len(list_new_column)))


            # One hot encoding : les valeurs des colonnes dans la liste 
            # list_new_column sont mises a 1; les autres sont mises à 0.
            if False :
               # Old algorithm
               for index in df.index :
                   content_value = df.loc[index,column]
                   if content_value in list_new_column :
                       df.loc[index,content_value] = 1
                   else :
                       df.loc[index,content_value] = 0
            else :
               # New algorithm
               for col_value in list_value_to_encode :
                  # Getting all indexes for which column is assigned with
                  # col_value
                  col_value_index = df[df[column]== col_value ].index
                  list_index = col_value_index.tolist()
                  
                  # Getting new column name from col_value
                  new_col_name = col_value.strip().replace(' ','_')
                  
                  # Assigning 1 to all indexes in new column name
                  for index in list_index :
                     df.loc[index,new_col_name] = 1

            # New columns name are recorded into a dictionnary
            dict_list_new_column[column] = list_new_column
            del(df[column])
        else :
            pass
        
    return df, dict_list_new_column
# ------------------------------------------------------------------------------

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def df_encode_with_separator(df, column, separator='|') :
    ''' Cette fonction substitue les valeurs de la colonne genres 
    à la colonne genres .
    Le nombre de colonnes croit.
    Un encogage one-hot est appliqué aux nouvelles colonnes.
    '''
    #column = 'genres'
    
    if column not in df.columns : 
      print("\n WARNING : no column = '"+column+"' Found!")
      return df
    
    # Parcours de toutes les lignes du dataframe
    for ind in df.index :  
        # Extraction de la valeur sur la colonne genres correspondant a la ligne ind
        mydf = df.loc[[ind],[column]]
        value = mydf[column].values[0]
        
        # Les genres aggrégés avec le caractere '|' est splitté en une liste
        list_column = value.split(separator)
        #print(list_column)
        # Les éléments de la liste sont ajoutés aux colonnes du dataframe
        df  = df_add_list_column(df,list_column)
        
        # Pour la ligne ind, encodage de toutes les colonnes du dataframe 
        # correspondant aux elements de list_column
        for item in list_column :
            if item in df.columns :
                df.loc[[ind],[item]] = 1
            else : 
                df.loc[[ind],[item]] = 0

    # Purge de la colonne 'genres'
    if column in df.columns :
        del(df[column])
    #print(df.shape)
    return df
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_encode_genres(df, splitter='|') :
    ''' Cette fonction substitue les valeurs de la colonne genres 
    à la colonne genres .
    Le nombre de colonnes croit.
    Un encogage one-hot est appliqué aux nouvelles colonnes.
    '''
    column = 'genres'
    
    if column not in df.columns : 
        return df
    
    # Parcours de toutes les lignes du dataframe
    for ind in df.index :  
        # Extraction de la valeur sur la colonne genres correspondant a la ligne ind
        mydf = df.loc[[ind],[column]]
        value = mydf[column].values[0]
        
        # Les genres aggrégés avec le caractere '|' est splitter en une liste
        list_column = value.split(splitter)
        
        # Les éléments de la liste sont ajoutés aux colonnes du dataframe
        df  = df_add_list_column(df,list_column)
        
        # Pour la ligne ind, encodage de toutes les colonnes du dataframe 
        # correspondant aux elements de list_column
        for item in list_column :
            if item in df.columns :
                df.loc[[ind],[item]] = 1
            else : 
                df.loc[[ind],[item]] = 0

    # Purge de la colonne 'genres'
    if column in df.columns :
        del(df[column])
    return df
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_add_list_column(df, list_item):
    for column in list_item :
        if column not in df.columns :
            df[column] = pd.Series(np.zeros(df.shape[1]-1))
            df[column] = 0
    return df
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_colvalue_replace(df,column,replaced,replacer) : 
    '''
    Values from Series df[column] that are equal to value=replaced  are replaced
    with value replacer.
    '''
    if column in df.columns :
        df[column]=df[column].fillna(replacer)
        #for index, value in df[column].iteritems():
        #    if value == replaced or type(value) is not type(replacer):
        #       df[column][index] = replacer
        #    else :
        #        pass
    else :
        pass

    return df
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_colvalue_replace_by_mean(df, column) :
    if column in df.columns :
        print('In..')
        df[column].fillna(df[column].median())
    return df
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_moovies_scoring_build(df_moovies, list_scoring) :

   df_moovies_scoring = pd.DataFrame()
   
   for column in list_scoring :
      if column in df_moovies :
         df_moovies_scoring[column] = df_moovies[column]

   print(df_moovies_scoring.shape)
   return df_moovies_scoring
   
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_listcol_replace_value(df,list_col,replaced_value, replacer_value) :
    '''For each column in list_col, values from df[column] that are 
    equal to replaced_value are replaced with replacer_value.
    '''
    for column in list_col :
      if column in df.columns :
         print("Feature "+column+" In progress...")
         for index, value in df[column].iteritems():
            #if value == replaced_value or type(value) is not type(replaced_value):
            #if value == replaced_value or np.isnan(replaced_value) is True:
            if np.isnan(value) or value == replaced_value :
               #print(str(df[column][index]) + " is replaced with "+str(replacer_value))
               df[column][index] = replacer_value
            else :
                pass
         else :
            pass
    return df
# ------------------------------------------------------------------------------

    
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def ser_replace_value(ser,replaced_value, replacer_value) :
   '''For each values from ser given as parameter that are equal to 
   replaced_value those values are replaced with replacer_value.
   '''
   for index, value in ser.iteritems():
      try :
         if np.isnan(value) or value == replaced_value :
            #print(" {} is replaced with {}".format(value, replacer_value))
            ser[index] = replacer_value
         else :
             pass
      except TypeError :
         pass
 
   return ser[index]
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 
# ------------------------------------------------------------------------------
def get_dict_dict_value_threshold(df, threshold = 0.6) :
   '''Returns a dictonary containing dictionaries.
   Returned dictionary keys are columns (features) from dataframe df.
   Returned dictionary contain values that are dictionaries 
   Contained dictionary values are dictionary structures as following :
      * {feature_1: pearson_coefficient_1},..,{feature_2: pearson_coefficient_2} 
   '''

   dict_dict_value_threshold = dict()
   
   for column, ser in df.items():
       dict_value_threshold = dict()
       for index, value in ser.items():
           if value >= threshold and index != column :
               dict_value_threshold[index] = value 

       if 0 < len(dict_value_threshold) :# and column not in dict_value_threshold.keys() :
           dict_dict_value_threshold[column] = dict_value_threshold
        
    
   return dict_dict_value_threshold
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# 
# ------------------------------------------------------------------------------
def dict_dict_value_threshold_add_pearson(dict_dict_value_threshold, 
   correlation_threshold_value):
   '''Returns a dictonary containing dictionaries.

      Input dictionat structure : 
      * {feature : {feature_1: pearson_coefficient_1},..,{feature_2: pearson_coefficient_2}}

      Returned dictionay structure :
      * {feature : {feature_1: pearson_coefficient_1},..,{feature_2: pearson_coefficient_2},
                    'pearson' : {feature_k : pearson_coefficient_max}}

   '''
   for key in dict_dict_value_threshold.keys() :
       # Recuperation du plus grand coefficient de Pearson des colonnes corrélées avec key
       c_pearson = 0.0
       dict_value_threshold  = dict_dict_value_threshold[key]
       for key2 in dict_value_threshold.keys() :
           if dict_value_threshold[key2] > c_pearson :
               max_key2 = key2
               c_pearson = dict_value_threshold[key2]
               
       dict_value_threshold['pearson'] = {max_key2:c_pearson}
       dict_dict_value_threshold[key] = dict_value_threshold

   return dict_dict_value_threshold    
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def dict_dict_value_threshold_add_theta(dict_dict_value_threshold, df):
   '''Returns a dictonary containing dictionaries.

      Input dictionat structure : 
      * {feature : {feature_1: pearson_coefficient_1},..,{feature_2: pearson_coefficient_2},
                    'pearson' : {feature_k : pearson_coefficient_max}}

      Returned dictionay structure :
      * {feature : {feature_1: pearson_coefficient_1},..,{feature_2: pearson_coefficient_2},
                    'pearson' : {feature_k : pearson_coefficient_max}
      *             'theta'   : {feature_k : theta_array}}
         theta =[[a,b]] for y = a*x+b
   
   '''
   for column in dict_dict_value_threshold.keys() :   
       dict_value_threshold = dict_dict_value_threshold[column]
       dict_pearson = dict_value_threshold['pearson']
       for feature  in dict_pearson.keys():    
           # On décompose le dataset et on le transforme en matrices pour pouvoir effectuer notre calcul    
           X = np.matrix([np.ones(df.shape[0]), 
           df[column].as_matrix()]).T
           
           y = np.matrix(df[feature]).T

           # On effectue le calcul exact du paramètre theta
           theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
           break
       dict_value_threshold['theta'] = {feature:theta}
       dict_dict_value_threshold[column] = dict_value_threshold
   return dict_dict_value_threshold
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_substitute_nan_by_prediction(df, dict_dict_value_threshold) :
   '''
      Substitute Nan values with predicted values using linear regression 
      result.
   '''

   # Constitution de la liste des features dont les valeurs a NaN vont être 
   # prédites par regression ienaire et remplcées.
   list_feature = list(dict_dict_value_threshold.keys())

   df_1 = df[list_feature]

   #df_1_tmp, list_column_to_drop = df_clean_nan(df_1, verbose=False, action=True)
   #print(list_column_to_drop)    


   for feature in df_1.columns :
       dict_value_threshold = dict_dict_value_threshold[feature]
       dict_theta = dict_value_threshold['theta']
       for index, value in df_1[feature].iteritems() :
           if np.isnan(value) :
               for feature2 in dict_theta.keys() :
                   theta = dict_theta[feature2]
                   #print(feature, feature2, theta)
                   # Calcul de prédiction par regression linéaire
                   x = df_1.loc[index , feature2]
                   y_predicted = theta[1]*x + theta[0]
                   # Y = theta[1]*X + theta[0]
                   # X = (Y - theta[0])/theta[1]
                   #x_predicted = max(0,(y - theta[0])/theta[1])
                   df_1.at[index , feature] = y_predicted[0,0]
                   #print(x,y_predicted[0,0])

                   
                   break
                
            #dict_theta = dict_value_threshold['theta']
            #theta = dict_theta[index]

   for column in df_1 :
      del(df[column])
      df[column] = df_1[column]
   return df
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_get_std_scaled_values(df) :
   X = df.values
   X_scaled = None
   std_scale = preprocessing.StandardScaler().fit(X)
   X_scaled = std_scale.transform(X)
   return X_scaled
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_get_normalized_values(df) :
   X = df.values
   X_normalized = (X - X.mean()) / (X.max() - X.min())
   return X_normalized
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def clustering_compute_metrics(X_train, labels_trained, labels_predicted, md=False) :
    '''Compute and display metrics related to clusters labels.
    Input : 
        X_train : data that has been used to build clustered
        labels_trained : labels of clusters 
        labels_predicted : labels of clusters that have been built with data for prediction.
    '''

    try :
        if md is True :
           printmd("Homogeneity : %0.3f" % homogeneity_score(labels_trained, labels_predicted))
           printmd("Completeness: %0.3f" % completeness_score(labels_trained, labels_predicted))
           printmd("V-measure   : %0.3f" % v_measure_score(labels_trained, labels_predicted))
           printmd("ARI         : %0.3f" % adjusted_rand_score(labels_trained, labels_predicted))
           printmd("AMI         : %0.3f" % adjusted_mutual_info_score(labels_trained, labels_predicted))
           printmd("Silhouette  : %0.3f" % silhouette_score(X_train, labels_predicted))    
        else :
           print("Homogeneity : %0.3f" % homogeneity_score(labels_trained, labels_predicted))
           print("Completeness: %0.3f" % completeness_score(labels_trained, labels_predicted))
           print("V-measure   : %0.3f" % v_measure_score(labels_trained, labels_predicted))
           print("ARI         : %0.3f" % adjusted_rand_score(labels_trained, labels_predicted))
           print("AMI         : %0.3f" % adjusted_mutual_info_score(labels_trained, labels_predicted))
           print("Silhouette  : %0.3f" % silhouette_score(X_train, labels_predicted))    
        
    except ValueError as valueError:
        errorMessage = "ERROR : {}".format(valueError)
        if md is True :
           printmd_error(errorMessage)
        else :
           print("***"+errorMessage)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_get_value_from_raw(df, raw, raw_value, column, md=False) :
    if raw in df.columns :
        ser_value = df[df[raw]==raw_value][column]
        value = ser_value[ser_value.index[0]]
    else :
        if md is True :
          printmd_error("ERROR : no variable = {} from dataframe columns".format(raw))
        else :
          print("ERROR : no variable = {} from dataframe columns".format(raw))
        return -1.0
    return value
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_column_add(df, df_source, column, md=False) :
    ''' Add column value from df_source to df.'''
    
    if column in df : 
        del(df[column])

    if md is True :
       printmd(df.shape)
    else :
       print(df.shape)
       
    for ind in df_source.index :
        if ind in df.index :
            df.loc[ind,column] = df_source.loc[ind,column]
    
    if md is True :
       printmd(df.shape)
    else :
       print(df.shape)
    return df
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_compute_norml2(df) :
    if 'norml2' in df :
        del(df['norml2'])

    df_norml2 = pd.DataFrame(np.sqrt(np.square(df).sum(axis=1)), columns =['norml2'] ,index=df.index )
    max_norml2 = df_norml2['norml2'].sum()
    df_norml2['norml2'] = df_norml2.apply(lambda x : x/x.sum())
    df = pd.concat([df, df_norml2],axis=1)

    df.sort_values(by='norml2',ascending=False)['norml2']
    return df
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_dimensions(df) :
    print(df.shape)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_drop_raws_from_nan_value(df, column) :
   """Drop whole raw where nan value stand in column."""
   if column not in df :
      print("*** WARNING : column {} not in dataframe".format(column))
      return df

   df[column].dropna(axis=0 , inplace=True)

   return df
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_clean_nan(df, verbose=True, action=True) :
   ''' For all columns and for all values inside each column:
   if value is nan, then line is dropped.
   
   Input : 
      df : dataframe containing digital values only.
      verbose : when set to True, this flag allows to display column from 
      where cvalues are dropped.
      action : when flag is fixed to True, then raw is dropped.
   Output :
      df : cleaned dataframe
      list_dropped_unique : list of columns that have been dropped.
   '''
   list_dropped = list()
   drop_raws = 0
   for column in df :
      ser = df[column]
      for index, value in ser.iteritems() :
         if isinstance(value,float) and np.isnan(value):
             if verbose is True:
                 print(column,value)
             if action is True:
                 df = df.drop(index=index, inplace=False)
                 drop_raws+=1
             list_dropped.append(column)

   ser = pd.Series(list_dropped)
   list_dropped_unique = ser.unique().tolist()
   print("Number of droped raws = "+str(drop_raws))
   return df.copy(),list_dropped_unique
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_replace_nan_with_value_type(df_, column, value_type='zero') :
   '''Replace Nan value in column with value defined by value_type parameter.
   value_type : 
      'zero' : nan value is replaced with 0
      'mean' : nan value is replaecd with mean value from column
      'median' : nan value is replaecd with median value from column
   '''
   
   df = df_.copy()

   if column not in df :
      print("*** ERROR : column \'{}\' not in dataframe".format(column))
      return df

   ser = df[column]
   arr_ser = np.array(ser)
   
   if value_type == 'zero' :
      replaced_value = 0
   elif value_type == 'mean' :
      replaced_value = np.nanmean(arr_ser)
   elif value_type == 'median' :
      replaced_value = np.nanmedian(arr_ser)
   else :
      print("*** ERROR : Unkown value type= {}".format(value_type))
      return df

   # ---------------------------------------------------------------------------
   # Filter of nan value   
   # ---------------------------------------------------------------------------
   where_nan_index = np.isnan(arr_ser)

   # ---------------------------------------------------------------------------
   # Nan values are replaced following index
   # ---------------------------------------------------------------------------
   arr_ser[where_nan_index] = replaced_value
   
   # ---------------------------------------------------------------------------
   # Replace column from dataframe
   # ---------------------------------------------------------------------------
   df = df.drop(labels=column, axis=1, inplace=False)
   df[column] = pd.Series(arr_ser)

   return df
# ------------------------------------------------------------------------------
   
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_encode_column(df,column) :

   if column not in df : 
      print("*** WARNING : column {} not in dataframe".format(column))
      return df

   #----------------------------------------------------------------------------
   # Encoded array is built from dataframe column
   #----------------------------------------------------------------------------
   labelencoder=LabelEncoder()
   labelencoder.fit(df[column].tolist())
   arr_encoded = labelencoder.transform(df[column])
   
   #----------------------------------------------------------------------------
   # Column with values to be encoded is relaced with encoded array.
   #----------------------------------------------------------------------------
   print(df.shape)
   df = df.drop(labels=column, axis=1, inplace=False)
   print(df.shape)   
   df[column] = arr_encoded
   print(df.shape)
   
   return df
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_drop_column(df,column) :
   if column not in df : 
      print("*** WARNING : column {} not in dataframe".format(column))
      return df

   df = df.drop(labels=column, axis=1, inplace=False)
   
   return df
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_drop_list_column(df,list_column) :
   for column in list_column :
      df =    df_drop_column(df,column)
   return df
# ------------------------------------------------------------------------------

