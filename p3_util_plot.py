#-*- coding: utf-8 -*-

from IPython.display import Markdown, display

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sklearn.metrics

from   mpl_toolkits import mplot3d

#%matplotlib inline

import seaborn as sns

from sklearn import decomposition
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

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

from p3_util import *

medianprops = {'color':"black"}
meanprops = {'marker':'o', 'markeredgecolor':'black',
            'markerfacecolor':'firebrick'}

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def printmd_(string, color):
    formatedstr = "<p><font color='"+color+"'>"
    formatedstr += "**"
    formatedstr +=str(string)
    formatedstr += "**"
    formatedstr += "</font></p>"
    #print(formatedstr)
    display(Markdown(formatedstr))
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def printmd_error(string):
    printmd_(string,'red')
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def printmd(string):
    printmd_(string,'green')
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_boxplot_list_display(df, list_var, show_outliers=False ) :
    """ Display boxplot of filtered columns from dataframe given as parameter.
    Filtered columns belongs to list_var.
    Statistics dispersions are also displayed.
    """

    for var in list_var :
        print("="*60)
        print(str(var).center(60,'-'))
        print("Moyenne: {}".format(df[var].mean()))
        print("Mediane: {}".format(df[var].median()))
        print("Modes: {}".format(df[var].mode()))
        print("Variance: {}".format(df[var].var(ddof=0)))
        print("Ecart:{}".format(df[var].std(ddof=0)))
        df.boxplot(column=var, vert=False, showfliers=show_outliers, patch_artist=True, medianprops=medianprops,showmeans=True, meanprops=meanprops)
        plt.show()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_boxplot_display(df, column) :
    z = pd.DataFrame(df[column]).boxplot(showfliers= False, patch_artist=True\
    , medianprops=medianprops,showmeans=True, meanprops=meanprops)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_display_linear_regression_from_list(df_param, var_pivot,list_var) :
    """ Display linear regression between variable var_pivot given as parameter 
    and variables included into list_var given as parameter.
    """
    df_dimensions(df_param)
    for var in list_var : 
        df = df_param.copy()
        if var in df.columns :
            zmin, zmax = df_boxplot_min_max(df , var)
            df = df[df[var]<=zmax ]
            df = df[df[var]>=zmin ]
            df = df[df[var]>0.0]
            df_sns_joint_plot(df, var_pivot, var, parameter_color='grey')
            df_dimensions(df_param)
# ------------------------------------------------------------------------------

# 


# ------------------------------------------------------------------------------
def df_sns_joint_plot(df, var1, var2, parameter_kind='reg', parameter_color='grey') :
    with sns.axes_style('white') :
        sns.jointplot(var1, var2, data=df, kind = parameter_kind\
        , color = parameter_color)
# ------------------------------------------------------------------------------
        
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_display_hist_from_list(df_food, list_columns) :
    """ Display histograms issued from dataframe and filetered from list given 
    as parameter.
    Histogram represents frequencies from dataframe for each column from 
    list_columns.
    """
    z = plt.figure(figsize=(4,4))
    for column in list_columns :
        df = df_food.copy()
        zmin, zmax = df_boxplot_min_max(df, column)
        if zmin < zmax :
            list_name = remove_pattern([column],'100g')
            new_column = list_name[0]
            df.rename(columns={column: new_column}, inplace=True)
            column = new_column
            df = pd.DataFrame(df[column], index=df.index)
            df = df[df[column] <= zmax]
            df = df[df[column] >= zmin]
            df = df[df[column] > 0.0]
            #z = plt.figure()
            z = df.plot.hist(bins=50)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_display_dict_hist(dict_to_display, title, xlabel, ylabel, fontSize=12\
, color='blue'):
    """Display content from a dictionary where :
    * keys from dictionary are mapped on X axe 
    * values from dictionary are mapped on Y axe
    """
    # Gestion de l'affichage
    fig, ax = plt.subplots(figsize=(10,10))

    plt.rcParams.update({'font.size': fontSize})


    plt.title(title)
    ax.set_xlabel(xlabel, fontsize = fontSize, color=color)
    ax.set_ylabel(ylabel, fontsize = fontSize, color=color)
    ax.tick_params(axis='x', rotation=90)

    list_values   = dict_to_display.values()
    bar_locations = dict_to_display.keys()

    z = plt.bar(bar_locations, list_values, align='center', width=0.5)
    return
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def ser_hist(ser,title,xlabel, ylabel,param_bins=100, param_font_size=12, \
             param_rotation=90,figsize=(10,10)):
   '''Display histogram issued from Serie given as parameter 
   '''
   
   ax = ser.hist(bins=param_bins,figsize=figsize)
   z_ = ax.set_title(title, fontsize=param_font_size,color='b')
   z_ = ax.set_xlabel(xlabel,color='b')

   ax.tick_params(axis='x',rotation=param_rotation)
   z_ = ax.set_ylabel(ylabel,color='b')
   plt.show()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def pca_all_plot(X, plot=True) :
   std_scale = preprocessing.StandardScaler().fit(X)
   X_scaled = std_scale.transform(X)
   
   if plot is False :
      return X_scaled
   
   list_explained_variance_ratio = list()
   n_components_max = X_scaled.shape[1]
   ind_components = 1
   iviz = 0
   while ind_components <= n_components_max :
       pca = PCA(n_components=ind_components)
       pca.fit(X_scaled)
       list_explained_variance_ratio.append(pca.explained_variance_ratio_.sum())
       iviz +=1
       if 0 == iviz%100:
         print("** Component = "+str(ind_components))
       ind_components += 1

   # Le dernier incrément est retiré; 
   # il correspond à ind_components = n_components_max+1
   ind_components -=1   

   # Affichage de la courbe de la variance expliquée
   z = plt.figure(figsize=(10,10))
   z=plt.plot(range(1,n_components_max+1),list_explained_variance_ratio, 'o-')
   return X_scaled
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_pca_all_plot(df, plot=True) :
   X = df.values
   std_scale = preprocessing.StandardScaler().fit(X)
   X_scaled = std_scale.transform(X)
   
   if plot is False :
      return X_scaled
   
   list_explained_variance_ratio = list()
   n_components_max = df.shape[1]
   if False :
       if n_components_max <= df.shape[0] :
          print("\n*** Decomposition : PCA")
          algorithm = PCA
          dict_param={'svd_solver':'auto'}
       else :
          print("\n*** Decomposition : TruncatedSVD")
          algorithm = decomposition.TruncatedSVD
          dict_param=dict()
   
   algorithm,  dict_param = get_decomp_algo(df)
   #n_components_max = min(df.shape[0], df.shape[1])
   ind_components = 0
   iviz = 0
   while ind_components < n_components_max :
       #print(ind_components)
       decomp = algorithm(n_components=ind_components, **dict_param)
       try :
           decomp.fit(X_scaled)
       except ValueError as valueError :
           print("\n***ERROR : valueError : {}".format(valueError))
           return None
       list_explained_variance_ratio.append(decomp.explained_variance_ratio_.sum())
       iviz +=1
       if 0 == iviz%100:
         print("** Processed components : {}/{} ".format(ind_components+1,n_components_max), end='\r')
       ind_components += 1

   # Le dernier incrément est retiré; 
   # il correspond à ind_components = n_components_max+1
   ind_components -=1   

   # Affichage de la courbe de la variance expliquée
   z = plt.figure(figsize=(10,10))
   z=plt.plot(range(1,n_components_max+1),list_explained_variance_ratio, 'o-')
   return X_scaled
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def get_decomp_algo(df):
    n_components_max = df.shape[1]
    if n_components_max <= df.shape[0] :
        print("\n*** Decomposition : PCA")
        decomp_algo = PCA
        dict_param={'svd_solver':'auto'}
    else :
        print("\n*** Decomposition : TruncatedSVD")
        decomp_algo = decomposition.TruncatedSVD
        dict_param=dict()
    return    decomp_algo,  dict_param
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def plot_pca_variance(pca) :
    '''Displays curve of explained variance regarding components.
    '''
    # Affichage de la courbe de la variance expliquée
    n_components_max = pca.get_params()['n_components']
    list_explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
    z = plt.figure(figsize=(10,10))
    z=plt.plot(range(1,n_components_max+1),list_explained_variance_ratio, 'o-')
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def get_component_from_cum_variance(pca, var_percent) :
    '''Returns the number of component that explains the percent of variance 
    given as function parameter.
    
    Input : 
        *   pca : PCA based decomposition algorithm. It may be IncrementalPCA,
        KernelPCA, PCA....
        Class must implement explained_variance_ratio_() method.
        *   var_percent : percentage of explained variance expected.
    Output :
        * Number of compoenents.
    '''
    #---------------------------------------------------------------------------
    # Get cumulative sum of explained variance ratio.
    #---------------------------------------------------------------------------
    try :
        list_explained_variance = pca.explained_variance_ratio_.cumsum()
    except AttributeError as attributeError:
        print("\n*** ERROR : get_component_from_cum_variance(): {}".format(attributeError))
        return None
    #---------------------------------------------------------------------------
    # Get the list of all cumulative values less then var_percent.
    #---------------------------------------------------------------------------
    list_component = [component for component in list_explained_variance if \
                      component <= var_percent]

    #---------------------------------------------------------------------------
    # The dimension is the number of elements in list.
    #---------------------------------------------------------------------------
    return len(list_component)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_kpca_all_plot(df, kernel_pca= 'rbf', plot=True, dict_kernel_param=None) :
    ''' Affiche, pour le PCA avec noyau les contributions cumulées à la variance 
    des composantes principales.'''
    #kernel : “linear” | “   poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
    #----------------------------------------------------------------------------
    # Dataset is scaled with standard deviation.
    # Doing so, variabme importance depends on variance, and does not depend on 
    # variable magnetude.
    #----------------------------------------------------------------------------
    X = df.values
    std_scale = preprocessing.StandardScaler().fit(X)
    X_scaled = std_scale.transform(X)

    if plot is False :
        return X_scaled
   
    #----------------------------------------------------------------------------
    # Extract kernel PCA parameters from dictionary given in function parameters.
    #----------------------------------------------------------------------------
    if dict_kernel_param is not None :
        kernel_name = dict_kernel_param['kernel_name']
        #-----------------------------------------------------------------------
        if 'gamma' in  dict_kernel_param.keys() :
            parameter_gamma = dict_kernel_param['gamma']
        else :
            parameter_gamma = 1./df.shape[1]
        #-----------------------------------------------------------------------
        if 'degree' in dict_kernel_param.keys() :
            degree = dict_kernel_param['degree']
        else :
            degree = 3
    else :
        kernel_name = kernel_pca
        parameter_gamma = 1./df.shape[1]
        degree = 3
    
    #----------------------------------------------------------------------------
    # For any dimension from dataset, kernel PCA is computed and cumulated.
    #----------------------------------------------------------------------------
    list_explained_variance_ratio = list()
    n_components_max = df.shape[1]
    ind_components = 1
    while ind_components <= n_components_max :
       kpca = decomposition.KernelPCA(n_components=ind_components\
       , kernel=kernel_name, gamma=parameter_gamma, degree=degree)

       kpca.fit(X_scaled)
       list_explained_variance_ratio.append(kpca.lambdas_.sum())
       if 0 == ind_components%100:
        print("** Processed components {}/{}= ".format(ind_components,n_components_max))

       ind_components += 1

    #----------------------------------------------------------------------------
    # Le dernier incrément est retiré; 
    # il correspond à ind_components = n_components_max+1
    #----------------------------------------------------------------------------
    ind_components -=1   

    #----------------------------------------------------------------------------
    # Affichage de la courbe de la variance expliquée
    #----------------------------------------------------------------------------
    z = plt.figure(figsize=(10,10))
    z=plt.plot(range(ind_components),list_explained_variance_ratio, 'o-')
    return X_scaled
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 
# ------------------------------------------------------------------------------
def X_pca_components_plot(X_pca, X_scaled, nb_components=2, ratio=1.,\
 param_title=None, x_max_y_max=None, plane=(0,1)) :

   
    if ratio < 1 :
        #-----------------------------------------------------------------------
        # Randomly get nb_index of points from array to be plot.
        #-----------------------------------------------------------------------
        size=len(X_pca)
        nb_index = int(size*ratio)
        index_array = np.random.randint(0,size,nb_index)
        
        #-----------------------------------------------------------------------
        # Select part of array sliding original array with random indexes
        #-----------------------------------------------------------------------
        X_pca = X_pca[index_array]
        X_scaled = X_scaled[index_array]
    else : 
        pass

    print("\nShape of points to be plot: ".format(X_pca.shape))
    
    # Cadrage de l'affichage selon X
    comp = plane[0]
    xmin, xmax = np.min(X_pca[:,comp]),np.max(X_pca[:,comp])
    if x_max_y_max is None :
        pass
    else :
        xmax = x_max_y_max[0]
    
    plt.xlim([xmin-1, xmax+1])

    plt.xlabel('Main component '+str(comp), size=14, color='b')
    if param_title is None :
       plt.title('Main components', size=14, color='b')
    else :
       plt.title(param_title, size=14, color='b')

    if 2 == nb_components :
        # Cadrage de l'affichage selon Y
        comp = plane[1]
        ymin, ymax = np.min(X_pca[:,comp]),np.max(X_pca[:,comp])
        if x_max_y_max is None :
            pass
        else :
            ymax = x_max_y_max[1]
        
        plt.ylim([ymin-1, ymax+1])

        plt.ylabel('Main component '+str(comp), size=14, color='b')

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='r', alpha=0.2)
    
    elif  1 == nb_components :
        # First component fo treansformed PCA is X axis,  
        plt.scatter(X_pca[:, 0], np.zeros(X_scaled[:, 0].shape),c='r')

    else :
      print("*** WARNING : max PCA plot components = 2")

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 
# ------------------------------------------------------------------------------
def df_pca_components_plot(df, color_feature, nb_components=2\
                            , param_title=None, ratio=1.) :
    '''Displays projection of X_scaled over 1<= nb_components <= 2 principals 
    components. 
    Plot are colored depending on color_feature parameter magnetude.
    Return : pca

    '''

    if 0 >= nb_components :
      return None

    z = plt.figure(figsize=(10,10))

    # Get standardized data
    X_scaled = df_get_std_scaled_values(df)

    #Build PCA algorithme.
    pca = PCA(n_components=nb_components)
    pca.fit(X_scaled)

    X_projected = pca.transform(X_scaled)
    print(X_projected.shape)
   
    if ratio < 1 :
        #-----------------------------------------------------------------------
        # Randomly get nb_index
        #-----------------------------------------------------------------------
        size=len(X_projected)
        nb_index = int(size*ratio)
        index_array = np.random.randint(0,size,nb_index)
        
        #-----------------------------------------------------------------------
        # Select part of array sliding original array with random indexes
        #-----------------------------------------------------------------------
        X_projected = X_projected[index_array]
        print(X_projected.shape)
    else : 
        pass

    # Cadrage de l'affichage selon X
    xmin, xmax = np.min(X_projected[:,0]),np.max(X_projected[:,0])
    plt.xlim([xmin-1, xmax+1])

    plt.xlabel('1er composante principale', size=14, color='b')
    if param_title is None :
       plt.title('Composantes principales', size=14, color='b')
    else :
       plt.title(param_title, size=14, color='b')

    if 2 == nb_components :
      # Cadrage de l'affichage selon Y
      ymin, ymax = np.min(X_projected[:,1]),np.max(X_projected[:,1])
      plt.ylim([ymin-1, ymax+1])

      plt.ylabel('2eme composante principale', size=14, color='b')

      if color_feature in df.columns :
         spectral_value = df.describe()[color_feature].loc['max']
         # Colorier en utilisant les valeurs de la variable color_feature
         plt.scatter(X_projected[:, 0], X_projected[:, 1], c=df[color_feature] \
         , edgecolor='none', alpha=0.2,\
         cmap=plt.cm.get_cmap('nipy_spectral', spectral_value))   

      else :
         plt.scatter(X_projected[:, 0], X_projected[:, 1], c='r', alpha=0.2)
    elif  1 == nb_components :
      if color_feature in df.columns :
         # Colorier en utilisant les valeurs de la variable color_feature
         plt.scatter(X_projected[:, 0], np.zeros(X_scaled[:, 0].shape)\
         , c=df[color_feature] , edgecolor='none',
         cmap=plt.cm.get_cmap('nipy_spectral', spectral_value))   

      else :
         plt.scatter(X_projected[:, 0], np.zeros(X_scaled[:, 0].shape),c='r')

    else :
      print("*** WARNING : max PCA plot components = 2")
      return pca


      
    # Affichage de la barre de couleur
    if color_feature in df.columns :
      z = plt.colorbar()
    #z = plt.figure(figsize=(10,10))
    return pca
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_pcs2_plot(df, pca) :
   '''
   Plot all components in the repair of the 2 principals components.
   '''
   z = plt.figure(figsize=(20,20))

   pcs = pca.components_

   for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
       # Afficher un segment de l'origine au point (x, y)
       plt.plot([0, x], [0, y], color='r')
       # Afficher le nom (data.columns[i]) de la performance
       plt.text(x, y, df.columns[i], fontsize='16', color='b')
       #print(df_moovies_scoring.columns[i],x,y)

   # Afficher une ligne horizontale y=0
   plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')

   # Afficher une ligne verticale x=0
   plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')

   plt.xlim([-0.7, 0.7])
   plt.ylim([-0.7, 0.7])
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_data_pca_1_plot(df, model, model_1D, X, parameter_title=None\
,original=False) :
    
    '''Plot data given as X along with 1st principal component from model_1D 
    algorithm.
    model_1D algorithme is initialized by caller.
    Input : 
        model : algorithm for PCA; initialization has been performed by caller.
        model_1D : algorithm for PCA for 1 dimension; initialization has been 
        performed by caller.
        X        : data to be plotted. 
        original : when True, then X_scaled is plotted from original space.
                    when False, then X_scaled is plotted from princpals 
                    components space.
    '''

    #pca_1 = PCA(n_components=1)
    model_1D.fit(X)
    X_projected_1 = model_1D.transform(X)

    print("Original : {}".format(X.shape))
    print("Projection 1D: {}".format(X_projected_1.shape))


    X_inv_1 = model_1D.inverse_transform(X_projected_1)

    z = plt.figure(figsize=(10,10))
    if original is True :
        # Plot original data
        plt.xlabel('X',size=14, color='b')
        plt.ylabel('Y',size=14, color='b')
        plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
        # Plot 1st dimension into the original data space
        plt.scatter(X_inv_1[:, 0], X_inv_1[:, 1], alpha=0.8, color='r')
    else :
        # Plot projected data
        model.fit(X)
        X_projected = model.transform(X)
        print("Projection : {}".format(X_projected.shape))
        plt.xlabel('Composante 1',size=14, color='b')
        plt.ylabel('Composante 2',size=14, color='b')
        plt.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.2, color='black')
        
    # Plot title
    if parameter_title is None :
        plt.title('Données en 2D',size=14, color='b')
    else :
        plt.title(parameter_title,size=14, color='b')


    plt.axis('equal');
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def df_kpca_components_plot(df, X_scaled, color_feature=None, gamma_parameter=15\
, nb_components=2, original_data=False, dict_kernel_param=None) :

    z = plt.figure(figsize=(10,10))
    
    if dict_kernel_param is None :
        kernel_name = 'rbf'
        parameter_gamma = 1./df.shape[1]
    else :
        kernel_name = dict_kernel_param['kernel_name']
        if 'gamma' in  dict_kernel_param.keys() :
            parameter_gamma = dict_kernel_param['gamma']
        else :
            parameter_gamma = 1./df.shape[1]
        if 'degree' in dict_kernel_param.keys() :
            degree = dict_kernel_param['degree']
        else :
            degree = 3

    kpca = decomposition.KernelPCA(n_components=nb_components,\
    kernel=kernel_name, gamma=parameter_gamma, degree=degree)

        

    #----------------------------------------------------------------------------
    # Perform KPCA transformation
    #----------------------------------------------------------------------------
    X_projected = kpca.fit_transform(X_scaled)
    print(X_projected.shape)

    #----------------------------------------------------------------------------
    # Projection selon la 1er composante principale.
    #----------------------------------------------------------------------------
    kpca_1 = KernelPCA(n_components=1, kernel=kernel_name, gamma=gamma_parameter\
    , fit_inverse_transform=True, degree=degree)
    
    X_projected_1 = kpca_1.fit_transform(X_scaled)

    #----------------------------------------------------------------------------
    # Retour à l'espace originel
    #----------------------------------------------------------------------------
    X_inv_1 = kpca_1.inverse_transform(X_projected_1)

    #----------------------------------------------------------------------------
    # Cadrage de l'affichage selon X et Y
    #----------------------------------------------------------------------------
    xmin, xmax = np.min(X_projected[:,0]),np.max(X_projected[:,0])
    plt.xlim([xmin-1, xmax+1])

    ymin, ymax = np.min(X_projected[:,1]),np.max(X_projected[:,1])
    plt.ylim([ymin-1, ymax+1])

    plt.title('KPCA('+kernel_name+'): 2 composantes principales', color='b', size=14)

    print("X_proj dans l'espace orginel : {}".format(X_inv_1.shape))
    plt.scatter(X_inv_1[:, 0], X_inv_1[:, 1], c='r', s=20)

    if color_feature in df.columns :
         #----------------------------------------------------------------------------
         # colorer en utilisant la variable color_feature
         #----------------------------------------------------------------------------

      if original_data is False :
          plt.xlabel('Composante Pcple 1', color='b', size=14)
          plt.ylabel('Composante Pcple 2', color='b', size=14)
          plt.scatter(X_projected[:, 0], X_projected[:, 1],c=df.get(color_feature))
      else :
          plt.xlabel('X', color='b', size=14)
          plt.ylabel('Y', color='b', size=14)
          plt.scatter(X_scaled[:, 0], X_scaled[:, 1],c=df.get(color_feature))
        
    else :
      if original_data is False :
          plt.xlabel('Composante Pcple 1', color='b', size=14)
          plt.ylabel('Composante Pcple 2', color='b', size=14)
          plt.scatter(X_projected[:, 0], X_projected[:, 1])
      else :
          plt.xlabel('X', color='b', size=14)
          plt.ylabel('Y', color='b', size=14)
          plt.scatter(X_scaled[:, 0], X_scaled[:, 1])
        
    # Affichage de la barre de couleur
    if color_feature in df.columns :
      z = plt.colorbar()

    return kpca
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def clustering_plot(X_clustered, cluster_labels, nclusters, title=None\
, X_center=None):
    z= plt.figure(figsize=(10, 10))

    if len(cluster_labels) > 0 :
        if X_center is not None :
         alpha = 0.3
        else: 
         alpha = 0.6
        
        print("clustering_plot()"+str(X_clustered.shape))
        print("clustering_plot()"+str(X_clustered[0, 0]))
        
        for i in range(X_clustered.shape[0]):
            plt.scatter(X_clustered[i, 0], X_clustered[i, 1],\
            c=cm.Oranges(cluster_labels[i] / nclusters), alpha=alpha)
           
        if X_center is not None :
           for i in range(X_center.shape[0]):
               plt.scatter(X_center[i, 0], X_center[i, 1],marker="o",s=400,c='red')
               plt.annotate(str(i), (X_center[i, 0], X_center[i, 1]))
    else : 
        for i in range(X_clustered.shape[0]):
            plt.scatter(X_clustered[i, 0], X_clustered[i, 1],c='b')
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17, color='b')
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def plot_2D(X, title, parameter_figsize=(10,10), colordir=0, frame=None) :
    
    z = plt.figure(figsize=parameter_figsize)
    plt.title(title, size=17, color='b')
    # Cadrage de l'affichage selon X,Y
    if frame is None :
       xmin, xmax = np.min(X[:,0]),np.max(X[:,0])
       ymin, ymax = np.min(X[:,1]),np.max(X[:,1])
    else :
      xmin, xmax = frame[0],frame[1]
      ymin, ymax = frame[2],frame[3]
      
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    print(xmin, xmax)
    print(ymin, ymax)
    if colordir is not None :
        colorize = dict(c=X[:, colordir], cmap=plt.cm.get_cmap('rainbow', 10))
        plt.scatter(X[:, 0], X[:, 1], **colorize, alpha=0.6)
    else : 
        plt.scatter(X[:, 0], X[:, 1], c='grey', alpha=0.6)
        
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def plot_3D(X, title, parameter_figsize=(10,10), colordir=1):

   nb_components = X.shape[1]

   _z = plt.figure(figsize=parameter_figsize)
   
   if 3 == nb_components :
      # Cadrage de l'affichage selon X,Y,Z
      xmin, xmax = np.min(X[:,0]),np.max(X[:,0])
      ymin, ymax = np.min(X[:,1]),np.max(X[:,1])
      zmin, zmax = np.min(X[:,2]),np.max(X[:,2])
      ax = plt.axes(projection='3d')
      # Displays title upper left
      ax.text2D(0.05, 0.95, title, transform=ax.transAxes)

      ax.set_xlim(xmin, xmax)
      ax.set_ylim(ymin, ymax)
      ax.set_zlim(zmin, zmax)
      
      _z= ax.set_title(title, size=17, color='b')
      _z= ax.set_xlabel('X', size=17, color='b')
      _z= ax.set_ylabel('Y', size=17, color='b')
      _z= ax.set_zlabel('Z', size=17, color='b')
      if colordir is not None :
         colorize = dict(c=X[:, colordir], cmap=plt.cm.get_cmap('rainbow', 5))
         _z = ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], **colorize,  alpha=0.6);
      else : 
         _z = ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c='grey', alpha=0.6);
   elif 2 == nb_components :
      plot_2D(X, title, parameter_figsize=parameter_figsize, colordir=colordir)
   else :
      print("ERROR : can't plot for components= {}".format(nb_components))
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def clustering_dbscan_plot_and_metrics(X_projected_train\
    ,X_projected_test,parameter_eps=5, parameter_min_samples=10) :
    #----------------------------------------------------
    # Building cluster 
    #----------------------------------------------------
    print(X_projected_train.shape)
    
    #dbscan = DBSCAN(eps=6, min_samples=10).fit(X_projected_train)
    dbscan = DBSCAN(eps=parameter_eps\
    , min_samples=parameter_min_samples).fit(X_projected_train)
    
    labels_trained = dbscan.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels_trained)) - (1 if -1 in labels_trained else 0)
    print("DBSCAN : estimation du nombre de clusters : {}".format(n_clusters_))

    if n_clusters_ <= 1 :
        printmd_error("DBSCAN : cluster bruité!")
        return None, None
    else : 
        #----------------------------------------------------
        # Plotting cluster 
        #----------------------------------------------------
        nclusters = n_clusters_
        title = "DBSCAN: clusters= {}".format(nclusters)
        clustering_plot(X_projected_train, labels_trained, nclusters, title)

        #----------------------------------------------------
        # Compute metrics for this cluster
        #----------------------------------------------------
        clustering_dbscan = dbscan.fit(X_projected_test)
        labels_predicted = clustering_dbscan.labels_
        clustering_compute_metrics(X_projected_train, labels_trained\
        , labels_predicted)
        return labels_trained, labels_predicted
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def clustering_kmeans_plot_and_metrics(X_train, X_test ,parameter_clusters=20\
, md=True) :

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=parameter_clusters)
    kmeans.fit(X_train)

    # Prediction de la position des clusters
    labels_trained = kmeans.predict(X_train)

    print(labels_trained.shape)

    #----------------------------------------------------
    # Plotting cluster 
    #----------------------------------------------------
    title = "K-means: clusters= {}".format(parameter_clusters)
    clustering_plot(X_train, labels_trained, parameter_clusters, title)

    #----------------------------------------------------
    # Compute metrics for this cluster
    #----------------------------------------------------
    X_clustered_test_label = kmeans.fit_predict(X_test)
    labels_predicted = kmeans.labels_
    clustering_compute_metrics(X_train, labels_trained, labels_predicted,md=md)
    return labels_trained, labels_predicted
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
def clustering_gmm_plot_and_metrics(X_train, X_test, parameter_clusters=20\
    ,gmm_clustering=None) :
    #----------------------------------------------------
    # Building cluster 
    #----------------------------------------------------
    if gmm_clustering is None : 
        gmm_clustering = GaussianMixture(n_components=parameter_clusters).fit(X_train)
    else : 
        pass
    labels_trained = gmm_clustering.predict(X_train)

    #----------------------------------------------------
    # Plotting cluster 
    #----------------------------------------------------
    title = "GMM: clusters= {}".format(parameter_clusters)
    clustering_plot(X_train, labels_trained, parameter_clusters, title)

    #----------------------------------------------------
    # Compute metrics for this cluster
    #----------------------------------------------------
    labels_predicted = gmm_clustering.fit_predict(X_test)
    clustering_compute_metrics(X_train, labels_trained, labels_predicted)    
    return labels_trained, labels_predicted
# ------------------------------------------------------------------------------


