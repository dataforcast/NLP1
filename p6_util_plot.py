import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image

import p6_util
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def dict_plot_frequency(dict_freq, doc_type, query=None, p_figsize=(10,10)):
    """Plot word frequencies onto a bar diagram from words into a dictionary.
    
    Input : 
        * dict_freq : nltk.probability.FreqDist object type; a kind of Python
        dictionary containing words as keys and words frequencies as values.
        
        * doc_type : string used for title.
        
        * query : used to restrict (keys, values) to be displayed.
        dict_freq is converted as a DataFrame object. Then query method is
        applied on such object.
    """
    z_=plt.figure(figsize=p_figsize)
    if dict_freq is not None:
        df= p6_util.get_df_from_FreqDist(dict_freq, query)
        title = "Words frequency for : "+doc_type
        title += " / "+query
        z_=df.plot(kind='bar', title=title, color='orange')
    else:
        print("\n*** ERROR : given FreqDist object is None!")
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def display_word_cloud(dict_word_freq, file_name=None):
    """Display a cloud of words from fiven dictionary formated as follow : 
    {word:word_count}.
    """
    wordcloud_generator = WordCloud(background_color='white', max_words=100)
    wordcloud_generator.generate_from_frequencies(dict_word_freq)
    
    if file_name is not None:
        wordcloud_generator.to_file(file_name)
    else:
        pass

    plt.figure(figsize=[20,10])
    plt.imshow(wordcloud_generator, interpolation='bilinear')

    plt.axis("off")
    plt.show()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def X_y_plot(X,y,item_count, title, p_x_title, p_y_title) :
    fig, ax = plt.subplots(figsize=(20,10))

    if item_count is not None:
        X_plot = X[:item_count]
        y_plot = y[:item_count]
    else:
        X_plot = X.copy()
        y_plot = y.copy()
    

    ax.plot(X_plot,y_plot)
    
    if item_count is not None:
        #ax.set_xticklabels(X[:item_count], rotation=90)
        pass
    else:
        #ax.set_xticklabels(X, rotation=90)
        pass

    if p_x_title is None :
        ax.set_xlabel('Accuracy')
    else :
        ax.set_xlabel(p_x_title)

    if p_y_title is None :
        ax.set_ylabel('Classifiers')
    else : 
        ax.set_ylabel(p_y_title)
        
    if title is not None : 
        ax.set_title(title)
        
    ax.grid(linestyle='-', linewidth='0.1', color='grey')
    fig.patch.set_facecolor('#E0E0E0')

    plt.show()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def ser_item_occurency_plot(ser_item_name, ser_item_count
, item_count=None, title=None, p_reverse=True, p_x_title=None
, p_y_title=None, shift=0, legend=None):
    """Plot values issued from 2 input Series as following : 
    Input :
        * ser_item_name :  Series containing names of items to be plot;
        these name will be ticked on X axis.
        * ser_item_count : second Series contains occurrencies for each item.
        They contribute to Y axis.
    
    """
    df_item_dict={item:count for item, count \
    in zip(ser_item_name, ser_item_count)}
    
    if p_reverse is not None :
        list_item_sorted \
        = sorted(df_item_dict.items(), key=lambda x: x[1], reverse=p_reverse)

        dict_item_sorted = dict()
        for tuple_value in list_item_sorted :
            dict_item_sorted[tuple_value[0]] = tuple_value[1]
    else :
        dict_item_sorted = df_item_dict.copy()
    

    X = list(dict_item_sorted.keys())
    y = list(dict_item_sorted.values())
    
    #X_y_plot(X,y,item_count)
    fig, ax = plt.subplots(figsize=(20,10))

    if item_count is not None:
        X_plot = X[:item_count]
        y_plot = y[:item_count]
    else:
        X_plot = X.copy()
        y_plot = y.copy()
    
    ax.plot(X_plot[shift:],y_plot[shift:])
    ax.set_xticklabels(X[:item_count], rotation=90)

    if p_x_title is None :
        ax.set_xlabel('Accuracy')
    else :
        ax.set_xlabel(p_x_title)

    if p_y_title is None :
        ax.set_ylabel('Classifiers')
    else : 
        ax.set_ylabel(p_y_title)
        
    if title is not None : 
        ax.set_title(title)
    ax.grid(linestyle='-', linewidth='0.1', color='grey')
    if legend is not None :
        ax.legend(legend, loc=0)
    fig.patch.set_facecolor('#E0E0E0')

    plt.show()
#-------------------------------------------------------------------------------

