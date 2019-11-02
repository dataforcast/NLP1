#!/usr/bin/python3.6
#-*- coding: utf-8 -*-
import chart_studio as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF

import numpy as np
import pandas as pd
import scipy

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

import researchpy as rp

from statsmodels.formula.api import ols

import p9_util

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------    
def df_anova_group_level(df,column_name, is_verbose=True) :
    print("\n=================================")
    print("======= "+column_name+" =======")
    print("===================================")
    label_value = 0
    left_limit = 0.0
    right_limit= 0.0 # not taken into acount with left opertor value fixed to (0,0)

    df = p9_util.df_column_cont2labelRange(df, column_name, left_limit, right_limit, label_value, \
                                      t_left_operator=(0,0), t_right_operator=(-1,0)) 
    

    list_left_limit =  [limit/10 for limit in range(0,10)]
    list_right_limit = [limit/10 for limit in range(1,11)]
    list_label_value = [label_value/10 for label_value in range(1,11)]
    for left_limit, label_value, right_limit in zip(list_left_limit, list_label_value, list_right_limit) :
        df = p9_util.df_column_cont2labelRange(df, column_name, left_limit, right_limit, label_value, \
                                      t_left_operator=(1,1), t_right_operator=(-1,0), is_verbose=is_verbose)
        
    df[column_name] = df[column_name]*10

    dict_replace = {float(label)*10:'level_'+str(int(label*10)) for label in [0,]+list_label_value}

    df[column_name].replace( dict_replace, inplace= True)
    if is_verbose :
        print("")
        print(dict_replace)        
    
    ser_groupby = df['target'].groupby(df[column_name])
    min_group_count = rp.summary_cont(ser_groupby).dropna()['N'].min()
    min_group_count
    
    list_label_value = [int(10*label_value) for label_value in [0]+list_label_value]
    if is_verbose :
        print("")
        print(list_label_value)
    
    df = p9_util.build_balanced_groups(df, column_name, list_label_value, \
    min_group_count, is_verbose=is_verbose)
    
    ser_groupby = df['target'].groupby(df[column_name])
    if is_verbose :
        print("")
        print(rp.summary_cont(ser_groupby).dropna().sort_values(by=['Mean']))
    
    
    print("\nLevene test for {}".format(column_name))
    print(scipy.stats.levene(df['target'][df[column_name] == 'level_0'],
             df['target'][df[column_name] == 'level_1'],
             df['target'][df[column_name] == 'level_2'],
             df['target'][df[column_name] == 'level_3'],                   
             df['target'][df[column_name] == 'level_4'],                   
             df['target'][df[column_name] == 'level_5'],                   
             df['target'][df[column_name] == 'level_6'],                   
             df['target'][df[column_name] == 'level_7'],                   
             df['target'][df[column_name] == 'level_8'],                   
             df['target'][df[column_name] == 'level_9'],                   
             df['target'][df[column_name] == 'level_10'],                   
                  ))

    model_name = ols('target ~ C('+column_name+')', data=df[['target',column_name]]).fit()
    print("\nShapiro normality test for {}".format(column_name))
    print(scipy.stats.shapiro(model_name.resid))

    print("")
    print(model_name.summary())
    
    # Post-hoc test
    print("")
    mc = MultiComparison(df['target'], df[column_name])
    mc_results = mc.tukeyhsd()
    print(mc_results)
