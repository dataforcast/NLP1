#!/usr/bin/python3.6
#-*- coding: utf-8 -*-



'''Functions used to compute metrics in order to evaluate a model biais over 
a sample with sub-groups used to compute label.
'''
import pandas as pd
from sklearn import metrics
import numpy as np

import p9_util

TOXICITY_COLUMN = p9_util.COLUMN_NAME_TARGET
SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive
# List all identities
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def compute_auc(y_true, y_pred, is_debug=False):
    '''Computes AUC given true labels and predicted labels.
    '''
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError as valueError:
        if is_debug :
            print("\n*** ERROR on AUC computation : {}".format(valueError))
        return np.nan
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def compute_identity_auc(df, identity, label, predict):
    '''This metric allows to evaluate how the model performs to 
    distinguish toxic and non toxic comments that mention identity.
    
    A low value in this metric means that the model, for a given identity, 
    confuses non-toxic comments with toxic comments.
    
    
    When the AUC is low this means model is biased in the sens it predicts 
    lower toxicity scores than it should for safe texts and lower toxicity scores 
    then it should for toxic texts.

    
    Input : 
        df : contains all dataset with true labels and predictions; 
        it is used along with identity column in order to define a subset 
        of dataset.
        
        identity : the identity against with model performance is evaluated.
        
        label : True value of toxicity
        
        predict : toxicity prediction
        
    '''
    
    #---------------------------------------------------------------------------
    # The subset isuued from dataset that contain identity is defined.
    #---------------------------------------------------------------------------
    subset_identity = df[df[identity]]

    #---------------------------------------------------------------------------
    # Measurement of confusion for predictions where this sub-group is present.
    #
    # subset_identity[label] contains true values of toxicity of the subset 
    # defined with identity.
    #
    # While subset_identity[predict] contains toxicity predictions for the subset 
    # of data where identity is mentioned.
    #---------------------------------------------------------------------------    
    return compute_auc(subset_identity[label], subset_identity[predict])
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def compute_bpsn_auc(df, identity, true_label, prediction_label):
    '''Background positive (toxic) subgroup negative (safe of toxicity) :
    Background is the dataset free of the given identity.
    Subgroup refers to the dataset subset in which given identity is mentioned.
    
    Here, we restrict the test set to : 
        -> the non-toxic comments that mention the identity 
        and 
        -> the toxic comments that do not mention the identity. 
    
    A low value in this metric means that the model confuses non-toxic comments 
    that mention the identity with toxic comments that do not.
    
    A low  value means that a large part of safe texts mentionning identity has not been detected.
    
    This means model is biased in the sens it predicts higher toxicity scores than it should 
    for safe texts mentioning identity.
        
    Input :
        * df : the dataset from which subsets will be extracted to evaluate performance model. 
        * identity : column name from the given dataset that references an identity.
        * true_label : true toxicity values from dataset
        * prediction_label : toxicity predictions applied on the given dataset and computed wih the 
                       predictive model.
    Output :
        * Area under ROC curve value for 
    '''
    """
    Computes the AUC of the within-subgroup negative examples and the background positive examples.
    """
    #---------------------------------------------------------------------------
    # Non toxic texts that mention identity
    #---------------------------------------------------------------------------
    df_identity_freetoxic = df[df[identity] & ~df[true_label]]

    #---------------------------------------------------------------------------
    # Toxic textes that do not mention identity
    # ~df[identity] : select rows from dataframe where identity is not there
    # df[true_label] : selects rows from dataframe where text is toxic
    #---------------------------------------------------------------------------
    df_background_toxic = df[~df[identity] & df[true_label]]
    
    #---------------------------------------------------------------------------
    # Create a new dataset with 
    #  * toxic background : texts that are toxic and that do not mention identity 
    #  * identity safe : texts that are asfe and that mention identity.
    #---------------------------------------------------------------------------
    df_bpsn = df_identity_freetoxic.append(df_background_toxic)
    
    #---------------------------------------------------------------------------
    # Measure model performance for sub-dataset df_bpsn
    #---------------------------------------------------------------------------
    return compute_auc(df_bpsn[true_label], df_bpsn[prediction_label])
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def compute_bnsp_auc(df, identity, true_label, prediction_label):
    '''Background negative (safe of toxicity) subgroup positive (toxic).
    
    Background is the dataset free of the given identity.
    Subgroup refers to the dataset subset in which given identity is mentioned.
    
    Here, we restrict the test set to : 
        -> the non-toxic comments that do not mention the identity given as parameter
        and 
        -> the toxic comments that do mention the identity given as parameter
    
    A low value in this metric means that the model confuses non-toxic comments 
    that do not mention the identity with toxic comments that do mention identity.
    
    A low  value means that a large part of toxics texts mentionning identity has not been detected.
    
    This means model is biased in the sens it predicts lower toxicity scores than it should 
    for toxic texts mentioning identity.
        
    Input :
        * df : the dataset from which subsets will be extracted to evaluate performance model. 
        * identity : column name from the given dataset that references an identity.
        * true_label : true toxicity values from dataset
        * prediction_label : toxicity predictions applied on the given dataset and computed wih the 
                       predictive model.
    Output :
        * Area under ROC curve value for 
    '''
        
    #--------------------------------------------------------------------------
    # Select the subset of dataset in which :
    # identity is mentioned and text is toxic.
    #--------------------------------------------------------------------------
    df_identity_toxic = df[df[identity] & df[true_label]]
    
    #--------------------------------------------------------------------------
    # Select the subset of dataset in which :
    # identity is not mentioned and text is safe.
    # This combination of criterias leads to safe background subset.
    #--------------------------------------------------------------------------
    df_background_safe = df[~df[identity] & ~df[true_label]]
    
    #--------------------------------------------------------------------------
    # Aggregate subset with toxic texts and saf texts.
    #--------------------------------------------------------------------------
    df_bnsp = df_identity_toxic.append(df_background_safe)

    #--------------------------------------------------------------------------
    # Evaluate model against toxicity predictions over a subset of dataset 
    # with toxic texts mentioning identity and safe background.
    #--------------------------------------------------------------------------
    return compute_auc(df_bnsp[true_label], df_bnsp[prediction_label])
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def compute_bias_metrics_for_model(dataset,
                                   list_subgroup,
                                   predict_column_name,
                                   true_column_name,
                                   include_asegs=False):
    """For each one of the sub-groups, AUC metrics are computed.
    Input : 
        * dataset : input data.
        * list_subgroup : list of sub-groups (identities) 
        * predict_column_name : column name containing model predictions.
        * true_column_name : column name from datasset containing True values 
        * include_asegs : NOT USED
    Output :
        * A dataframe containing metrics into AUC columns, sorted on 'subgroup_auc' column
    """
    records = []
    for subgroup in list_subgroup:
        record = {
            'subgroup': subgroup,
            'subgroup_size':len(dataset[dataset[subgroup]])
        }
        record[SUBGROUP_AUC] = compute_identity_auc(dataset, subgroup, true_column_name, predict_column_name)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, true_column_name, predict_column_name)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, true_column_name, predict_column_name)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def calculate_overall_auc(df, model_name, toxicity_column):
    true_labels = df[toxicity_column]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC].dropna(), POWER),
        power_mean(bias_df[BPSN_AUC].dropna(), POWER),
        power_mean(bias_df[BNSP_AUC].dropna(), POWER)
    ])
    bias_score_contribution = (1 - OVERALL_MODEL_WEIGHT) * bias_score
    print("Bias score contribution : {}".format(bias_score_contribution))
    return (OVERALL_MODEL_WEIGHT * overall_auc) + (bias_score_contribution)
#-------------------------------------------------------------------------------


