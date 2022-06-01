import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
import seaborn as sns
import cufflinks as cf
import plotly as py
import torch
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
py.offline.init_notebook_mode(connected = True)
cf.go_offline()
sns.set()
import sklearn
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from toolbox_02450 import rlr_validate
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary,correlated_ttest, rlr_validate


# ========================================================================================================================================
# Importing the data and feature transformation
# ========================================================================================================================================
random_seed = 2
new_df = pd.read_csv("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data")
new_df = new_df.drop(columns=['row.names'],axis=1)
dic = {'Present':1, 'Absent':0}
new_df['famhist'] = new_df['famhist'].replace(dic)

attribute_names = ['sbp','tobacco','ldl','adiposity','famhist','typea','obesity','alcohol','age']
class_name      = ['chd']

X = new_df[attribute_names].to_numpy(dtype=np.float32)
y = new_df[class_name].to_numpy(dtype=np.float32)

N = np.shape(X)[0] 
M = np.shape(X)[1] 
# ========================================================================================================================================
# Classification
# ========================================================================================================================================  
## Standardizing
X_tilde = X - np.ones((N,1))*X.mean(axis=0)
X_tilde = X_tilde*(1/np.std(X_tilde,0))
X = X_tilde

print_cv_inner_loop_text = False
print_cv_outer_loop_text = True

apply_KNN      = True 
apply_logistic = True 

apply_setup_ii = True

# Regularization options KNN
min_k_KNN = 1 
max_k_KNN = 30 

# Regularization options Logistic Regresssion
lambda_interval = range(10,35)

## KNN options (only needed if apply_KNN is set to True)
dist          = 2 # Distance metric (corresponds to 2nd norm, euclidean distance). You can set dist=1 to obtain manhattan distance (cityblock distance).
metric        = 'minkowski'
metric_params = {} # no parameters needed for minkowski

# Set K-fold 
K_1   = 10 # Number of outer loops
K_2   = 10 # Number of inner loops
CV_1  = sklearn.model_selection.KFold(n_splits=K_1,shuffle=True, random_state = random_seed)
CV_2  = sklearn.model_selection.KFold(n_splits=K_2,shuffle=True, random_state = random_seed)
CV_setup_ii = sklearn.model_selection.KFold(n_splits=K_1,shuffle=True, random_state = random_seed + 1)

# Statistical test 
loss_in_r_function = 2  
r_baseline_vs_logistic  = []                  
r_baseline_vs_sec_model = []                  
r_sec_model_vs_logistic = []                  
alpha_t_test            = 0.05
rho_t_test              = 1/K_1

# Define list for outer CV 
test_error_outer_baseline                = [] 
test_error_outer_KNN                     = []
test_errors_outer_logistics              = []
data_outer_test_length                   = []
optimal_regularization_param_KNN         = []
optimal_regularization_param_logistic = []

# Outer loop
k_outer = 0
for train_outer_index, test_outer_index in CV_1.split(X):
    if(print_cv_outer_loop_text):
        print('Computing CV outer fold: {0}/{1}..'.format(k_outer+1,K_1))
    X_train_outer, y_train_outer = X[train_outer_index,:], y[train_outer_index]
    X_test_outer, y_test_outer = X[test_outer_index,:], y[test_outer_index]
    
    data_outer_train_length    = float(len(y_train_outer))
    data_outer_test_length_tmp = float(len(y_test_outer))
    
    best_inner_model_baseline = []
    error_inner_baseline      = [] 
    data_validation_length    = [] 
    
    validation_errors_inner_KNN_matrix         = np.array(np.ones(max_k_KNN-min_k_KNN + 1))  
    validation_errors_inner_logistics_matrix   = np.array(np.ones(len(lambda_interval))) 
       
    # Inner loop
    k_inner=0
    for train_inner_index, test_inner_index in CV_2.split(X_train_outer):
        if(print_cv_inner_loop_text):
            print('Computing CV inner fold: {0}/{1}..'.format(k_inner+1,K_2))
            
        X_train_inner, y_train_inner = X[train_inner_index,:], y[train_inner_index]
        X_test_inner, y_test_inner = X[test_inner_index,:], y[test_inner_index]        
        
        best_inner_model_baseline_tmp = stats.mode(y_train_inner).mode[0][0]
        y_est_test_inner_baseline     = np.ones((y_test_inner.shape[0],1))*best_inner_model_baseline_tmp
        
        validation_errors_inner_baseline = np.sum(y_est_test_inner_baseline != y_test_inner) / float(len(y_test_inner))
                
        best_inner_model_baseline.append(best_inner_model_baseline_tmp)
        
        error_inner_baseline.append(validation_errors_inner_baseline)
        
        data_validation_length.append(float(len(y_test_inner)))
        
        #KNN
        if (apply_KNN):
            validation_errors_inner_KNN   = []
            for k_nearest_neighbour_tmp in range(min_k_KNN,max_k_KNN + 1):
                # Fit classifier and classify the test points
                model = KNeighborsClassifier(n_neighbors=k_nearest_neighbour_tmp, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
                
                model = model.fit(X_train_inner.squeeze(), y_train_inner.squeeze()) ## knclassifier.fit requires .squeeze of input matrices
                
                y_est_inner_model_KNN_tmp       = model.predict(X_test_inner)
                validation_errors_inner_KNN_tmp = np.sum(y_est_inner_model_KNN_tmp != y_test_inner.squeeze()) / float(len(y_test_inner))
                validation_errors_inner_KNN.append(validation_errors_inner_KNN_tmp)
            
            validation_errors_inner_KNN = np.array(validation_errors_inner_KNN)
            validation_errors_inner_KNN_matrix = np.vstack((validation_errors_inner_KNN_matrix,validation_errors_inner_KNN))

        # Estimate logistic regression if apply_logistic is true
        if (apply_logistic):
            validation_errors_inner_logistics  = []
            for s in range(0, len(lambda_interval)):
                model       = LogisticRegression(penalty='l2', C=1/lambda_interval[s], solver = 'liblinear')
                model       = model.fit(X_train_inner, y_train_inner.squeeze())            
                y_est_inner = model.predict(X_test_inner)
                w_est = model.coef_[0]   #Check weights from here
                validation_errors_inner_logistics_tmp = np.sum(y_est_inner != y_test_inner.squeeze()) / float(len(y_test_inner))
                validation_errors_inner_logistics.append(validation_errors_inner_logistics_tmp)
            
            validation_errors_inner_logistics = np.array(validation_errors_inner_logistics)
            validation_errors_inner_logistics_matrix = np.vstack((validation_errors_inner_logistics_matrix,validation_errors_inner_logistics)) 
            
        # add 1 to inner counter
        k_inner+=1
        
    # Estimate generalization error of each model    
    generalized_error_inner_baseline_model = np.sum(np.multiply(data_validation_length,error_inner_baseline)) * (1/data_outer_train_length)
          
    # 'Fit' baseline model on outside data (chooses the class with most obs - aka. the mode in statistics)
    best_outer_model_baseline_tmp = stats.mode(y_train_outer).mode[0][0]
    y_est_test_outer_baseline     = np.ones((y_test_outer.shape[0],1))*best_outer_model_baseline_tmp
      
    # Estimate the test error (best model from inner fitted on the outer data)
    test_error_outer_baseline_tmp = np.sum(y_est_test_outer_baseline != y_test_outer) / float(len(y_test_outer))
    test_error_outer_baseline.append(test_error_outer_baseline_tmp)
    
    # Add length of outer test data
    data_outer_test_length.append(data_outer_test_length_tmp)
    
    # Find optimal model of KNN (if apply_KNN is true)
    if(apply_KNN):        
        validation_errors_inner_KNN_matrix = np.delete(validation_errors_inner_KNN_matrix,0,0) ## Removes the first 1d array with ones.
        validation_errors_inner_KNN_matrix = np.transpose(validation_errors_inner_KNN_matrix) ## Need to transpose validation_errors_inner_KNN_matrix, such that the dimensions are (20 x 10). That is, a vector for each models performance on the inner loop CV) 
        estimated_inner_test_error_KNN_models = []
        for s in range(0,len(validation_errors_inner_KNN_matrix)):
            tmp_inner_test_error = np.sum(np.multiply(data_validation_length,validation_errors_inner_KNN_matrix[s])) / data_outer_train_length
            estimated_inner_test_error_KNN_models.append(tmp_inner_test_error)
        lowest_est_inner_error_KNN_models = min(estimated_inner_test_error_KNN_models)
        optimal_regularization_param_KNN.append(list(estimated_inner_test_error_KNN_models).index(lowest_est_inner_error_KNN_models) + 1)
        knclassifier = KNeighborsClassifier(n_neighbors=optimal_regularization_param_KNN[k_outer], p=dist, 
                            metric=metric,
                            metric_params=metric_params)
        
        knclassifier.fit(X_train_outer.squeeze(), y_train_outer.squeeze()) ## knclassifier.fit requires .squeeze of input matrices
        
        y_est_outer_model_KNN       = knclassifier.predict(X_test_outer)
        test_error_outer_KNN_tmp        = np.sum(y_est_outer_model_KNN != y_test_outer.squeeze()) / float(len(y_test_outer))
        test_error_outer_KNN.append(test_error_outer_KNN_tmp)
        
    if(apply_logistic):
         validation_errors_inner_logistics_matrix = np.delete(validation_errors_inner_logistics_matrix,0,0) 
         validation_errors_inner_logistics_matrix   = np.transpose(validation_errors_inner_logistics_matrix)
         estimated_inner_test_error_logistic_models = []
         for s in range(0,len(validation_errors_inner_logistics_matrix)):
             tmp_inner_test_error = np.sum(np.multiply(data_validation_length,validation_errors_inner_logistics_matrix[s])) / data_outer_train_length
             estimated_inner_test_error_logistic_models.append(tmp_inner_test_error)
         
         
         # Saves the regularization parameter for the best performing logit model
         lowest_est_inner_error_logistic_models = min(estimated_inner_test_error_logistic_models)
         index_lambda = list(estimated_inner_test_error_logistic_models).index(lowest_est_inner_error_logistic_models) 
         optimal_regularization_param_logistic.append(lambda_interval[index_lambda])         
        
         ## Estimate the test error on the outer test data
         model                = LogisticRegression(penalty='l2', C=1/lambda_interval[index_lambda], solver = 'lbfgs')
         model                = model.fit(X_train_outer, y_train_outer.squeeze())            
         y_est_outer_logistic = model.predict(X_test_outer)
         
         test_errors_outer_logistics_tmp = np.sum(y_est_outer_logistic != y_test_outer.squeeze()) / float(len(y_test_outer))
         test_errors_outer_logistics.append(test_errors_outer_logistics_tmp)
            
    k_outer+=1

# Estimate the generalization error
generalization_error_baseline_model = np.sum(np.multiply(test_error_outer_baseline,data_outer_test_length)) * (1/N) 
print('est gen error of baseline model: ' +str(round(generalization_error_baseline_model, ndigits=3)))  
if (apply_KNN):
    generalization_error_KNN_model = np.sum(np.multiply(test_error_outer_KNN,data_outer_test_length)) * (1/N)
    print('est gen error of KNN model: ' +str(round(generalization_error_KNN_model, ndigits=3)))    
if (apply_logistic):
    generalization_error_logistic_model = np.sum(np.multiply(test_errors_outer_logistics,data_outer_test_length)) * (1/N)
    print('est gen error of logistic model: ' +str(round(generalization_error_logistic_model, ndigits=3)))
    
# Create output table as dataframe
n_of_cols                  = sum([apply_KNN,apply_logistic])*2 + 2  
n_of_index                 = K_1 + 1 
df_output_table            = pd.DataFrame(np.ones((n_of_index,n_of_cols)),index=range(1,n_of_index + 1))
df_output_table.index.name = "Outer fold"
if(apply_KNN):
    df_output_table.columns                = ['test_data_size','K','KNN_test_error','lambda','Logistic_test_error','baseline_test_error']
    optimal_regularization_param_KNN.append('')
    optimal_regularization_param_logistic.append('')
    data_outer_test_length.append('')
    col_2                                  = list(np.array(test_error_outer_KNN).round(3)*100)
    col_2.append(round(generalization_error_KNN_model*100,ndigits=1))
    col_4                                  = list(np.array(test_errors_outer_logistics).round(3)*100)
    col_4.append(round(generalization_error_logistic_model*100,ndigits=1))    
    col_5                                  = list(np.array(test_error_outer_baseline).round(3)*100)
    col_5.append(round(generalization_error_baseline_model*100,ndigits=1))       
        
    df_output_table['test_data_size']      = data_outer_test_length    
    df_output_table['K']                   = optimal_regularization_param_KNN
    df_output_table['KNN_test_error']      = col_2
    df_output_table['lambda']              = optimal_regularization_param_logistic
    df_output_table['Logistic_test_error'] = col_4
    df_output_table['baseline_test_error'] = col_5    
# Export to csv
df_output_table.to_csv('Classification_table.csv')
# ========================================================================================================================================
# Statistical Test Evaluation (SETUP II)
# ======================================================================================================================================== 
if(apply_setup_ii):
    most_common_lambda    = stats.mode(optimal_regularization_param_logistic).mode[0].astype('float64')    
    y_true = []
    yhat = []
    
    k = 0
    for train_index,test_index in CV_setup_ii.split(X):
        print('Computing setup II CV K-fold: {0}/{1}..'.format(k+1,K_1))
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]

        model_baseline = stats.mode(y_train).mode[0][0]
        model_logistic = sklearn.linear_model.LogisticRegression(penalty='l2', C=1/most_common_lambda, solver = 'lbfgs').fit(X_train,y_train.squeeze())
        
        yhat_baseline  = np.ones((y_test.shape[0],1))*model_baseline.squeeze()
        yhat_logistic  = model_logistic.predict(X_test).reshape(-1,1) ## use reshape to ensure it is a nested array

        if(apply_KNN):
            most_common_regu_KNN  = stats.mode(optimal_regularization_param_KNN).mode[0].astype('int64')
            model_second          = KNeighborsClassifier(n_neighbors=most_common_regu_KNN, p=dist,metric=metric,
                                    metric_params=metric_params).fit(X_train.squeeze(), y_train.squeeze())
            y_hat_second_model = model_second.predict(X_test.squeeze()).reshape(-1,1) 
            
        ## Add true classes and store estimated classes    
        y_true.append(y_test)
        yhat.append( np.concatenate([yhat_baseline, yhat_logistic,y_hat_second_model], axis=1) )
        
        ## Compute the r test size and store it
        r_baseline_vs_logistic.append( np.mean( np.abs( yhat_baseline-y_test ) ** loss_in_r_function - np.abs( yhat_logistic-y_test) ** loss_in_r_function ) )
        r_baseline_vs_sec_model.append( np.mean( np.abs( yhat_baseline-y_test ) ** loss_in_r_function - np.abs( y_hat_second_model-y_test) ** loss_in_r_function ) )
        r_sec_model_vs_logistic.append( np.mean( np.abs( y_hat_second_model-y_test ) ** loss_in_r_function - np.abs( yhat_logistic-y_test) ** loss_in_r_function ) )
        
        ## add to counter
        k += 1

    # Baseline vs logistic regression    
    p_setupII_base_vs_log, CI_setupII_base_vs_log = correlated_ttest(r_baseline_vs_logistic, rho_t_test, alpha=alpha_t_test) 
    # Baseline vs 2nd model    
    p_setupII_base_vs_sec_model, CI_setupII_base_vs_sec_model = correlated_ttest(r_baseline_vs_sec_model, rho_t_test, alpha=alpha_t_test)
    # Logistic regression vs 2nd model    
    p_setupII_log_vs_sec_model, CI_setupII_log_vs_sec_model = correlated_ttest(r_sec_model_vs_logistic, rho_t_test, alpha=alpha_t_test)

    ## Create output table for statistic tests
    df_output_table_statistics = pd.DataFrame(np.ones((3,5)), columns = ['H_0','p_value','CI_lower','CI_upper','conclusion'])
    df_output_table_statistics[['H_0']] = ['err_baseline-err_logistic=0','err_KNN-err_logistic=0','baseline_model_err-err_KNN=0']
    df_output_table_statistics[['p_value']]         = [p_setupII_base_vs_log,p_setupII_log_vs_sec_model,p_setupII_base_vs_sec_model]
    df_output_table_statistics[['CI_lower']]        = [CI_setupII_base_vs_log[0],CI_setupII_log_vs_sec_model[0],CI_setupII_base_vs_sec_model[0]]
    df_output_table_statistics[['CI_upper']]        = [CI_setupII_base_vs_log[1],CI_setupII_log_vs_sec_model[1],CI_setupII_base_vs_sec_model[1]]
    rejected_null                                   = (df_output_table_statistics.loc[:,'p_value']<alpha_t_test)
    df_output_table_statistics.loc[rejected_null,'conclusion']   = 'H_0 rejected'
    df_output_table_statistics.loc[~rejected_null,'conclusion']  = 'H_0 not rejected'
    df_output_table_statistics                      = df_output_table_statistics.set_index('H_0')
    
    ## Export df as csv
    df_output_table_statistics.to_csv('Classification_Assignment_2_statistic_test.csv',encoding='UTF-8')