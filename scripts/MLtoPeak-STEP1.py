# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:23:59 2024

@author: Mengyu Tan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix as CM 
from sklearn.metrics import precision_recall_fscore_support, log_loss
import joblib
import sys
import argparse
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

def get_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
 
    return os.path.normpath(os.path.join(base_path, relative_path))

train_RF_model = {
        'M': joblib.load(r'./model/RF_model_M.pkl'),
        'S': joblib.load(r'./model/RF_model_S.pkl'),
        'S&M': joblib.load(r'./model/RF_model_S&M.pkl'),
    }

train_LR_model = {
        'M': joblib.load(r'./model/LR_model_M.pkl'),
        'S': joblib.load(r'./model/LR_model_S.pkl'),
        'S&M': joblib.load(r'./model/LR_model_S&M.pkl'),
    }

train_SVM_model = {
        'M': joblib.load(r'./model/SVM_model_M.pkl'),
        'S': joblib.load(r'./model/SVM_model_S.pkl'),
        'S&M': joblib.load(r'./model/SVM_model_S&M.pkl'),
    }

train_MLP_model = {
        'M': joblib.load(r'./model/MLP_model_M.pkl'),
        'S': joblib.load(r'./model/MLP_model_S.pkl'),
        'S&M': joblib.load(r'./model/MLP_model_S&M.pkl'),
    }

#%% prediction probability (train)
def probability_excel_out(clf, filename):
    predict_proba = pd.DataFrame(clf.predict_proba(data_x),columns=data_y.drop_duplicates())
    
    #Save both the predicted and the actual
    output = raw_data.copy()
    type(output)
    
    output = output.reset_index(drop=True)
    output['pre_y'] = clf.predict(data_x)
    output['Actual_y'] = data_y
    output = pd.merge(output
                      , predict_proba
                      , left_on=output.index.values
                      , right_on=predict_proba.index.values
                      , how="inner"
                      )
    
    output.to_excel(args.TrainOutputPath + '/' + filename + "_预测和真实值都有.xlsx")
#%% Get each type of prediction probability (test)
def probability_excel_out_test(clf, filename):
    predict_proba = pd.DataFrame(clf.predict_proba(test_x)
                                 ,columns=['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']
                                 )
    
    #同时将预测的和实际的都保存下来
    output = test_data.copy()
    type(output)
    
    output = output.reset_index(drop=True)
    output['prediction'] = clf.predict(test_x)
    # output['Actual_y'] = data_y
    output = pd.merge(output
                      , predict_proba
                      , left_on=output.index.values
                      , right_on=predict_proba.index.values
                      , how="inner"
                      )
    output.to_excel(args.TestOutputPath + '/' + filename + "预测统计.xlsx")
    print(filename + ' ' + 'probability_excel_out'+ 'ok!')
#%%###confusion_matrics###
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 12,
}

def confusion_matrics(clf, filename, dataX, dataY):
    plt.clf() 
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 10
    confusion_matrics = CM(y_true = dataY
                           , y_pred = clf.predict(dataX)                                        
                           , labels = ['Allele', 'Back_Stutter', 'Forward_Stutter', 'Pull_Up']) 
    if dataY.shape == (round(raw_data.shape[0]*0.7),):
        title = 'train'
        plt.imshow(confusion_matrics, cmap=plt.cm.Oranges)
        plt.title('The ' + filename +' confusion matrics of train set', font1)  # 图形标题
    elif dataY.shape == (round(raw_data.shape[0]*0.3),):
        title = 'test'
        plt.imshow(confusion_matrics, cmap=plt.cm.GnBu)
        plt.title('The ' + filename +' confusion matrics of test set', font1) 
    
    indices = range(len(confusion_matrics))
    plt.xticks(indices, labels = ['Allele', 'B_Stutter', 'F_Stutter', 'Pull_Up'])
    plt.yticks(indices, labels = ['Allele', 'B_Stutter', 'F_Stutter', 'Pull_Up'])
    plt.xlabel("Predicted", font1)
    plt.ylabel("True", font1)
    plt.colorbar()
    for first_index in range(len(confusion_matrics)):    #第几行
        for second_index in range(len(confusion_matrics[first_index])):#第几列
            if confusion_matrics[second_index][first_index]/dataY.shape[0] > 0.2:
                plt.text(first_index, second_index
                         , confusion_matrics[second_index][first_index]
                         , font1
                         , horizontalalignment='center', verticalalignment='center'
                         , color='white')
            else:
                plt.text(first_index, second_index
                         , confusion_matrics[second_index][first_index]
                         , font1
                         , horizontalalignment='center', verticalalignment='center')
    plt.savefig(args.TrainOutputPath + '/' + filename + '_' + title + '_confusion_matrics.png', dpi=300) #指定分辨率保存

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a script.")
    parser.add_argument('--Model', type=str, default="M",
                        help="Enter the type of model(\"S\",\"M\" and \"S&M\" respectively represents model trained"
                             " using peak data from single-source profiles, mixed DNA profiles and a combination of "
                             "peak data from single-source profiles and mixed DNA profiles)")
    parser.add_argument('--Customizable', type=str, default="N",
                        help="Please choose whether to customize the model. The default is not to customize. Enter any character other than 'N' to enable customization.")
    parser.add_argument('--TrainInputFile', type=str, default="None", help="Enter the path and filename of training data file") 
    parser.add_argument('--TrainOutputPath', type=str, default="./", help="Enter the path of training output file") 
    parser.add_argument('--TestInputFile', type=str, default="None", help="Enter the path and filename of testing data file")
    parser.add_argument('--TestOutputPath', type=str, default="./", help="Enter the path of output file")

    args = parser.parse_args()

    if args.Model:
        print(f"The type of model: {args.Model}")
    else:
        print("Please enter the type of model.")

    if args.TrainInputFile != "None":
        raw_data = pd.read_csv(args.TrainInputFile)
        
        col = raw_data.columns.values.tolist()
        col1 = col[1:-1]
        data_x = np.array(raw_data[col1]) 
        data_y = raw_data['Classification'] #label
        
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, stratify=data_y, random_state=10)
        cv = StratifiedShuffleSplit(n_splits=30, test_size=0.3, random_state=420)

#%% SVM (Gridsearch)
        print("Searching for SVM model hyperparameters......")
        kernel_list = ['rbf', 'poly']
        C_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        gamma_range = np.logspace(-10, 1, 20)
        
        param_grid = dict(kernel = kernel_list
                          , C = C_range
                          , gamma = gamma_range)
        
        # cv = StratifiedShuffleSplit(n_splits=30, test_size=0.3, random_state=420)
        grid = GridSearchCV(SVC(coef0 = 0.0, degree=1,cache_size=5000), param_grid=param_grid,cv=cv)
        grid.fit(x_train,y_train)
        
        print("The best parameters of SVM are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))

## SVM model
        SVM = SVC(kernel=grid.best_params_.get('kernel')
                      , C=grid.best_params_.get('C')
                      , gamma = grid.best_params_.get('gamma')
                      , coef0 = 0.0
                      , degree = 1
                      , cache_size=5000
                      , probability=True
                      ).fit(x_train, y_train)
        
        
        probability_excel_out(SVM, 'SVM')
        
        confusion_matrics(SVM, 'SVM', x_train, y_train)
        confusion_matrics(SVM, 'SVM', x_test, y_test) 
        joblib.dump(SVM, r'./model/new_SVM_model_' + args.Model + '.pkl')
        print('SVM Classifier has been modeled')
        print('SVM model test metrics:')
        print('Accuracy：' + str(SVM.score(x_test, y_test)))
        precision, recall, F1, _ = precision_recall_fscore_support(np.array(y_test), np.array(SVM.predict(x_test)), average='macro')
        print('precision:{0:.8f}, recall:{1:.8f}, F1:{2:.8f}'.format(precision, recall, F1))
        loss = log_loss(y_test, SVM.predict_proba(x_test))
        print(f"Log Loss: {loss}")
        
#%% LR (Gridsearch)
        print("Searching for LR model hyperparameters......")
        solver_range = ["lbfgs", "newton-cg", "sag", "saga"]
        C_range = np.linspace(0.05, 1, 19)
        
        param_grid = dict(solver = solver_range
                          , C = C_range
                          )
        LR = LogisticRegression(max_iter=10000, multi_class="multinomial", random_state=10)
        # cv = StratifiedShuffleSplit(n_splits=30, test_size=0.3, random_state=420)
        grid = GridSearchCV(LR, param_grid=param_grid, cv=cv)
        grid.fit(x_train, y_train)
        
        print("The best parameters of LR are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))

## LR model
        LR = LogisticRegression(solver=grid.best_params_.get('solver')
                                , max_iter=10000
                                , multi_class="multinomial"
                                , C=grid.best_params_.get('C')
                                , random_state=10).fit(x_train, y_train)
        
        
        probability_excel_out(LR, 'LR')
        confusion_matrics(LR, 'LR', x_train, y_train)
        confusion_matrics(LR, 'LR', x_test, y_test)
        joblib.dump(LR, r'./model/new_LR_model_' + args.Model + '.pkl')
        print('LR Classifier has been modeled')
        print('LR model test metrics:')
        print('Accuracy：' + str(LR.score(x_test, y_test)))
        precision, recall, F1, _ = precision_recall_fscore_support(np.array(y_test), np.array(LR.predict(x_test)), average='macro')
        print('precision:{0:.8f}, recall:{1:.8f}, F1:{2:.8f}'.format(precision, recall, F1))
        loss = log_loss(y_test, LR.predict_proba(x_test))
        print(f"Log Loss: {loss}")
        
                     
#%% RF (Gridsearch)
        print("Searching for RF model hyperparameters......(may take a long time)")
        n_estimators_range = np.arange(1,101,10)
        max_depth_range = np.arange(1,20,2)
        min_samples_split_range = np.arange(2,2+10,2)
        min_samples_leaf_range = np.arange(1,10,2)
        max_features_range = np.arange(10,300,10)
        
        param_grid = dict(n_estimators = n_estimators_range
                          , max_depth = max_depth_range
                          , min_samples_split = min_samples_split_range
                          , min_samples_leaf = min_samples_leaf_range
                          , max_features = max_features_range
                          )
        RFC = RandomForestClassifier(random_state=10)
        # cv = StratifiedShuffleSplit(n_splits=30, test_size=0.3, random_state=420)
        grid = GridSearchCV(RFC, param_grid=param_grid, cv=cv)
        grid.fit(x_train, y_train)
        
        print("The best parameters of RF are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))
        
## RF model
        RF = RandomForestClassifier(n_estimators=grid.best_params_.get('n_estimators')
                                      , max_depth=grid.best_params_.get('max_depth')
                                      , min_samples_leaf=grid.best_params_.get('min_samples_leaf')
                                      , min_samples_split=grid.best_params_.get('min_samples_split')
                                      , max_features=grid.best_params_.get('max_features')
                                      , random_state=10).fit(x_train, y_train)
        
        probability_excel_out(RF, 'RF')##
        confusion_matrics(RF, 'RF', x_train, y_train)
        confusion_matrics(RF, 'RF', x_test, y_test) 
        joblib.dump(RF, r'./model/new_RF_model_' + args.Model + '.pkl')
        print('RF Classifier has been modeled')
        print('RF model test metrics:')
        print('Accuracy：' + str(RF.score(x_test, y_test)))
        precision, recall, F1, _ = precision_recall_fscore_support(np.array(y_test), np.array(RF.predict(x_test)), average='macro')
        print('precision:{0:.8f}, recall:{1:.8f}, F1:{2:.8f}'.format(precision, recall, F1))
        loss = log_loss(y_test, RF.predict_proba(x_test))
        print(f"Log Loss:{loss}")
        

#%% MLP (Gridsearch)
        print("Searching for MLP model hyperparameters......(may take a long time)")
        hidden_layer_sizes_range = [(50,100,100,50)
                                    , (50,100,50)
                                    , (100,100)
                                    , (50,100)
                                    , (100)
                                    ]
        activation_range = ["identity", "logistic", "tanh", "relu"]
        solver_range = ["lbfgs", "sgd", "adam"]
        alpha_range = [0.0001, 0.05]
        
        param_grid = dict(hidden_layer_sizes = hidden_layer_sizes_range
                          , activation = activation_range
                          , solver = solver_range
                          , alpha = alpha_range
                          )
        DNN = MLPClassifier(max_iter=10000, random_state=10)
        # cv = StratifiedShuffleSplit(n_splits=30, test_size=0.3, random_state=420)
        grid = GridSearchCV(DNN, param_grid=param_grid, cv=cv)
        grid.fit(x_train, y_train)
        
        print("The best parameters of MLP are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))

## MLP model
        MLP = MLPClassifier(hidden_layer_sizes=grid.best_params_.get('hidden_layer_sizes')
                             , activation=grid.best_params_.get('activation')
                             , solver=grid.best_params_.get('solver')
                             , alpha=grid.best_params_.get('alpha')
                             , max_iter=10000
                             , random_state=10).fit(x_train, y_train)
        
        
        probability_excel_out(MLP, 'MLP')
        confusion_matrics(MLP, 'MLP', x_train, y_train)
        confusion_matrics(MLP, 'MLP', x_test, y_test) 
        joblib.dump(MLP, r'./model/new_MLP_model_' + args.Model + '.pkl')
        print('MLP Classifier has been modeled')
        print('MLP model test metrics:')
        print('Accuracy：' + str(MLP.score(x_test, y_test)))
        precision, recall, F1, _ = precision_recall_fscore_support(np.array(y_test), np.array(MLP.predict(x_test)), average='macro')
        print('precision:{0:.8f}, recall:{1:.8f}, F1:{2:.8f}'.format(precision, recall, F1))
        loss = log_loss(y_test, MLP.predict_proba(x_test))
        print(f"Log Loss: {loss}")        
#%%    
    else:
        test_data = pd.read_csv(args.TestInputFile)
        col = test_data.columns.values.tolist()
        col1 = col[1:]
        test_x = np.array(test_data[col1]) 
        
        if args.Customizable != "N":
            RF = joblib.load(r'./model/' + 'new_RF_model_' + args.Model + '.pkl')
            LR = joblib.load(r'./model/' + 'new_LR_model_' + args.Model + '.pkl')
            SVM = joblib.load(r'./model/' + 'new_SVM_model_' + args.Model + '.pkl')
            MLP = joblib.load(r'./model/' + 'new_MLP_model_' + args.Model + '.pkl')
        else:
            RF = train_RF_model[args.Model]
            LR = train_LR_model[args.Model]
            SVM = train_SVM_model[args.Model]
            MLP = train_MLP_model[args.Model]
        
        probability_excel_out_test(RF, 'RF')
        probability_excel_out_test(LR, 'LR')
        probability_excel_out_test(SVM, 'SVM')
        probability_excel_out_test(MLP, 'MLP')
