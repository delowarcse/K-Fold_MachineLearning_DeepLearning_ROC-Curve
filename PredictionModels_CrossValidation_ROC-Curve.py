#!/usr/bin/env python
# coding: utf-8

# # This code is the implementation of ROC Curve of Machine Learning Techniques: Logistic Regression, Decision Tree, Random Forest, Random Forest with Hyperparamters Tuning, SVM, and DNN methods

# In[ ]:


# Import necessary dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model_evaluation_utils as meu
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')

# Load and merge datasets # white = control; red = stroke; wine = data
stroke_data = pd.read_csv('Injured Participants data.csv', delim_whitespace=False)
control_data = pd.read_csv('Healthy Control Participants data.csv', delim_whitespace=False)

# store wine type as an attribute
stroke_data['data_type'] = 'stroke'   
control_data['data_type'] = 'control'

# merge control and stroke data
datas = pd.concat([stroke_data, control_data])
#datas = datas.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare Training and Testing Datasets
stp_features = datas.iloc[:,:-1]
stp_feature_names = stp_features.columns
stp_class_labels = np.array(datas['data_type'])

X_data = datas.iloc[:,:-1]
y_label = datas.iloc[:,-1]

# Data Normalization
ss = StandardScaler().fit(X_data)
X = ss.transform(X_data)
le = LabelEncoder()
le.fit(y_label)
y = le.transform(y_label)


# In[ ]:


# plots 10-fold with gold & GOLDERNROD
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp

#from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.colors as colors

import warnings
warnings.filterwarnings("ignore")

#X, y = make_classification(n_samples=500, random_state=100, flip_y=0.3)

#kf = KFold(n=len(y), n_folds=10)
kf_lr = KFold(n_splits=10, random_state=42, shuffle=True)

tprs_lr = []
aucs_lr = []
base_fpr_lr = np.linspace(0, 1, 101)

plt.figure(figsize=(6, 6))
#fig, ax = plt.subplots(1)

for i, (train_lr, test_lr) in enumerate(kf_lr.split(X,y)):
    model_lr = LogisticRegression().fit(X[train_lr], y[train_lr])
    y_score_lr = model_lr.predict_proba(X[test_lr])
    fpr_lr, tpr_lr, _ = roc_curve(y[test_lr], y_score_lr[:, 1])
    auc_lr = auc(fpr_lr, tpr_lr)

    #plt.plot(fpr_lr, tpr_lr, 'b', alpha=0.15) #label='LR (area = {:.2f})'.format(auc_lr))
    plt.plot(fpr_lr, tpr_lr, lw=2, alpha=3, color='gold', label='ROC fold %d (AUC=%f)'%(i,auc_lr))
    tpr_lr = interp(base_fpr_lr, fpr_lr, tpr_lr)
    tpr_lr[0] = 0.0
    tprs_lr.append(tpr_lr)
    aucs_lr.append(auc_lr)
    i=i+1

tprs_lr = np.array(tprs_lr)
mean_tprs_lr = tprs_lr.mean(axis=0)
std_lr = tprs_lr.std(axis=0)

tprs_upper_lr = np.minimum(mean_tprs_lr + std_lr, 1)
tprs_lower_lr = mean_tprs_lr - std_lr

plt.plot(base_fpr_lr, mean_tprs_lr, 'b')
#plt.plot(base_fpr_lr, mean_tprs_lr)
mean_auc_lr = auc(base_fpr_lr, mean_tprs_lr)
#print('Auc: ', aucs_lr)
#print('mean auc: %f' %mean_auc_lr)

plt.plot(base_fpr_lr, mean_tprs_lr, color='goldenrod', label=r'Mean ROC (AUC=%f)'%(mean_auc_lr), lw=2, alpha=1)
#plt.fill_between(base_fpr_lr, tprs_lower_lr, tprs_upper_lr, color='grey', alpha=0.3)

plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('LR ROC Curve')
plt.axes().set_aspect('equal', 'datalim')

plt.legend(loc="lower right")
plt.show()


# In[ ]:


# Green
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier

kf_dt = KFold(n_splits=10, shuffle=True)

tprs_dt = []
aucs_dt = []
base_fpr_dt = np.linspace(0, 1, 101)

plt.figure(figsize=(6, 6))
#fig, ax = plt.subplots(1)

for i, (train_dt, test_dt) in enumerate(kf_dt.split(X,y)):
    model_dt = DecisionTreeClassifier(max_depth=4).fit(X[train_dt], y[train_dt])
    y_score_dt = model_dt.predict_proba(X[test_dt])
    fpr_dt, tpr_dt, _ = roc_curve(y[test_dt], y_score_dt[:, 1])
    auc_dt = auc(fpr_dt, tpr_dt)

    #plt.plot(fpr_dt, tpr_dt, 'b', alpha=0.15) #label='LR (area = {:.2f})'.format(auc_lr))
    plt.plot(fpr_dt, tpr_dt, lw=2, alpha=3, color='lightgreen', label='ROC fold %d (AUC=%f)'%(i,auc_dt))
    tpr_dt = interp(base_fpr_dt, fpr_dt, tpr_dt)
    tpr_dt[0] = 0.0
    tprs_dt.append(tpr_dt)
    aucs_dt.append(auc_dt)

tprs_dt = np.array(tprs_dt)
mean_tprs_dt = tprs_dt.mean(axis=0)
std_dt = tprs_dt.std(axis=0)

tprs_upper_dt = np.minimum(mean_tprs_dt + std_dt, 1)
tprs_lower_dt = mean_tprs_dt - std_dt

plt.plot(base_fpr_dt, mean_tprs_dt, 'b')
mean_auc_dt = auc(base_fpr_dt, mean_tprs_dt)
#print('Auc_dt: ', aucs_dt)
#print('mean auc: %f' %mean_auc_dt)

#plt.fill_between(base_fpr_dt, tprs_lower_dt, tprs_upper_dt, color='grey', alpha=0.3)
plt.plot(base_fpr_dt, mean_tprs_dt, color='green', label=r'Mean ROC (AUC=%f)'%(mean_auc_dt), lw=2, alpha=1)

plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('DT ROC Curve')
plt.axes().set_aspect('equal', 'datalim')

# Hide the right and top spines
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
#ax.yaxis.set_ticks_position('left')
#ax.xaxis.set_ticks_position('bottom')

plt.legend(loc="lower right")
plt.show()
plt.close()


# In[ ]:


# royalblue & blue
from sklearn.ensemble import RandomForestClassifier

cv_rf = KFold(n_splits=10, shuffle=True) #random_state=42,

tprs_rf = []
aucs_rf = []
base_fpr_rf = np.linspace(0, 1, 101)

plt.figure(figsize=(6, 6))
#fig, ax = plt.subplots(1)

for i, (train_rf, test_rf) in enumerate(cv_rf.split(X,y)):
    model_rf = RandomForestClassifier().fit(X[train_rf], y[train_rf])
    y_score_rf = model_rf.predict_proba(X[test_rf])
    fpr_rf, tpr_rf, _ = roc_curve(y[test_rf], y_score_rf[:, 1])
    auc_rf = auc(fpr_rf, tpr_rf)

    #plt.plot(fpr_rf, tpr_rf, 'b', alpha=0.15) #label='LR (area = {:.2f})'.format(auc_lr))
    plt.plot(fpr_rf, tpr_rf, lw=2, alpha=3, color='royalblue', label='ROC fold %d (AUC=%f)'%(i,auc_rf))
    tpr_rf = interp(base_fpr_rf, fpr_rf, tpr_rf)
    tpr_rf[0] = 0.0
    tprs_rf.append(tpr_rf)
    aucs_rf.append(auc_rf)

tprs_rf = np.array(tprs_rf)
mean_tprs_rf = tprs_rf.mean(axis=0)
std_rf = tprs_rf.std(axis=0)

tprs_upper_rf = np.minimum(mean_tprs_rf + std_rf, 1)
tprs_lower_rf = mean_tprs_rf - std_rf
#print(tprs_upper_rf)
#print(tprs_lower_rf)

plt.plot(base_fpr_rf, mean_tprs_rf, 'b')
mean_auc_rf = auc(base_fpr_rf, mean_tprs_rf)
#print('Auc_rf: ', aucs_rf)
#print('mean auc: %f' %mean_auc_rf)

#plt.fill_between(base_fpr_rf, tprs_lower_rf, tprs_upper_rf) # , color='grey' , alpha=0.3
plt.plot(base_fpr_rf, mean_tprs_rf, color='blue', label=r'Mean ROC (AUC=%f)'%(mean_auc_rf), lw=2, alpha=1)

plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('RF ROC Curve')
plt.axes().set_aspect('equal', 'datalim')

plt.legend(loc="lower right")
plt.show()
plt.close()


# In[ ]:


# aquamarine & turquoise
# Random Forest with Hyperparameter Tuning
from sklearn.ensemble import RandomForestClassifier

cv_rft = KFold(n_splits=10, shuffle=True) # random_state=42

tprs_rft = []
aucs_rft = []
base_fpr_rft = np.linspace(0, 1, 101)

plt.figure(figsize=(6, 6))
#fig, ax = plt.subplots(1)

for i, (train_rft, test_rft) in enumerate(cv_rft.split(X,y)):
    model_rft = RandomForestClassifier(n_estimators=200, max_features='auto', random_state=42).fit(X[train_rft], y[train_rft])
    y_score_rft = model_rft.predict_proba(X[test_rft])
    fpr_rft, tpr_rft, _ = roc_curve(y[test_rft], y_score_rft[:, 1])
    auc_rft = auc(fpr_rft, tpr_rft)

    #plt.plot(fpr_rft, tpr_rft, 'b', alpha=0.15) #label='LR (area = {:.2f})'.format(auc_lr))
    plt.plot(fpr_rft, tpr_rft, lw=2, alpha=3, color='aquamarine', label='ROC fold %d (AUC=%f)'%(i,auc_rft))
    tpr_rft = interp(base_fpr_rft, fpr_rft, tpr_rft)
    tpr_rft[0] = 0.0
    tprs_rft.append(tpr_rft)
    aucs_rft.append(auc_rft)

tprs_rft = np.array(tprs_rft)
mean_tprs_rft = tprs_rft.mean(axis=0)
std_rft = tprs_rft.std(axis=0)

tprs_upper_rft = np.minimum(mean_tprs_rft + std_rft, 1)
tprs_lower_rft = mean_tprs_rft - std_rft

plt.plot(base_fpr_rft, mean_tprs_rft, 'b')
mean_auc_rft = auc(base_fpr_rft, mean_tprs_rft)
#print('Auc_rft: ', aucs_rft)
#print('mean auc: %f' %mean_auc_rft)

#plt.fill_between(base_fpr_rft, tprs_lower_rft, tprs_upper_rft, color='grey', alpha=0.3)
plt.plot(base_fpr_rft, mean_tprs_rft, color='turquoise', label=r'Mean ROC (AUC=%f)'%(mean_auc_rft), lw=2, alpha=1)

plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('RFT ROC Curve')
plt.axes().set_aspect('equal', 'datalim')

# Hide the right and top spines
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
#ax.yaxis.set_ticks_position('left')
#ax.xaxis.set_ticks_position('bottom')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# cyan & darkcyan
from sklearn.svm import SVC

cv_svm = KFold(n_splits=10, shuffle=True) # random_state=42

tprs_svm = []
aucs_svm = []
base_fpr_svm = np.linspace(0, 1, 101)

plt.figure(figsize=(6, 6))
#fig, ax = plt.subplots(1)

for i, (train_svm, test_svm) in enumerate(cv_svm.split(X,y)):
    model_svm = SVC(probability=True).fit(X[train_svm], y[train_svm])
    y_score_svm = model_svm.predict_proba(X[test_svm])
    fpr_svm, tpr_svm, _ = roc_curve(y[test_svm], y_score_svm[:, 1])
    auc_svm = auc(fpr_svm, tpr_svm)

    #plt.plot(fpr_svm, tpr_svm, 'b', alpha=0.15) #label='LR (area = {:.2f})'.format(auc_lr))
    plt.plot(fpr_svm, tpr_svm, lw=2, alpha=3, color='cyan', label='ROC fold %d (AUC=%f)'%(i,auc_svm))
    tpr_svm = interp(base_fpr_svm, fpr_svm, tpr_svm)
    tpr_svm[0] = 0.0
    tprs_svm.append(tpr_svm)
    aucs_svm.append(auc_svm)

tprs_svm = np.array(tprs_svm)
mean_tprs_svm = tprs_svm.mean(axis=0)
std_svm = tprs_svm.std(axis=0)

tprs_upper_svm = np.minimum(mean_tprs_svm + std_svm, 1)
tprs_lower_svm = mean_tprs_svm - std_svm

plt.plot(base_fpr_svm, mean_tprs_svm, 'b')
mean_auc_svm = auc(base_fpr_svm, mean_tprs_svm)
#print('Auc_svm: ', aucs_svm)
#print('mean auc: %f' %mean_auc_svm)

#plt.fill_between(base_fpr_svm, tprs_lower_svm, tprs_upper_svm, color='grey', alpha=0.3)
plt.plot(base_fpr_svm, mean_tprs_svm, color='darkcyan', label=r'Mean ROC (AUC=%f)'%(mean_auc_svm), lw=2, alpha=1)

plt.plot([0, 1], [0, 1],'b--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('SVM ROC Curve')
plt.axes().set_aspect('equal', 'datalim')

# Hide the right and top spines
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
#ax.yaxis.set_ticks_position('left')
#ax.xaxis.set_ticks_position('bottom')
plt.legend(loc="lower right")
plt.show()
plt.close()


# In[ ]:


# darkred and red
# https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
import numpy as np

# define k-fold cross validation
KF_dnn = KFold(n_splits=10, random_state=42, shuffle=True)

plt.figure(figsize=(6,6))

mean_fpr_dnn = np.linspace(0, 1, 101)
tprs_dnn = []
aucs_dnn = []
for i, (train_dnn, test_dnn) in enumerate(KF_dnn.split(X,y)):
    #create model
    model_dnn = Sequential()
    model_dnn.add(Dense(12, input_dim=12, activation='relu'))
    model_dnn.add(Dense(8, activation='relu'))
    model_dnn.add(Dense(1, activation='sigmoid'))
    
    #compile & fit
    model_dnn.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model_dnn.fit(X[train_dnn],y[train_dnn], epochs=100, batch_size=10, verbose=0)
                  
    # evaluate
    y_pred_keras = model_dnn.predict_proba(X[test_dnn]).ravel()
    
    fpr_dnn, tpr_dnn, thresholds = roc_curve(y[test_dnn], y_pred_keras)              
    tprs_dnn.append(interp(mean_fpr_dnn, fpr_dnn,tpr_dnn))
    roc_auc_dnn = auc(fpr_dnn,tpr_dnn)
    aucs_dnn.append(roc_auc_dnn)   
    plt.plot(fpr_dnn, tpr_dnn, lw=2, alpha=3, color='darkred', label='ROC fold %d (AUC=%f)'%(i,roc_auc_dnn))
    i = i+1
                  
plt.plot([0,1], [0,1], linestyle='--', lw=2, color='black')
mean_tpr_dnn = np.mean(tprs_dnn, axis=0)
mean_auc_dnn = auc(mean_fpr_dnn, mean_tpr_dnn)
plt.plot(mean_fpr_dnn, mean_tpr_dnn, color='red', label=r'Mean ROC (AUC=%f)'%(mean_auc_dnn), lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DNN ROC Curve')
plt.legend(loc="lower right")
plt.show()                  


# In[ ]:


fig, ax =plt.subplots(1)
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot([0,1], [0,1], 'k--')
plt.plot(base_fpr_lr, mean_tprs_lr, color='goldenrod', label=r'LR - Mean ROC (AUC=%f)'%(mean_auc_lr))
plt.plot(base_fpr_dt, mean_tprs_dt, color='green', label=r'DT - Mean ROC (AUC=%f)'%(mean_auc_dt))
plt.plot(base_fpr_rf, mean_tprs_rf, color='blue', label=r'RF - Mean ROC (AUC=%f)'%(mean_auc_rf))
plt.plot(base_fpr_rft, mean_tprs_rft, color='turquoise', label=r'RFT - Mean ROC (AUC=%f)'%(mean_auc_rft))
plt.plot(base_fpr_svm, mean_tprs_svm, color='darkcyan', label=r'SVM - Mean ROC (AUC=%f)'%(mean_auc_svm))
plt.plot(mean_fpr_dnn, mean_tpr_dnn, color='red', label=r'DNN - Mean ROC (AUC=%f)'%(mean_auc_dnn))

plt.xlabel('False Positive Rate/Specificity')
plt.ylabel('True Positive Rate/Sensitivity')
#plt.title('ROC Curve for All Models')
plt.legend(loc="lower right")

# hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# 

plt.show()


# In[ ]:




