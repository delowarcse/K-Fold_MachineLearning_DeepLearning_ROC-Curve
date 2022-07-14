# Import necessary dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model_evaluation_utils as meu
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

%matplotlib inline

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
