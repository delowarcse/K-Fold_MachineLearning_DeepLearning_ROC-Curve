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

# import libraries for cross validation
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import roc_curve, auc

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
