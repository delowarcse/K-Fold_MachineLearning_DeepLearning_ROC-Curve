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

# Random Forest with Hyperparameters Tuning
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import roc_curve, auc

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
