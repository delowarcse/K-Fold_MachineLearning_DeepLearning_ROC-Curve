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

# Random Forest
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import roc_curve, auc

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
