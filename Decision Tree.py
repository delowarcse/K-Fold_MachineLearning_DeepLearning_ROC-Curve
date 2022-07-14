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


# Decision Tree Classifier
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
