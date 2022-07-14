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
