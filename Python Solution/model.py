#!/usr/bin/env python
# coding: utf-8

# ## Importing Modules




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Sampling
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

#Testing
from scipy.stats import pearsonr

#Metrics
from sklearn.metrics import roc_auc_score, precision_score, recall_score

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb

import pickle


# ## Defining Functions for Accuracy Metrics & Modelling




def acc_metrics(actual_values,predicted_values):
    roc_auc=round(roc_auc_score(actual_values,predicted_values),2)
    precision=round(precision_score(actual_values,predicted_values),2)
    recall = round(recall_score(actual_values,predicted_values),2)
        
    print('\nROC_AUC Score: ',roc_auc)
    print('Precision Score: ',precision)
    print('Recall Score: ',recall,'\n')
    
    return roc_auc,precision,recall





def modelling(x_train, x_test, y_train, y_test):
    
    print('\nLogistic Regression Modelling')
    LR_model=LogisticRegression(solver='liblinear')
    LR_model.fit(x_train,y_train)
    acc_metrics(y_test,LR_model.predict(x_test))
    
    print('\nDecision Tree Classifier Modelling')
    DT_model=DecisionTreeClassifier(criterion='entropy')
    DT_model.fit(x_train,y_train)
    acc_metrics(y_test,DT_model.predict(x_test))
    
    print('\nRandom Forest Classifier Modelling')
    RF_model=RandomForestClassifier(n_estimators = 10)
    RF_model.fit(x_train,y_train)
    acc_metrics(y_test,RF_model.predict(x_test))
    
    print('\nNaive Bayes Modelling')
    NB_model=GaussianNB()
    NB_model.fit(x_train,y_train)
    acc_metrics(y_test,NB_model.predict(x_test))
    
    return LR_model,RF_model, DT_model, NB_model


# ## Importing Datasets




train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')





print('Train Data Shape: ',train_data.shape,'\ntest Data Shape: ',test_data.shape)





train_data.info()





test_data.info()





train_data.head()





test_data.head()


# ###### Observations :  
# Train & Test Data both contains 200000 rows, 200 feature columns [ var_0,var_1,...,var_199],       one 'ID_code' column.Train data has 'target' variable as label 

# ######   

# ## Data Pre-Processing




# Dropping the string column ‘ID_code’, as this feature only represents the index of the observations
# in the given dataset & have no impact in prediction of the target variable.

train_data.drop(columns='ID_code',inplace=True)





#shape of train data

train_data.shape


# ######   

# ## Exploratory Data Analysis

# ### 1. Missing Value Check




# Training Data
train_data.isna().sum().sum()





#Testing Data
test_data.isna().sum().sum()


# ###### Observations : 
# No missing values in Train or Test Data

# ######   

# ### 2. Target Class Distribution 




#sns.barplot(x=train_data['target'].unique(),y=train_data['target'].value_counts())
plt.figure(figsize=(10,6))
sns.countplot(x ='target', data = train_data) 
plt.title('TARGET CLASS DISTRIBUTION')
plt.xlabel('TARGET')
plt.ylabel('COUNT')
plt.show()





train_data.target.value_counts()





per=train_data.target.value_counts()
print('Percentage of class 0: ',round(per[0]/(per[0]+per[1]),2)*100,'%')
print('Percentage of class 1: ',round(per[1]/(per[0]+per[1]),2)*100,'%')


# ###### Observations : 
# The above distribution shows that the dataset is an imbalanced Dataset & have majority of class 0 i.e. 90% of data is for class 0. 

# ######   

# ### 3. Data Description




train_data.describe()


# ###### Observations : 
# Mean of some variables are very high & difference between mean and max for some variables is quite high too. Which shows the presence of outliers in the dataset.

# ######   

# ### 4. Outlier Analysis




#Detect and replace outliers with mean
for i in train_data.drop(columns='target').columns:
    
    q75, q25 = np.percentile(train_data.loc[:,i], [75 ,25])
    iqr = q75 - q25
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
        
    train_data[i][train_data[i] < minimum]= np.nan
    train_data[i][train_data[i] > maximum]= np.nan
    print(i)
    print('Max out bound: ',maximum)
    print('Min out bound: ',minimum)
    print('Total No. of Outliers: ',train_data[i].isna().sum())
    train_data[i].fillna(train_data[i].mean(),inplace=True)
    
    


# ###### Observations : 
# Total outliers found in train data is 26536. These Outliers are then replaced by the column’s mean value.

# ######   

# ### 5. Numerical Data Distribution




plt.figure(figsize=(30,100))
for i,col in enumerate(train_data.drop(columns='target').columns):
    plt.subplot(25,8,i+1)
    sns.distplot(train_data[col])
    plt.title(col)
plt.show()    


# ###### Observations :
# All the features in Dataset, pretty much follows normalised distribution.
# 

# ######   

# ### 6. Numerical Data Distribution per target Class




plt.figure(figsize=(40,200))
for i,col in enumerate(train_data.drop(columns='target').columns):
    plt.subplot(50,4,i+1)
    sns.distplot(train_data[train_data['target']==0][col],hist=False,label='0',color='green')
    sns.distplot(train_data[train_data['target']==1][col],hist=False,label='1',color='red')


# ###### Observations :
# Numerical columns almost follows same distribution for both classes.

# ######   

# ### 7. Distribution of Mean, Median & Standard Deviation




plt.figure(figsize=(16,6))
features=train_data.drop(columns='target').columns
plt.title("Distribution of mean, median & standard deviation values per column in the train set")

sns.distplot(train_data[features].mean(axis=0),color="green", kde=True,bins=120, label='Mean')
sns.distplot(train_data[features].median(axis=0),color="blue", kde=True,bins=120, label='Median')
sns.distplot(train_data[features].std(axis=0),color="red", kde=True,bins=120, label='StD')
plt.legend()
plt.show()


# ###### Observations:
# Mean values are distributed over a large range.
# 
# Moreover mean and median have similar distribution.
# 
# Standard deviation is relatively large.
# 
# 

# ######   

# ### 8. Distribution of Skew values per column




t0 = train_data.loc[train_data['target'] == 0]
t1 = train_data.loc[train_data['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of skew values per column in the train set")
sns.distplot(t0[features].skew(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].skew(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# ######   

# ### 9. Distribution of Kurtosis values per column




t0 = train_data.loc[train_data['target'] == 0]
t1 = train_data.loc[train_data['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis values per column in the train set")
sns.distplot(t0[features].kurtosis(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].kurtosis(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# ######   

# ### 10. Feature Correlation with each other




col=['var_'+str(i) for i in range(0,200)]
corr=train_data.loc[:,col].corr()
d=pd.DataFrame(corr.abs().unstack().sort_values().reset_index())
d=d[d.level_0!=d.level_1]
d





plt.title("Feature-Feature Correlation plot")
sns.distplot(d[0],kde=True)


# ######  Observations :
# All Features have inter feature correlation value less than 0.02, Thus features are not correlated to each other.
# 

# ######   

# ### 11. Feature-Target Correlation




col=['var_'+str(i) for i in range(0,200)]
col.append('target')
corr=train_data.loc[:,col].corr()
d=pd.DataFrame(corr.abs().unstack().sort_values(ascending=False).reset_index())
d=d[d.level_0!=d.level_1]
d=d[d.level_0=='target']
d





plt.title("Feature-Target Correlation plot")
sns.distplot(d[0],kde=True)





print('Top 10 most correlated variable:')
d.head(10)





print('Top 10 least correlated variable:')
d.tail(10).sort_values(by=0,ascending=True)


# ###### Observations :
# 10 Most Correlated Feature with Target: [ var_81, var_139, var_12, var_6, var_53, var_110, var_26, var_174, var_76, var_146].
# 
# 10 Least Correlated Feature with Target: [ var_185, var_30, var_27, var_17, var_38, var_41, var_126, var_103, var_10, var_100]
# 

# ######   

# ### 12. Feature Selection

# Pearson Correlation & Significance Hypothesis Testing:
#  
# Lets assume Significance Level, SL is 0.05
# 
# Null Hypothesis:       Feature and Target doesnot have Linear Relationship : if calc_SL > assumed_SL
# 
# Alternate Hypothesis:  Feature and target have Linear Relationship : if calc_SL <= assumed_SL




col=['var_'+str(i) for i in range(0,200)]
col_reduced=[]
for i in col:
    corr,p_value = pearsonr(train_data[i],train_data["target"])
   
    if p_value > 0.05:
        col_reduced.append(i)
        print("For Column: ",i)
        print("Correlation value: ",corr,"   P Value: ",p_value)
        print("Null Hypothesis Passed. ",i," and Target doesnot have Linear Relationship\n\n")       





col=col_reduced.copy()
col.append('target')
corr=train_data.loc[:,col].corr()
f, ax = plt.subplots(figsize=(10, 10))
plt.title('Feature - Target correlation map')
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)


# ###### Feature variable ['var_7', 'var_10', 'var_17', 'var_27', 'var_30', 'var_38', 'var_39', 'var_41', 'var_96', 'var_98', 'var_100', 'var_103', 'var_117', 
# ###### 'var_126', 'var_136', 'var_158', 'var_161', 'var_185']  shows zero correlation with the target variable. Thus dropping these feature.




train_data.drop(columns=col_reduced,inplace=True)
test_data.drop(columns=col_reduced,inplace=True)





train_data.shape


# ######   

# ### 13. Feature Scaling

# #### Standarization




scaler=StandardScaler()
scaler.fit(train_data.drop(columns='target').values)





num_cols=train_data.drop(columns='target').columns





train_data[num_cols]=scaler.transform(train_data.drop(columns='target'))
test_data[num_cols]=scaler.transform(test_data.drop(columns='ID_code'))





print('Train Data after Standardization:')
train_data.head()


# ######   

# ## Train, Validation Data Split




def split_data(data):
    x_train, x_test, y_train, y_test= train_test_split(data.drop(columns='target'),data.target,test_size=0.2,random_state=1,stratify=data.target)
    return x_train, x_test, y_train, y_test





x_train, x_test, y_train, y_test=split_data(train_data)





x_train.shape





x_test.shape





y_train.shape





y_test.shape


# ######   

# ## Modelling with selected features & standardized data




LR_model,RF_model, DT_model, NB_model=modelling(x_train, x_test, y_train, y_test)


# ###### Observations :
# On the basis of Precision, Recall & ROC_AUC score, the best performing model is Naive Bayes model.
# 
# ROC_AUC Score:  0.67
# 
# Precision Score:  0.71
# 
# Recall Score:  0.36 

# ###### The Precision & ROC_AUC score are quite average & Recall score is quite low, which means there is High False Negative Rate

# #### Since the data contains imbalanced target class, we are getting high false negative rate in our predictions. The above algorithms designed in a way that they favours the majority class in predictions. So to improve the predictions scores we have to deal with imbalanced situation.
# 
# ##### Trying two approaches to improve the results.
# 
# 1. Sampling Techniques. [ Over Sampling, Under Sampling, SMOTE Sampling]
# 2. Lightgbm Algorithm.

# ######   

# ##   Treating Imbalanced Dataset

# ### 1. Sampling techniques




transaction=train_data[train_data['target']==1]
no_transaction=train_data[train_data['target']==0]


# #### i. Over Sampling the lower class




transaction_oversampled = resample(transaction,
                          replace=True, # sample with replacement
                          n_samples=len(no_transaction), # match number in majority class
                          random_state=27) # reproducible results

oversampled=pd.concat([transaction_oversampled,no_transaction])
oversampled.target.value_counts()





x_train, x_test, y_train, y_test=split_data(oversampled)





NB_model=GaussianNB()
NB_model.fit(x_train,y_train)
roc_auc,precision,recall=acc_metrics(y_test,NB_model.predict(x_test))


# #### ii. Under Sampling the higher class




transaction_undersampled = resample(no_transaction,
                          replace=True, # sample with replacement
                          n_samples=len(transaction), # match number in majority class
                          random_state=27) # reproducible results

undersampled=pd.concat([transaction_undersampled,transaction])
undersampled.target.value_counts()





x_train, x_test, y_train, y_test=split_data(undersampled)





NB_model=GaussianNB()
NB_model.fit(x_train,y_train)
roc_auc,precision,recall=acc_metrics(y_test,NB_model.predict(x_test))


# #### iii. SMOTE Samplimg




x_train, x_test, y_train, y_test=split_data(train_data)





sm = SMOTE(random_state=1)
x_train, y_train = sm.fit_sample(x_train, y_train)





NB_model=GaussianNB()
NB_model.fit(x_train,y_train)
roc_auc,precision,recall=acc_metrics(y_test,NB_model.predict(x_test))


# ###### Observations :
# Over Sampling & Under Sampling performs equally, while SMOTE performs poorly. Selecting Over sampling with Naive Bayes Model for further processes




x_train, x_test, y_train, y_test=split_data(oversampled)





NB_model=GaussianNB()
NB_model.fit(x_train,y_train)
roc_auc,precision,recall=acc_metrics(y_test,NB_model.predict(x_test))


# ##### Naive Bayes Result:
#     ROC_AUC Score:  0.81
#     Precision Score:  0.81
#     Recall Score:  0.8 

# ### 2. Lightgbm




param = {
    'bagging_freq': 5,  'bagging_fraction': 0.5,  'boost_from_average':False,   
    'boost': 'gbdt',    'feature_fraction': 0.08, 'learning_rate': 0.01,
    'max_depth': -1,    'metric':'auc',             'min_data_in_leaf': 80,     
    'min_sum_hessian_in_leaf': 10.0,'num_leaves': 50,  'num_threads': 20,            
    'tree_learner': 'serial',   'objective': 'binary',       'verbosity': 1,
    'max_bin': 100, 'subsample_for_bin': 100, 'subsample': 1,
    'subsample_freq': 1, 'colsample_bytree': 0.8, 'min_split_gain': 0.45, 
    'min_child_weight': 1, 'min_child_samples': 5, 'is_unbalance':True,
}





training_data = lgb.Dataset(x_train, label=y_train)
validation_data = lgb.Dataset(x_test, label=y_test)

lgb_model=lgb.train(params=param,train_set=training_data,num_boost_round=10000,valid_sets=validation_data,verbose_eval=1000, early_stopping_rounds = 5000)





y_pred=lgb_model.predict(x_test)





roc_auc_score(y_test,y_pred)





pred=np.where(y_pred>=0.5,1,0)
acc_metrics(y_test,pred)


# ##### Lightgbm Result:
#     ROC_AUC Score:  0.95
#     Precision Score:  0.93
#     Recall Score:  0.98

# ###### Observations :
# On the basis of Precision, Recall & ROC_AUC Score, the best performing model is Lightgbm model.

# ######   

# ## Dumping Models




pickle.dump(lgb_model,open('Santander_Prediction_model.model','wb'))
pickle.dump(col_reduced,open('columns_to_drop.list','wb'))
pickle.dump(scaler,open('scaler.model','wb'))


# ######   

# ## Conclusion
# This was a classification problem on a typically unbalanced dataset with no missing values.
# Predictor variables are anonymous and numeric and target variable is categorical. Visualising
# descriptive features and finally I got to know that these variables are not correlated among
# themselves. After that I decided to treat imbalanced dataset and built different models with original
# data and chosen LightGBM as my final model with final value of AUC-Score is 0.95.

# ######   

# ## Test.csv prediction




# Importing Models

model=pickle.load(open('Santander_Prediction_model.model','rb'))
scaler=pickle.load(open('scaler.model','rb'))
columns_to_drop=pickle.load(open('columns_to_drop.list','rb'))





# Importing Dataset

test_data=pd.read_csv('test.csv')





# Dropping Columns

test_data.drop(columns=columns_to_drop,inplace=True,axis=1)





# Scaling the data

num_cols=test_data.drop(columns='ID_code').columns
test_data[num_cols]=scaler.transform(test_data.drop(columns='ID_code'))





# Predicting

pred=model.predict(test_data.drop(columns='ID_code'))
pred=np.where(pred>=0.5,1,0)





#Structuring data

test_data['target']=pred
test_data=test_data[['ID_code','target']]





# Submitting result in csv file

test_data.to_csv('predictions.csv',index=False)

