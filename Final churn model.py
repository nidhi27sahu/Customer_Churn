#!/usr/bin/env python
# coding: utf-8

# Despite this intense competition, the telecommunications industry experiences more than 10% churn rate every year. Customer retention has now become even more important than customer acquisition, since acquiring a new customer costs 5-10 times more than retaining an existing one. Building Classification Model to find probability of customer churn with the help of given customer features and help telecom companies to reduce their churn rate for business profit.

# In[1]:


get_ipython().system('pip install ppscore')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import ppscore as ps
import plotly.express as pe

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[3]:


# reading the customer churn dataset

df = pd.read_csv(r'Churn.csv')
df.head()


# # EDA

# In[4]:


# dropping column name Unnamed:0 beacuse it is also referring to the index number or series of number 

df.drop(columns=['Unnamed: 0', 'state', 'area.code'], inplace=True)
df


# In[5]:


df.shape # we have data of 5000 customers based on 18 features


# In[6]:


df.rename(columns={'account.length':'account_length', 'voice.plan':'voice_plan',
                   'voice.messages':'voice_messages', 'intl.plan':'intl_plan', 'intl.mins': 'intl_mins',
                   'intl.calls': 'intl_calls', 'intl.charge':'intl_charge', 'day.mins':'day_mins',
                   'day.calls':'day_calls', 'day.charge':'day_charge', 'eve.mins':'eve_mins',
                   'eve.calls':'eve_calls', 'eve.charge':'eve_charge', 'night.mins': 'night_mins',
                   'customer.calls':'customer_calls', 'night.calls':'night_calls', 'night.charge':'night_charge'
}, inplace=True)


# In[7]:


df[df.duplicated()] # no duplicate values


# In[8]:


df.info()


# In[9]:


cat_col = df.select_dtypes(object)
cat_col

We found dtypes as object for 2 float column i.e. day.charge and eve.mins
To convert object to NaN value during conversion of dtypes to numeric we use error='corece

# In[10]:


df['day_charge'] = pd.to_numeric(df['day_charge'], errors='coerce')
df['eve_mins'] = pd.to_numeric(df['eve_mins'], errors='coerce')


# In[11]:


# Values

df.isnull().sum()


# In[12]:


sns.heatmap(df.isna())
plt.show()


# In[13]:


# percentage

null_val = df.isnull().sum()/len(df)*100
null_val

As percentage of null values are less than 5% compairing with the length of dataset we can drop the drop the Null values
# In[14]:


df.dropna(inplace=True) 


# In[15]:


df.info()


# In[16]:


df.describe()


# In[17]:


# including all columns i.e. including categorical column

df.describe(include=['O'])


# In[18]:


final_df = df.copy()     # for  finalized model building

# few temp object which will be used for feature scaling and engineering

vis_df = df.copy()       # for visualization
out_df= df.copy()        # for outliers detection
feature_df = df.copy()   # for feature selection


# # Visualization

# In[19]:


oe = OrdinalEncoder()
le = LabelEncoder()


# In[20]:


vis_df[['voice_plan', 'intl_plan']] = oe.fit_transform(vis_df[['voice_plan', 'intl_plan']])
vis_df.head(2)


# In[21]:


sns.set_style(style='darkgrid')


# In[22]:


#To get the Donut Plot to analyze churn
data = df['churn'].value_counts()
explode = (0, 0.2)
plt.pie(data, explode = explode,autopct='%1.1f%%',shadow=True,radius = 2.0, labels = ['Not churned customer','Churned customer'],colors=['royalblue' ,'lime'])
circle = plt.Circle( (0,0), 1, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('Donut Plot for Churn')
plt.show()

The total number of churned customer is 14.2 %
And the non churned customer is 85.8%
# In[23]:


plt.figure(figsize=(7,4))

[plt.subplot(1,2,1), sns.countplot(data=vis_df, x='churn')],
[plt.subplot(1,2,2), sns.countplot(data=vis_df, x='customer_calls', hue='churn')]
plt.grid()
plt.tight_layout()
plt.show()

-We noticed that problems of customer are resolved after 1,2 and 3 calls to customer care.
-Churn rate is higher if calls to custmer care increases above 4
# In[24]:


plt.figure(figsize=(7,4))

[plt.subplot(1,2,1), sns.countplot(data=vis_df, x='intl_plan', hue= 'churn')],
[plt.subplot(1,2,2), sns.countplot(data=vis_df, x='voice_plan', hue='churn')]
plt.grid()
plt.tight_layout()
plt.show()

-When the international plan is active, churn rate is higher. Usage of international plan is strong feature.
-We do not observe the same effect with voice mail plan
# In[25]:


plt.figure(figsize=(7,4))

sns.jointplot(data=vis_df,x='day_mins', y='night_mins', hue='churn')
plt.show()

We can see people who uses services at day for more than 225 mins are tends to churn more
# In[26]:


plt.rcParams['figure.figsize']= (10,5)

[plt.subplot(1,4,1), sns.histplot(data=vis_df, x='intl_charge', hue= 'churn')],
[plt.subplot(1,4,2), sns.histplot(data=vis_df, x='day_charge', hue= 'churn')],
[plt.subplot(1,4,3), sns.histplot(data=vis_df, x='eve_charge', hue= 'churn')],
[plt.subplot(1,4,4), sns.histplot(data=vis_df, x='night_charge', hue= 'churn')],
plt.tight_layout()
plt.show()

fig 1
-Most of the customer are paying Peso 3
-Customer paying Peso 2.5 to 3.5 of charges are more   tends to churn
fig 2
-customers paying above Peso 40 are churing more
fig 3
-customers paying peso 15 to 25 are going to churn
fig 4
-customer paying peso 7 to 12 are churing
# # Outlier Detection

# In[27]:


# Outlier detection of all columns using boxplot 
out_df[['voice_plan', 'intl_plan']] = oe.fit_transform(out_df[['voice_plan', 'intl_plan']])
plt.figure(figsize=(30,20))
sns.boxplot(data=out_df,orient='v')
plt.show()


# In[28]:


pe.box(out_df['night_mins'], orientation='v')


# In[29]:


iso = IsolationForest(random_state=1)
iso.fit(out_df.iloc[:,:-1])
outlier = iso.fit_predict(out_df.iloc[:,:-1])
outlier


# In[30]:


out_df['outliers']= outlier
out_df


# In[31]:


out_df[out_df['outliers']==-1]

--We have detected 571 outliers in the dataset, which is more than 10% of length od dataset.
-So we have to use capping
# # Capping outlier using IQR

# In[32]:


out_df.drop(columns=['outliers'], inplace=True)
def detect_outliers_iqr(data, column):
    q1 = data[column].quantile(0.25) 
    q3 = data[column].quantile(0.75)
    # print(q1, q3)
    IQR = q3-q1
    lower = q1-(1.5*IQR)
    upper = q3+(1.5*IQR)
    data[column] = np.where(data[column]>upper, upper, np.where(data[column]<lower, lower, data[column]))
# np.where(condition, true, false)


# In[33]:


out_df.columns


# In[34]:


detect_outliers_iqr(out_df, 'account_length' )
detect_outliers_iqr(out_df, 'voice_messages' )
detect_outliers_iqr(out_df, 'intl_mins' )
detect_outliers_iqr(out_df, 'intl_charge' )
detect_outliers_iqr(out_df, 'intl_calls' )
detect_outliers_iqr(out_df, 'day_mins' )
detect_outliers_iqr(out_df, 'day_calls' )
detect_outliers_iqr(out_df, 'day_charge' )
detect_outliers_iqr(out_df, 'eve_mins' )
detect_outliers_iqr(out_df, 'eve_calls' )
detect_outliers_iqr(out_df, 'eve_charge' )
detect_outliers_iqr(out_df, 'night_mins' )
detect_outliers_iqr(out_df, 'night_calls' )
detect_outliers_iqr(out_df, 'night_charge' )
detect_outliers_iqr(out_df, 'customer_calls' )


# In[35]:


sns.boxplot(data=out_df,orient='v')
plt.show()

All the outliers in the  columns has been removed
# # Feature Selection

# PP score

# In[36]:


ppscore = ps.matrix(out_df)
ppscore[ppscore['y']=='churn']


# In[37]:


out_df['churn'] = out_df['churn'].map({'no':0, 'yes':1})


# In[38]:


x= out_df.iloc[:,:-1]
y = out_df.iloc[:,-1]


# # Chi Square

# In[39]:


from sklearn.feature_selection import chi2, SelectKBest, RFE


# In[40]:


test = SelectKBest(score_func=chi2, k=5)
stats= test.fit(x,y)


# In[41]:


stats.scores_


# In[42]:


chidf = pd.DataFrame(stats.scores_).T
chidf.columns = x.columns
chidf

Type Markdown and LaTeX:  𝛼2
# # RFE

# In[43]:


lr = LogisticRegression()

rfe = RFE(lr, n_features_to_select=5)
rfe.fit(x,y)


# In[44]:


rfe.ranking_


# In[45]:


rfedf = pd.DataFrame(rfe.ranking_).T
rfedf.columns=x.columns
rfedf


# # Tree Based Model

# In[46]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')

dt.fit(x,y)


# In[47]:


dt.feature_importances_


# In[48]:


dtdf = pd.DataFrame(dt.feature_importances_).T
dtdf.columns = x.columns
dtdf

Chi2
-voice_message,intl_plan,day_mins,day_charge,eve_mins,night_mins,customer_calls

RFE

-voice_messages,voice_plan,intl_plan,intl_calls,day_mins,customer_calls

Tree

-customer_calls,day_mins,intl_plan,intl_mins,intl_charge,day_charge

PP SCORE

-day_mins, day_charge, customer_calls

Common Features

-intl_plan, day_mins, customer_calls,voice_message, day_charge
# In[49]:


out_df.head(3)


# In[50]:


x= out_df.loc[:,['voice_messages','intl_plan', 'day_mins', 'day_charge', 'customer_calls' ]]
y= out_df.iloc[:,-1]
y


# In[51]:


xtrain,xtest, ytrain, ytest = train_test_split(x,y, test_size=0.25, random_state=1)


# In[52]:


xtrain


# In[53]:


xtest


# In[54]:


ytrain


# In[55]:


ytest


# # Model Building

# In[56]:


models = []

models.append(('LG', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('DT', DecisionTreeClassifier()))


results = []

for name, model in models:
    kfold = KFold(n_splits=10)
    cv = cross_val_score(model, xtrain,ytrain, cv=kfold, scoring='accuracy')
    results.append(cv)
    print(f'Model: {name}---> Accuracy: {cv.mean()}')


# In[57]:


acc = pd.DataFrame(results).T
acc.columns=['LG', 'KNN', 'SVC', 'DT']
acc


# In[58]:


acc.boxplot()


# Checking the accuracy for the same models after using the standard scaler

# In[59]:


pipelines = []

pipelines.append(('Scaled_LR', Pipeline([('Scaler',StandardScaler()), ('model', LogisticRegression())])))
pipelines.append(('Scaled_KNN', Pipeline([('Scaler',StandardScaler()), ('model', KNeighborsClassifier())])))
pipelines.append(('Scaled_SVC', Pipeline([('Scaler',StandardScaler()), ('model', SVC())])))
pipelines.append(('Scaled_DT', Pipeline([('Scaler',StandardScaler()), ('model', DecisionTreeClassifier())])))


results = []

for name, model in pipelines:
    kfold = KFold(n_splits=10)
    cv = cross_val_score(model, xtrain,ytrain, cv=kfold, scoring='accuracy')
    results.append(cv)
    print(f'Model: {name}---> Accuracy: {cv.mean()}')


# In[60]:


acc = pd.DataFrame(results).T
acc.columns=['Scaled LG', 'Scaled KNN', 'Scaled SVC', 'Scaled DT']
acc


# In[61]:


acc.boxplot()

After applying Standard Scaler on the dataset, KNN and SVC is giving higher accuracy.
But SVC is much more consistent along with higher accuracy of 89.63%
# # Ensembled techniques

# In[62]:


ensemble = []
ensemble.append(('AB', AdaBoostClassifier()))
ensemble.append(('GM', GradientBoostingClassifier()))
ensemble.append(('RF', RandomForestClassifier()))

results = []

for name, model in ensemble:
    kfold = KFold(n_splits=10)
    cv = cross_val_score(model, xtrain,ytrain, cv=kfold, scoring='accuracy')
    results.append(cv)
    print(f'Model: {name} Accuracy: {cv.mean()}')


# In[63]:


acc = pd.DataFrame(results).T
acc.columns= ['Ada Boost', "Gradient Boost", 'Random Forest']
acc


# In[64]:


acc.boxplot()

Gradient boost have higher accuracy but not a consistent model.
Ensemble technique is not providing a consistent model, so I cannot choose any relevant ensembeled model.
Based on model consisteny and accuracy I'm selecting scaled SVC model for the customer churn prediction# 
# # Final Model Building

# SVC

# In[65]:


svm = SVC()
svm.fit(xtrain,ytrain)
ypred = svm.predict(xtest)

print(f"Training AC: {svm.score(xtrain, ytrain)}\nTesting AC: {svm.score(xtest, ytest)}")


# In[66]:


print(classification_report(ytest, ypred))


# # Hyper parameter tuning

# In[67]:


params = {
    'C' : [0.001, 0.01, 1, 10],
    'kernel' : [ 'rbf', 'sigmoid'],
    'gamma' : [0.001, 0.01, 1, 10]
    
}


# In[68]:


grid = GridSearchCV(SVC(), param_grid=params)


# In[69]:


grid.fit(xtrain, ytrain)


# In[70]:


grid.best_estimator_


# In[71]:


grid.best_score_


# In[72]:


svm = SVC(C=1, gamma=0.01)
svm.fit(xtrain,ytrain)
ypred = svm.predict(xtest)

print(f"Training AC: {svm.score(xtrain, ytrain)}\nTesting AC: {svm.score(xtest, ytest)}")


# In[73]:


print(classification_report(ytest, ypred))

-Here model accuracy is high but recall value is very low.
-Recall is very low because our dataset is not balanced, we have 84% of No class and rest of Yes class
# In[74]:


svm = SVC(C=10,class_weight='balanced' ,kernel='sigmoid', gamma=0.01)
svm.fit(xtrain,ytrain)
ypred = svm.predict(xtest)

print(f"Training AC: {svm.score(xtrain, ytrain)}\nTesting AC: {svm.score(xtest, ytest)}")


# In[75]:


print(classification_report(ytest, ypred))

Very low Accuracy
# # Unsampling- dataset is imbalanced

# In[76]:


get_ipython().system('pip install imblearn')


# In[77]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=1)

x_re,y_re = ros.fit_resample(x,y)


# In[78]:


x_re.shape, y_re.shape


# In[79]:


x_re


# In[80]:


x_re.boxplot()

After upsampling the dataset we have some outliers in voice_messages and intl_plan
We have to use capping method to treat outliers using the funtion we define before
# In[81]:


#detect_outliers_iqr(x_re, 'voice_messages') if we remove outliers from voice messages then all values turn 0


# In[82]:


x_re.boxplot()

We removed the outliers from our dataset, now we can build the model again
# In[83]:


x= x_re
y=y_re


# In[84]:


xtrain,xtest, ytrain, ytest = train_test_split(x,y, test_size=0.25, random_state=1)


# In[85]:


xtrain


# In[86]:


xtest


# In[87]:


ytrain


# In[88]:


ytest


# # Model Building after Over Sampling

# In[89]:


models = []

models.append(('LG', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('DT', DecisionTreeClassifier()))


results = []

for name, model in models:
    kfold = KFold(n_splits=10)
    cv = cross_val_score(model, xtrain,ytrain, cv=kfold, scoring='accuracy')
    results.append(cv)
    print(f'Model: {name}---> Accuracy: {cv.mean()}')


# In[90]:


acc = pd.DataFrame(results).T
acc.columns=['LG', 'KNN', 'SVC', 'DT']
acc


# In[91]:


acc.boxplot()
plt.show()

This 4 model is build on non-scaled data.
We have to check the accuracy after feature scaling.
# In[92]:


pipelines = []

pipelines.append(('Scaled_LR', Pipeline([('Scaler',StandardScaler()), ('model', LogisticRegression())])))
pipelines.append(('Scaled_KNN', Pipeline([('Scaler',StandardScaler()), ('model', KNeighborsClassifier())])))
pipelines.append(('Scaled_SVC', Pipeline([('Scaler',StandardScaler()), ('model', SVC())])))
pipelines.append(('Scaled_DT', Pipeline([('Scaler',StandardScaler()), ('model', DecisionTreeClassifier())])))


results = []

for name, model in pipelines:
    kfold = KFold(n_splits=10)
    cv = cross_val_score(model, xtrain,ytrain, cv=kfold, scoring='accuracy')
    results.append(cv)
    print(f'Model: {name}---> Accuracy: {cv.mean()}')


# In[93]:


acc = pd.DataFrame(results).T
acc.columns=['Scaled LG', 'Scaled KNN', 'Scaled SVC', 'Scaled DT']
acc


# In[94]:


acc.boxplot()

-Not a huge change has been noticed after scaling the data except SVC
-Accuracy of SVC has been increased by 18% after scaling.
-All 4 models are consistent in nature but KNN and DT has higher accuracy.
-we can select DT as our model with accuracy of 92.63%
# # Ensembeled Techniques

# In[95]:


ensemble = []
ensemble.append(('AB', AdaBoostClassifier()))
ensemble.append(('GM', GradientBoostingClassifier()))
ensemble.append(('RF', RandomForestClassifier()))

results = []

for name, model in ensemble:
    kfold = KFold(n_splits=10)
    cv = cross_val_score(model, xtrain,ytrain, cv=kfold, scoring='accuracy')
    results.append(cv)
    print(f'Model: {name}---> Accuracy: {cv.mean()}')
    


# In[96]:


acc= pd.DataFrame(results).T
acc.columns= ['AB', 'GB', 'RF']
acc


# In[97]:


acc.boxplot()
plt.show()

-Random Forest holds the highest accuracy of 92.57% and consistency
-I'm choosing Random Forest as the final model for the project and deployment.
-We can choose DT also but it is a combination of some decisions, whereas a random forest combines several decision trees
# In[98]:


x_re.head(2)


# # Building the finalized Model: Random Forest

# Converting 0 and 1 into yes and no format so that we can perform OnehotEncoding and Label Encoding while building the model

# In[99]:


df_new = pd.concat([x_re, y_re], axis=1)
df_new


# In[100]:


df_new['intl_plan'] = df_new['intl_plan'].map({0.0:'no', 1.0:'yes'})
df_new['churn'] = df_new['churn'].map({0.0:'no', 1.0:'yes'})
df_new


# In[101]:


transformer = ColumnTransformer(transformers=[
    ('tf1', OneHotEncoder(),[1]),
    ('tf2', StandardScaler(), [0,2,3,4]),   
    #('tf3', LabelEncoder(),[5])    
], remainder= 'passthrough')


# In[102]:


xtrain, xtest, ytrain, ytest = train_test_split(df_new.drop(columns=['churn']), df_new['churn'], test_size=0.25, random_state=1)


# In[103]:


xtrain.shape, ytrain.shape


# In[104]:


xtest.shape , ytest.shape


# In[105]:


rf = Pipeline(steps=[('transform', transformer), ('model', RandomForestClassifier())])
rf.fit(xtrain,ytrain)
ypred = rf.predict(xtest)
ypred


# In[106]:


print(classification_report(ytest, ypred))


# In[107]:


print(rf.score(xtrain,ytrain))
print(rf.score(xtest,ytest))


# In[108]:


hyper_df= df_new.copy()
hyper_df


# In[109]:


le= LabelEncoder()


# In[110]:


hyper_df['intl_plan'] = oe.fit_transform(hyper_df[['intl_plan']])
hyper_df['churn'] = le.fit_transform(hyper_df['churn'])


# In[111]:


hyper_df


# In[112]:


x = hyper_df.iloc[:,:-1]
y = hyper_df.iloc[:,-1]


# In[113]:


xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.25, random_state=1)


# In[114]:


params = {
    'n_estimators': [150,200,250,300],
    'max_depth' : [3,4,5],
    'criterion' : ["gini", "entropy", "log_loss"]
}


# In[115]:


grid = GridSearchCV(RandomForestClassifier(),param_grid= params)


# In[116]:


grid.fit(xtrain, ytrain)


# In[117]:


grid.best_estimator_


# In[118]:


grid.best_score_


# In[119]:


df_new


# In[120]:


df_new['customer_calls'] = df_new['customer_calls'].replace(3.5,4)


# In[121]:


df_new['voice_messages'] = df_new['voice_messages'].replace(42.5,43)


# In[122]:


x = df_new.iloc[:,:-1]
y = df_new.iloc[:,-1]


# In[123]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25, random_state=1)


# In[124]:


transformer = ColumnTransformer(transformers=[
    ('tf1',OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),[1]),
    ('tf2', StandardScaler(), [0,2,3,4]),   
    #('tf3', LabelEncoder(),[5])    
])


# In[125]:


rf = Pipeline(steps=[('transform', transformer), ('model', RandomForestClassifier(max_depth=5, n_estimators=200))])
rf.fit(xtrain,ytrain)
ypred = rf.predict(xtest)


# In[126]:


print(classification_report(ytest, ypred))


# In[127]:


print(rf.score(xtrain,ytrain))
print(rf.score(xtest,ytest))


# In[128]:


import pickle


# In[129]:


pickle.dump(rf, open('churn.pkl', 'wb')) 
#file saved by the name of churn.pkl


# In[130]:


df_new.describe()


# In[ ]:




