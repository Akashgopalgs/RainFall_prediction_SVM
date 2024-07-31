#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,OrdinalEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix , recall_score , precision_score , f1_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')
c=(0.48366628618847957, 0.1286467902201389, 0.31317188565991266)


# In[2]:


df = pd.read_csv(r"C:\Users\akash\OneDrive\Documents\dataset\kaggle dataset\Rain_Austrailia.zip")

df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


nans=df.isna().sum().sort_values(ascending=False)
pct=nans*100/df.shape[0]
uniques=df.nunique()
noted=pd.concat([nans,pct,uniques,df.dtypes],axis=1)
noted.columns=['Null count','Null percentage','n_unique values','data_type']
noted


# In[6]:


print('number of rows in the dataset is ',df.shape[0],'.')
print('number of cols in the dataset is ',df.shape[1],'.')


# In[7]:


df.duplicated().sum()


# ### Handling missing values and out layers

# In[8]:


plt.figure(figsize=(10,6))
sns.barplot(x=noted.index,y=noted['Null percentage'],color=c)
plt.xticks(rotation=45)
plt.show()


# In[9]:


plt.figure(figsize=(10,6))
sns.barplot(x=noted.index,y=noted['n_unique values'],color=c)
plt.xticks(rotation=45)
plt.show()


# In[10]:


def impute_missing(df):
    loc_unique = df['Location'].unique()
    num_col = df.select_dtypes(exclude='object').columns
    cat_col = df.select_dtypes(include='object').columns

    for col in num_col:
        for loc in loc_unique:
            filt = df['Location'].isin([loc])
            med = df[filt][col].median()
            df.loc[filt, col] = df[filt][col].fillna(med)
    
    for col in cat_col:
        for loc in loc_unique:
            filt = df['Location'].isin([loc])
            if df[filt][col].empty:
                continue  # Skip to next location if empty
            mode = df[filt][col].mode()
            if not mode.empty:
                med = mode[0]
                df.loc[filt, col] = df[filt][col].fillna(med)
    return df


# In[11]:


df=impute_missing(df)


# In[12]:


remaining_nulls=df.isnull().sum().sort_values(ascending=False)


# In[13]:


plt.figure(figsize=(10,6))
sns.barplot(x=remaining_nulls.index,y=remaining_nulls.values,color=c)
plt.xticks(rotation=45)
plt.show()


# In[14]:


df.dropna(subset=['WindGustDir' , 'WindGustSpeed' , 'WindDir9am', 'WindDir3pm' , 'Pressure9am' , 'Pressure3pm' , 'RainToday' ,  'RainTomorrow',
                  'Evaporation','Sunshine', 'Cloud9am' , 'Cloud3pm']
                    , inplace=True  , axis= 0)


# In[15]:


df['Date'] = pd.to_datetime(df['Date'] )


# In[16]:


df['Day']=df['Date'].dt.day
df['Month']=df['Date'].dt.month
df['year']=df['Date'].dt.year


# In[17]:


df.drop('Date',axis=1,inplace=True)


# In[18]:


num_col = df.select_dtypes(exclude='object').columns
cat_col = df.select_dtypes(include='object').columns


# In[19]:


len(num_col)


# In[20]:


co=df['RainTomorrow'].value_counts()/df['RainTomorrow'].count()
sns.barplot(x=co.index,y=co.values,color=c)
plt.plot()


# In[21]:


fig,ax=plt.subplots(5,3,figsize=(20,35))
idx=0
for i in range(5):
    for j in range(3):
        sns.boxplot(ax=ax[i, j], x=df[num_col[idx]],color=c)
        ax[i, j].set_title(num_col[idx])
        idx=idx+1


# In[22]:


from scipy import stats
import numpy as np

def handle_outliers(df,impute_strategy='median'):
    num_col = df.select_dtypes(exclude='object').columns
    for col in num_col:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = np.where(z_scores > 2)[0]  

        if len(outliers) == 0:
            continue 
            
        if impute_strategy == 'median':
            imputed_value = df[col].median()
        elif impute_strategy == 'mean':
            imputed_value = df[col].mean()
            
        df.loc[outliers, col] = imputed_value

    return df


# In[23]:


fig,ax=plt.subplots(5,3,figsize=(20,35))
idx=0
for i in range(5):
    for j in range(3):
        sns.boxplot(ax=ax[i, j], x=df[num_col[idx]],color=c)
        ax[i, j].set_title(num_col[idx])
        idx=idx+1


# In[24]:


def handle_outlires_IQR(df):
    num_col = df.select_dtypes(exclude='object').columns
    for col in num_col:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        TQR=1.5*IQR
        outliers = df[ ( df[col] < (Q1 -IQR)) | (df[col] > (Q3 +IQR) ) ][col]
        med_value=df[col].median()
        df[df[col].isin([outliers])][col]=med_value
    return df


# In[25]:


df=handle_outlires_IQR(df)


# In[26]:


fig,ax=plt.subplots(5,3,figsize=(20,35))
idx=0
for i in range(5):
    for j in range(3):
        sns.kdeplot(ax=ax[i, j], x=df[num_col[idx]],color=c,alpha=0.4,fill=True)
        ax[i, j].set_title(num_col[idx])
        idx=idx+1


# In[27]:


num_pipeline=Pipeline(steps=[
    ('impute',SimpleImputer(strategy='mean')),
    ('scale',StandardScaler())
]
        
)
num_pipeline


# In[28]:


cat_pipeline=Pipeline( steps=[
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encoder',OrdinalEncoder())
])
cat_pipeline


# In[29]:


features=df.drop('RainTomorrow',axis=1)
labels=df['RainTomorrow']


# In[30]:


num_col = features.select_dtypes(exclude='object').columns
cat_col = features.select_dtypes(include='object').columns


# In[31]:


x_train,x_test,y_train,y_test =train_test_split(features,labels,test_size=0.30,random_state=42)


# In[32]:


col_transformer=ColumnTransformer(
    transformers=[('num_pipeline',num_pipeline,num_col)
                ,('cat_pipeline',cat_pipeline,cat_col)
                ]
    , remainder='passthrough',n_jobs=-1

)
col_transformer


# In[34]:


rf = DecisionTreeClassifier(random_state=42)


# In[45]:


log = LogisticRegression(random_state=42)


# In[54]:


svc = SVC(random_state=42)


# In[47]:


# pipe=make_pipeline(col_transformer,log)
# pipe


# In[36]:


# pipefinal=make_pipeline(col_transformer,rf)
# pipefinal


# In[55]:


pipe_linefinal=make_pipeline(col_transformer,svc)
pipe_linefinal


# ### Modeling and Evaluating

# In[48]:


# pipe.fit(x_train,y_train)


# In[50]:


# prd=pipe.predict(x_test)
# print('Accuracy Score :', accuracy_score(y_test, prd) , '\n')
#
# print('Classification Report :', '\n',classification_report(y_test, prd))


# In[37]:


# pipefinal.fit(x_train,y_train)


# In[62]:


# pred=pipefinal.predict(x_test)


# In[52]:


# print('Accuracy Score :', round(accuracy_score(y_test, pred)*100,2),"%" , '\n')
#
# print('Classification Report :', '\n',classification_report(y_test, pred))


# In[40]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
# y_pred=cross_val_predict(pipefinal,x_train,y_train,cv=3)
# cm=confusion_matrix(y_train,y_pred)
# cm


# In[59]:


pipe_linefinal.fit(x_train,y_train)


# In[63]:


prediction=pipe_linefinal.predict(x_test)


# In[64]:


print('Accuracy Score :', round(accuracy_score(y_test, prediction)*100,2),"%" , '\n')

print('Classification Report :', '\n',classification_report(y_test, prediction))


# *Accuracy of Suport vector classifier algorithm is 79.49 %* 

# In[ ]:




# Save the model to a file
import joblib

# Fit the final model
pipe_linefinal.fit(features, labels)

# Save the pipeline to a file
joblib.dump(pipe_linefinal, 'rain_prediction_model.pkl')


# from sklearn.compose import ColumnTransformer

# Example of creating a ColumnTransformer with n_jobs=1
# col_transformer = ColumnTransformer(
#     transformers=[
#         # Example transformers
#         ('num_pipeline', num_pipeline, num_col),
#         ('cat_pipeline', cat_pipeline, cat_col)
#     ],
#     remainder='passthrough',
#     n_jobs=1  # Disable parallel execution
# )

from sklearn.preprocessing import OrdinalEncoder

# Other code remains the same...

cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Rest of the code remains the same...
