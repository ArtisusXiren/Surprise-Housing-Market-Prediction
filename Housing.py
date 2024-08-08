#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[2]:


import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error,mean_absolute_error
import joblib


# In[3]:


train_csv=pd.read_csv(r"C:\Users\ArtisusXiren\Desktop\Surprise_Housing\train.csv")


# In[4]:


train_1_csv=train_csv.iloc[:,1:17]
train_2_csv=train_csv.iloc[:,17:33]
train_3_csv=train_csv.iloc[:,33:49]
train_4_csv=train_csv.iloc[:,49:65]
train_5_csv=train_csv.iloc[:,65:81]


# In[5]:


train_1_csv=train_1_csv.drop(['LotFrontage','Alley'],axis=1)
null_values=train_1_csv.isnull()
null_values=null_values.sum()
null_values


# In[6]:


train_2_csv=train_2_csv.drop(['MasVnrType'],axis=1)
BsmtQual=train_2_csv['BsmtQual'].mode()[0]
train_2_csv['BsmtQual'].fillna(BsmtQual,inplace=True)
BsmtCond =train_2_csv['BsmtCond'].mode()[0]
train_2_csv['BsmtCond'].fillna(BsmtCond,inplace=True)
BsmtExposure=train_2_csv['BsmtExposure'].mode()[0]
train_2_csv['BsmtExposure'].fillna(BsmtExposure,inplace=True)
MasVnrArea=train_2_csv['MasVnrArea'].mode()[0]
train_2_csv['MasVnrArea'].fillna(MasVnrArea,inplace=True)
null_values=train_2_csv.isnull()
null_values=null_values.sum()
null_values


# In[7]:


BsmtFinType2=train_3_csv['BsmtFinType2'].mode()[0]
train_3_csv['BsmtFinType2'].fillna(BsmtFinType2,inplace=True)
Electrical=train_3_csv['Electrical'].mode()[0]
train_3_csv['Electrical'].fillna(Electrical,inplace=True)
BsmtFinType1=train_3_csv['BsmtFinType1'].mode()[0]
train_3_csv['BsmtFinType1'].fillna(BsmtFinType1,inplace=True)
null_values=train_3_csv.isnull()
null_values=null_values.sum()
null_values


# In[8]:


train_4_csv=train_4_csv.drop(['FireplaceQu'],axis=1)
GarageType=train_4_csv['GarageType'].mode()[0]
train_4_csv['GarageType'].fillna(GarageType,inplace=True)
GarageYrBlt=train_4_csv['GarageYrBlt'].mode()[0]
train_4_csv['GarageYrBlt'].fillna(GarageYrBlt,inplace=True)
GarageFinish=train_4_csv['GarageFinish'].mode()[0]
train_4_csv['GarageFinish'].fillna(GarageFinish,inplace=True)
GarageQual=train_4_csv['GarageQual'].mode()[0]
train_4_csv['GarageQual'].fillna(GarageQual,inplace=True)
GarageCond=train_4_csv['GarageCond'].mode()[0]
train_4_csv['GarageCond'].fillna(GarageCond,inplace=True)
null_values=train_4_csv.isnull()
null_values=null_values.sum()
null_values


# In[9]:


train_5_csv=train_5_csv.drop(['PoolQC','Fence','MiscFeature'],axis=1)
null_values=train_5_csv.isnull()
null_values=null_values.sum()
null_values


# In[10]:


categorical=train_1_csv.select_dtypes(include=['object']).columns
categorical_2=train_2_csv.select_dtypes(include=['object']).columns
categorical_3=train_3_csv.select_dtypes(include=['object']).columns
categorical_4=train_4_csv.select_dtypes(include=['object']).columns
categorical_5=train_5_csv.select_dtypes(include=['object']).columns
print(f"Number of categorical columns: {categorical}")


# In[11]:


print(f"Number of categorical columns_2: {categorical_2}")


# In[12]:


print(f"Number of categorical columns_5: {categorical_5}")


# In[13]:


print(f"Number of categorical columns_3: {categorical_3}")


# In[14]:


print(f"Number of categorical columns_4: {categorical_4}")


# In[15]:
mappings={}
def create_mappings(df,column):
    unique_values=df[column].unique()
    mapping={value:float(idx) for idx,value in enumerate(unique_values)}
    df[column]=df[column].map(mapping)
    print(mapping)
    return mapping

for df, name in zip([train_1_csv, train_2_csv, train_3_csv, train_4_csv, train_5_csv],['train_1_csv', 'train_2_csv', 'train_3_csv', 'train_4_csv', 'train_5_csv']):
    cat_columns=df.select_dtypes(include=['object']).columns
    for column in cat_columns:
        mappings[f'{name}_{column}']=create_mappings(df,column)
joblib.dump(mappings,'mappings.pkl')

# In[16]:


colummns=list(train_5_csv.columns[:-1])
attributes=[i for i in colummns]
X=train_5_csv[attributes].values
y=train_5_csv['SalePrice'].values
k_best=SelectKBest(score_func=f_regression,k=9)
x_new=k_best.fit_transform(X,y)
indices_x=k_best.get_support(indices=True)
results=[attributes[i] for i in indices_x]
features=pd.DataFrame(x_new,columns=results)
features


# In[17]:


features['SalePrice']=y
merge_df=pd.concat([train_4_csv,features],axis=1)
columns_new=list(merge_df.columns[:-1])
attributes_new=[i for i in columns_new]
X=merge_df[attributes_new].values
y=merge_df['SalePrice'].values
k_best=SelectKBest(score_func=f_regression,k=22)
x_new=k_best.fit_transform(X,y)
indices_x=k_best.get_support(indices=True)
results=[attributes_new[i] for i in indices_x]
features_new=pd.DataFrame(x_new,columns=results)
features_new


# In[18]:


features_new['SalePrice']=y
merge_2_df=pd.concat([train_3_csv,features_new],axis=1)
columns_2=list(merge_2_df.columns[:-1])
attributes_2=[i for i in columns_2]
X=merge_2_df[attributes_2].values
y=merge_2_df['SalePrice'].values
k_best=SelectKBest(score_func=f_regression,k=36)
x_new=k_best.fit_transform(X,y)
indices_x=k_best.get_support(indices=True)
results=[attributes_2[i] for i in indices_x]
features_3=pd.DataFrame(x_new,columns=results)
features_3


# In[19]:


features_3['SalePrice']=y
merge_3_df=pd.concat([train_2_csv,features_3],axis=1)
columns_3=list(merge_3_df.columns[:-1])
attributes_3=[i for i in columns_3]
X=merge_3_df[attributes_3].values
y=merge_3_df['SalePrice'].values
k_best=SelectKBest(score_func=f_regression,k=49)
x_new=k_best.fit_transform(X,y)
indices_x=k_best.get_support(indices=True)
results=[attributes_3[i] for i in indices_x]
features_4=pd.DataFrame(x_new,columns=results)
features_4


# In[20]:


features_4['SalePrice']=y
merge_4_df=pd.concat([train_1_csv,features_4],axis=1)
columns_4=list(merge_4_df.columns[:-1])
attributes_4=[i for i in columns_4]
X=merge_4_df[attributes_4].values
y=merge_4_df['SalePrice'].values
k_best=SelectKBest(score_func=f_regression,k=61)
x_new=k_best.fit_transform(X,y)
indices_x=k_best.get_support(indices=True)
results=[attributes_4[i] for i in indices_x]
features_5=pd.DataFrame(x_new,columns=results)
features_5


# In[21]:


features_5['SalePrice']=y
new_attributes=[i for i in features_5.columns[:-1]]
X=features_5[new_attributes].values
y=features_5['SalePrice'].values
k_best=SelectKBest(score_func=f_regression,k=57)
x_new=k_best.fit_transform(X,y)
indices_x=k_best.get_support(indices=True)
results=[new_attributes[i] for i in indices_x]
final_features_x=pd.DataFrame(x_new,columns=results)
final_features_x


# In[22]:


q1=final_features_x.quantile(0.25)
q2=final_features_x.quantile(0.75)
iqr=q2-q1
def identify(column):
    lower_bound=q1[column]-1.5*iqr[column]
    upper_bound=q2[column]+1.5*iqr[column]
    outliers=(final_features_x[column]<lower_bound) | (final_features_x[column]>upper_bound)
    return outliers
outlier_dict={i:identify(i) for i in final_features_x.columns}
column_outlier=[]
for name,value in outlier_dict.items():
    if value.sum()>70:
        column_outlier.append(name)
column_outlier


# In[23]:


final_features_7=final_features_x.drop(['MSSubClass',
 'MSZoning',
 'LandContour',
 'LotConfig',
 'LandSlope',
 'BldgType',
 'HouseStyle',
 'OverallCond',
 'RoofStyle',
 'MasVnrArea',
 'ExterCond',
 'BsmtCond',
 'BsmtExposure',
 'CentralAir',
 'Electrical',
 'Functional',
 'PavedDrive',
 'OpenPorchSF',
 'EnclosedPorch',
 'ScreenPorch',
 'SaleType',
 'SaleCondition'],axis=1)
final_features_7


# In[24]:


for i in final_features_7.columns:
     final_features_7[i] = np.log1p(final_features_7[i])
model_attributes=[i for i in final_features_7.columns]
print(model_attributes)
model_X=final_features_7[model_attributes].values
X_train,X_test,y_train,y_test=train_test_split(model_X,y,test_size=0.3,random_state=42)


# In[25]:


Model_reg=xgb.XGBRegressor(objective='reg:squarederror',reg_alpha=1,reg_lambda=1,colsample_bytree=0.3,learning_rate=0.1,max_depth=5,n_estimators=350)
Model_reg.fit(X_train,y_train)
y_pred=Model_reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse= mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
print("RMSE:", rmse)


# In[28]:


Model_random=RandomForestRegressor(n_estimators=400)
Model_random.fit(X_train,y_train)
y_pred_random=Model_random.predict(X_test)
mae_random = mean_absolute_error(y_test, y_pred_random)
mse_random= mean_squared_error(y_test, y_pred_random)
rmse=np.sqrt(mse_random)
print("RMSE:",rmse)


# In[ ]:
joblib.dump(Model_reg,'xgb_model.pkl')
joblib.dump(Model_random,'rf_model.pkl')





