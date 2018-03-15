
# In[145]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[146]:

train_na = df_train.isnull().sum()
test_na = df_test.isnull().sum()
total_train_na = train_na.sort_values(ascending=False)
total_test_na = test_na.sort_values(ascending=False)
missing_data_train = pd.concat([total_train_na],axis=1,keys=['total_train_na'])
missing_data_test = pd.concat([total_test_na], axis=1, keys=['total_test_na'])


# In[147]:

## df_train에서 missing value에 대한 분석으로 나온 column들 위주로 각각 train과 test에서 지운다.
c = [col for col in df_test.columns if col in train_na[train_na>1]]
df_train = df_train.drop(c, 1)


# In[148]:

df_test = df_test.drop(c,1)


# In[149]:

df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df_test[col] = df_test[col].fillna(0)


# In[150]:

df_test["Functional"] = df_test["Functional"].fillna("Typ")


# In[151]:

for col in ('GarageArea', 'GarageCars'):
    df_test[col] = df_test[col].fillna(0)
df_test['SaleType'] = df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])
df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])
df_test['Exterior1st'] = df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])
df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])


# In[152]:

df_train['Electrical'] = df_train['Electrical'].fillna(df_train['Electrical'].mode()[0])


# In[153]:

df_test = df_test.drop('Utilities',1)
df_train = df_train.drop('Utilities',1)
#### -----------  여기까지 지우는 거 끝


# In[154]:

categorical_features = df_train.select_dtypes(include = ["object"]).columns
numerical_features = df_train.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
train_num = df_train[numerical_features]
train_cat = df_train[categorical_features]


# In[155]:

## log 변화 : 정규분포(numerical)
## train_cat은 할게 없음..
from scipy.stats import skew 
skewness = train_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)

skewness = skewness[abs(skewness)>0.5]
skewness.index
skew_features = df_train[skewness.index]
skew_features = np.log1p(skew_features)


# In[156]:

train_cat = pd.get_dummies(train_cat)


# In[181]:

train = pd.concat([train_cat,train_num],axis=1)
train.shape
train_ID = train['Id']
train.drop('Id',axis=1)


# In[167]:

df_train.SalePrice = np.log1p(df_train.SalePrice )
y = df_train.SalePrice


# In[170]:

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns


# In[171]:

train = pd.concat([train_cat,train_num],axis=1)
X_train,X_test,y_train,y_test = train_test_split(train,y,test_size = 0.3,random_state= 0)


# In[172]:

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[173]:

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_xgb.fit(X_train, y_train)



# In[174]:

xgb_train_pred = model_xgb.predict(X_train)
xgb_pred = np.expm1(model_xgb.predict(X_test))
print(rmsle(y_train, xgb_train_pred))



# In[212]:

sub = pd.DataFrame()
sub['SalePrice'] = xgb_pred
sub['Id'] = sub.index
sub = sub[['Id','SalePrice']]
sub.to_csv('submission.csv',index=False)

