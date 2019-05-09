#!/usr/bin/env python
# coding: utf-8

# # Used Cars Analysis and Prediction

# In[4]:


# Import Statements

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[6]:


df_v1 = pd.read_csv("cars.com.csv")


# In[7]:


df_v1.head()


# In[8]:


# Extracting year of manufacturing from the name column
df_v1['YEAR'] = df_v1['NAME'].str[:4]


# In[9]:


# Creating subtrings
df_v1['CNAME'] = df_v1['NAME'].str.split()


# In[10]:


df_v1.shape


# In[11]:


# Sperating manufacturing company from Name substrings
clist= []
for i in range(23655):
    clist.append(df_v1['CNAME'][i][1])
df_v1['COMPANY'] = clist


# In[12]:


#drop column CNAME, creating version 2
df_v2 = df_v1.drop(['CNAME','NAME'],axis=1)


# In[13]:


df_v2.head()


# In[14]:


df_v2.info()


# Dropping 3 rows with missing values

# In[15]:



df_v3 = df_v2.dropna()
df_v3.info()


# ## Exploratory Data Analysis 

# #### Our first hypothesis, miles driven is inversely correlated with price

# Numerical Data : Miles

# In[16]:


df_v3.MILES.hist() 


#  Need to apply some transformation as data 'MILES' is not distributed normally

# In[17]:


df_v3.MILES.apply(np.log).hist()


# In[18]:


df_v2.MILES.apply(np.sqrt).hist() 


# Best for OLS model, will apply this transformation when we finish our EDA.

# In[19]:


sns.lmplot(x = 'MILES', y= 'PRICE_USD',data = df_v3)


# Categorical variables

# In[20]:


df_v3['LOCATION'].value_counts()[:10].plot(kind='barh') # Top 10 location of cars sold


# In[21]:


df_v3['YEAR'].unique()


# In[22]:


df_v3['COMPANY'].unique()


# Creating a function to visualize The average price of each type of vehicle, for a given YEAR model

# In[23]:



def filt_year(year):
    plt.figure(figsize=(15,10))
    plt.xticks(rotation=90)
    df_v3_filtered=df_v3[(df_v3.YEAR.astype('int') == year)  ]
    sns.barplot(data=df_v3_filtered,x="TYPE",y="PRICE_USD",orient="H").set_title("Average price of Car Types of " + str( year)+ " models" )
    
year = 2017
filt_year(year)


# Creating a function to visualize The total number of vehicles for all companies for a given YEAR model

# In[24]:


def filt_year1(year1):
    plt.figure(figsize=(15,10))
    plt.xticks(rotation=90)
    df_v3_filtered=df_v3[(df_v3.YEAR.astype('int') == year1)  ]
    sns.countplot(df_v3_filtered['COMPANY'],orient = "h", order = df_v3_filtered['COMPANY'].value_counts().index).set_title("Number of Cars by Car Companies of " + str( year)+ " models" )

year1 = 2011
filt_year1(year1)


# ## Data Preprocessing
# 

# Transforming the numeric variable 'MILES' and normalizing it.

# In[25]:


df_v3.MILES =  df_v3.MILES.apply(np.sqrt)


# In[26]:


df_v3.MILES.head()


# Creating dummy variables for categorical variables!

# In[27]:


pd.get_dummies(df_v3).head().info()


# ### 133 columns!!! Too many. Lets use auto-encoders.

# In[28]:


df_v3.info()


# In[29]:


#import LabelEncoder
from sklearn.preprocessing import LabelEncoder
#convert column categories to numeric using Label Encoder
#Reference: http://pbpython.com/categorical-encoding.html
encoder = LabelEncoder()
df_v3["TYPE"] = encoder.fit_transform(df_v3["TYPE"])
df_v3["COMPANY"] = encoder.fit_transform(df_v3["COMPANY"])
df_v3["YEAR"] = encoder.fit_transform(df_v3["YEAR"])
df_v3["LOCATION"] = encoder.fit_transform(df_v3["LOCATION"])


# In[30]:


df_v3.head()


# ## Machine Learning

# In[31]:


df_v3['PRICE'] = df_v3.PRICE_USD


# In[32]:


df_v3 = df_v3.drop(['PRICE_USD'],axis=1)


# In[33]:


df_v3.head()


# Splitting into training and test data

# In[34]:


from sklearn.model_selection import train_test_split

X=df_v3.loc[:,"MILES":"COMPANY"]

Y = df_v3.loc[:,"PRICE"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[35]:


X_train.head()

shuffle_index = np.random.permutation(18921)

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
# ### K-Nearest Neighbors
# 

# In[36]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
knn =  Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor())
    ])
knn.fit(X_train, Y_train)

from sklearn.model_selection import cross_val_score
print("R Square:" , cross_val_score(knn, X_train, Y_train, scoring='r2',cv=5))

scores_mse = cross_val_score(knn, X, Y, scoring="neg_mean_squared_error", cv=5)
        
rmse_scores = np.sqrt(-scores_mse)
#Print  RMSE Score
print("RMSE Scores:", rmse_scores)
print("Mean RMSE Score:" , rmse_scores.mean())


# ## Linear Regression 

# In[37]:



from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression

linear = Pipeline([
        ("scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])
linear.fit(X_train, Y_train)


from sklearn.model_selection import cross_val_score
print("R Square:" , cross_val_score(linear, X_train, Y_train, scoring='r2',cv=5))

scores_mse = cross_val_score(linear, X, Y, scoring="neg_mean_squared_error", cv=5)
        
rmse_scores = np.sqrt(-scores_mse)
#Print  RMSE Score
print("RMSE Scores:", rmse_scores)
print("Mean RMSE Score:" , rmse_scores.mean())


# # Support Vector Regressor

# In[38]:


from sklearn.svm import LinearSVR

svm_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_reg", LinearSVR(epsilon = 1.5, C=5))
    ])
svm_reg.fit(X_train, Y_train)

from sklearn.model_selection import cross_val_score
cross_val_score(svm_reg, X_train, Y_train, scoring='r2',cv=5)


# Lets try increasing the C hyper paramter.

# In[39]:


from sklearn.svm import LinearSVR

svm_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_reg", LinearSVR(epsilon = 1.5, C=300))
    ])
svm_reg.fit(X_train, Y_train)

from sklearn.model_selection import cross_val_score
cross_val_score(svm_reg, X_train, Y_train, scoring='r2',cv=5)


# # Decision Tree
# 
# 

# In[40]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
dtree = DecisionTreeRegressor()
param_grid = { "criterion" : ["mse"]
              , "min_samples_leaf" : [10,12,14]
              , "min_samples_split" : [3,5]
              , "max_depth": [4,5,7]
              }
gs_tree = GridSearchCV(estimator=dtree, param_grid=param_grid, cv=3, n_jobs=1, verbose=1)
gs_tree = gs_tree.fit(X_train, Y_train)


# In[41]:


cvres = gs_tree.cv_results_
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(mean_score,params)


# In[42]:


print(gs_tree.best_score_)
print(gs_tree.best_params_)


# In[43]:


bp = gs_tree.best_params_
dtree = DecisionTreeRegressor(criterion=bp['criterion'],
                              min_samples_leaf=bp['min_samples_leaf'],
                              min_samples_split=bp['min_samples_split'],
                              max_depth=bp['max_depth'],
                              )
dtree.fit(X_train, Y_train)
# Explained variance score: 1 is perfect prediction
print('Score: %.2f' % dtree.score(X_test, Y_test))


# In[44]:


import graphviz
from sklearn import tree


# In[47]:


dot_data = tree.export_graphviz(dtree,out_file=None)
graph = graphviz.Source(dot_data)
graph.render('cars_tree')


# ## Random Forest Regressor
# 

# We'll train a random forest regressor and use Grid Search to optimize the hyperparameter values.

# In[48]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV  

rf = RandomForestRegressor()
param_grid = { "criterion" : ["mse"]
              , "min_samples_leaf" : [10,12,14]
              , "min_samples_split" : [3,5]
              , "max_depth": [2,4,5,7]
              , "n_estimators": [10,100]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=1, verbose=1) # Used K-Fold Cross Validation
gs = gs.fit(X_train, Y_train)


# In[39]:


cvres = gs.cv_results_
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(mean_score,params)


# In[49]:


print(gs.best_score_)
print(gs.best_params_)


# In[50]:


bp = gs.best_params_
forest = RandomForestRegressor(criterion=bp['criterion'],
                              min_samples_leaf=bp['min_samples_leaf'],
                              min_samples_split=bp['min_samples_split'],
                              max_depth=bp['max_depth'],
                              n_estimators=bp['n_estimators'])
forest.fit(X_train, Y_train)
# Explained variance score: 1 is perfect prediction
print('Score: %.2f' % forest.score(X_test, Y_test))


# Accuracy of 96% !!! That's really good I feeel! But its definitely overfitting, so we need to change the list of values of hyperparameters accordingly.

# In[51]:


predictions = forest.predict(X_test)


# In[52]:


#Creating a Data frame to compare the Actual v/s predicted prices
df_results = pd.DataFrame({'Actual Price':Y_test, 'Predicted': predictions })
df_results.head()


# So, lets not include the value 7 for max_depth, i.e 5 can be the highest value of max_depth.

# In[53]:


from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
param_grid = { "criterion" : ["mse"]
             , "min_samples_leaf" : [10,12,14]
             , "min_samples_split" : [3,5]
             , "max_depth": [2,4,5]
             , "n_estimators": [10,100]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=1, verbose=1) # Used K-Fold Cross Validation
gs = gs.fit(X_train, Y_train)


# In[54]:


print(gs.best_score_)
print(gs.best_params_)


# In[55]:


bp = gs.best_params_
forest = RandomForestRegressor(criterion=bp['criterion'],
                              min_samples_leaf=bp['min_samples_leaf'],
                              min_samples_split=bp['min_samples_split'],
                              max_depth=bp['max_depth'],
                              n_estimators=bp['n_estimators'])
forest.fit(X_train, Y_train)
# Explained variance score: 1 is perfect prediction
print('Score: %.2f' % forest.score(X_test, Y_test))


# We have regularized our model and the accuracy is 87. This model will generalize better 

# In[56]:


predictions = forest.predict(X_test)


# In[57]:


#Creating a Data frame to compare the Actual v/s predicted prices
df_results = pd.DataFrame({'Actual Price':Y_test, 'Predicted': predictions })
df_results.head()


# In[ ]:





# ## Ensembles of all the predictors

# In[60]:


from sklearn.ensemble import VotingClassifier
voting_clfr = VotingClassifier(estimators = [('rf',forest),('lin_reg',linear),('knn',knn),('svm',svm_reg)],
                               voting='hard')


# In[71]:


from sklearn.metrics import mean_squared_error
for clfr in [forest,linear,knn,svm_reg]:
    clfr.fit(X_train,Y_train)
    y_pred = clfr.predict(X_test)
    print(clfr.__class__.__name__ , mean_squared_error(Y_test, y_pred))


# # Gradient Boosting

# In[41]:


from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=100,max_depth = 4,learning_rate=0.1)
gb.fit(X_train,Y_train)
score(X_test,Y_test)


# In[44]:


#Code in this cell has been referred from link https://www.kaggle.com/ddmngml/trying-to-predict-used-car-value
#storing feature importances in a variable
feature_importances = forest.feature_importances_

#sorting importances
indices = np.argsort(feature_importances)[::-1]

#calculating standard deviation of feature importances
std = np.std([f.feature_importances_ for f in forest.estimators_],
             axis=0)  

for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X_train.columns.values[indices[f]], feature_importances[indices[f]]))

# Plot the feature importances of the forest regressor
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), feature_importances[indices],
       color="r", yerr=std[indices], align="center",tick_label = X_train.columns.values)
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# ### We can see that YEAR, TYPE and MILES travelled were the most important features in our model to predict the Price of the car! This should be taken into consideration when choosing a good deal! 
