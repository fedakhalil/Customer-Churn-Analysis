# In[1]:


# importing library

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings("ignore")


# In[2]:


data = pd.read_csv("churn.csv")
df = data.copy()
df.head()


# In[3]:


df.info()


# In[ ]:





# # Data Preprocessing

# * TotalCharges column type is not correct

# In[4]:


df["TotalCharges"].describe()


# In[5]:


# convert column arguments to float 

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# drop NaN values which converted from strings 

df.dropna(inplace = True)


# In[6]:


df.info()


# In[7]:


# index resetting

df.reset_index(drop = True, inplace = True)


# In[8]:


df.head()


# In[9]:


df.info()


# In[10]:


# save customer id

cus_id = df["CustomerID"]


# In[11]:


cus_id


# In[12]:


df = df.drop("CustomerID", axis = 1)


# In[13]:


df.head()


# In[14]:


# get unique values in each column

df.select_dtypes("object").apply(lambda x: x.unique())


# In[15]:


# manual labelling

change = {"No" : 0, "Yes" : 1, "Female" : 0, "Male" : 1, "No internet service" : 2, "No phone service" : 2,
         "DSL" : 1, "Fiber optic" : 2}


# In[16]:


df = df.replace(change)


# In[17]:


df.info()


# In[18]:


df


# In[19]:


# import LabelEncoder to get label of rest of columns

from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()


# In[20]:


df.select_dtypes("object")


# In[21]:


# labelling

df[["Contract","PaymentMethod"]] = df[["Contract","PaymentMethod"]].apply(lambda x: lbe.fit_transform(x))


# In[22]:


df


# In[23]:


df.describe().T


# In[ ]:





# # Modelling

# In[24]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV


# In[25]:


X = df.drop("Churn", axis = 1)
y = df["Churn"]


# In[26]:


# split train and test

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 42)


# In[27]:


# scale data to get better model performance

from sklearn.preprocessing import MinMaxScaler


# In[28]:


scaler = MinMaxScaler().fit(X_train)


# In[29]:


X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[30]:


y_test.value_counts()


# In[31]:


# check some statistics

import statsmodels.api as sm


# In[32]:


sm_model = sm.OLS(y_train, X_train_scaled).fit()
sm_model.summary()


# In[33]:


# import machine learning models to analysis

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[34]:


log_model = LogisticRegression()
bayes_model = GaussianNB()
svr_model = SVC()
rf_model = RandomForestClassifier()
gbm_model = GradientBoostingClassifier()
xgb_model = XGBClassifier()


# ## Models Performance Testing

# In[35]:


models = [log_model, bayes_model, svr_model, rf_model, gbm_model, xgb_model]


# In[36]:


# checking each model performance

def model_testing(model_list):
    for name in model_list:
        global model_fit
        model_fit = name.fit(X_train_scaled, y_train)
        train_score = accuracy_score(y_train, model_fit.predict(X_train_scaled))
        test_score = accuracy_score(y_test, model_fit.predict(X_test_scaled))
        val_score = cross_val_score(name, X_test_scaled, y_test, cv = 10, scoring = "accuracy").mean()
        conf_matrix = confusion_matrix(y_test, model_fit.predict(X_test_scaled))
        print("\n{}".format(name.__class__.__name__), "Train Score : ", train_score)
        print("{}".format(name.__class__.__name__), "Test Score : ", test_score)
        print("{}".format(name.__class__.__name__), "Test Matrix : ", str(conf_matrix).split())
        print("{}".format(name.__class__.__name__), "Validation Score : ", val_score), "\n"
        print(classification_report(y_test, model_fit.predict(X_test_scaled)))


# In[37]:


model_testing(models)


# In[ ]:





# * It might be say that Logistic Regression and Gradient Boosting are better performance

# In[ ]:





# # Models Hyperparameter Tuning

# ## Logistic Regression

# In[38]:


log_model = LogisticRegression()


# In[39]:


log_params = {"C" : [0.01, 0.1, 1, 5, 10, 30],
             "max_iter" : [100, 300, 500, 1000]}


# In[40]:


log_grid = GridSearchCV(log_model, log_params, cv = 10).fit(X_train_scaled, y_train)

log_grid.best_params_


# * result didn`t change

# In[ ]:





# ## Gradient Boosting

# In[41]:


gbm_model = GradientBoostingClassifier()
gbm_params = {"n_estimators" : [100,300,500],
            "min_samples_split" : [2, 5, 8],
            "min_samples_leaf" : [1, 2, 4],
            "learning_rate" : [0.01, 0.1],
             "max_depth" : [3,10]}

gbm_grid =  GridSearchCV(gbm_model, gbm_params, cv = 5, n_jobs = -1, verbose = 2).fit(X_train_scaled, y_train)


# In[42]:


# best model parameters

gbm_grid.best_params_


# In[43]:


gbm_model = GradientBoostingClassifier(min_samples_leaf=2, min_samples_split=5).fit(X_train_scaled, y_train)


# In[44]:


accuracy_score(y_test, gbm_model.predict(X_test_scaled))


# In[45]:


confusion_matrix(y_test, gbm_model.predict(X_test_scaled))


# In[46]:


print(classification_report(y_test, gbm_model.predict(X_test_scaled)))


# In[ ]:





# In[47]:


# predicted values

y_pred = gbm_model.predict(X_test_scaled)
y_pred


# In[48]:


# predicted positive values

y_pred_proba = gbm_model.predict_proba(X_test_scaled)[:,1]
y_pred_proba


# In[50]:


# create new dataframe from predicted values

proba_data = pd.DataFrame(data = {"True" : y_test, "Pred" : y_pred_proba})
proba_data


# In[51]:


cus_id


# In[52]:


# combine customer_id and predicted values

churn_data = pd.concat([cus_id, proba_data], axis = 1)
churn_data


# In[53]:


churn_data.dropna(inplace = True)
churn_data.drop("True", axis = 1, inplace = True)


# In[54]:


churn_data.reset_index(drop = True, inplace = True)


# In[55]:


churn_data


# In[56]:


# create churn probability column

def proba_name(x):
    if x < 0.40:
        x = "Low"
    elif x >= 0.40 and x < 0.70:
        x = "Medium"
    else:
        x = "High"
    return x


# In[57]:


churn_data["Pred"] = churn_data["Pred"].apply(proba_name)


# In[58]:


churn_data.rename(columns={"Pred" : "Churn_Probability"}, inplace = True)


# In[59]:


churn_data.head(10)


# In[60]:





# In[61]:


churn_data["Churn_Probability"].value_counts()


# In[62]:


# visualization

plt.figure(figsize = (8,6))
plt.xticks(fontsize = 10, color = "r", fontweight = "bold")
plt.yticks(fontsize = 10, color = "b", fontweight = "bold")
churn_data["Churn_Probability"].value_counts().plot.bar();

