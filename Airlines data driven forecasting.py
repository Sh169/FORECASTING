#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time


# In[4]:


df = pd.read_excel("Airlines+Data.xlsx")


# In[5]:


df


# In[6]:


df.rename(columns={"Passengers ('000)":"Passengers"},inplace=True)


# In[7]:


# Converting the normal index of Amtrak to time stamp 
df.index = pd.to_datetime(df.Month,format="%b-%y")


# In[8]:


df.Passengers.plot() # time series plot


# In[9]:


# Creating a Date column to store the actual Date format for the given Month column
df["Date"] = pd.to_datetime(df.Month,format="%b-%y")


# In[10]:


df["month"] = df.Date.dt.strftime("%b") # month extraction


# In[11]:


df["year"] = df.Date.dt.strftime("%Y") # year extraction


# In[12]:


# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=df,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")


# In[13]:


sns.boxplot(x="month",y="Passengers",data=df)


# In[14]:


sns.boxplot(x="year",y="Passengers",data=df)


# In[15]:


sns.catplot("month","Passengers",data=df,kind="box")


# In[16]:


# Line plot for Sales based on year  and for each month
sns.lineplot(x="year",y="Passengers",hue="month",data=df)


# In[17]:


# moving average for the time series to understand better about the trend character in df
df.Passengers.plot(label="org")


# In[18]:


df.shape


# In[19]:


for i in range(2,96,6):
    df["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)


# In[20]:


# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(df.Passengers,model="additive")
decompose_ts_add.plot()


# In[21]:


decompose_ts_mul = seasonal_decompose(df.Passengers,model="multiplicative")
decompose_ts_mul.plot()


# In[22]:


# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(df.Passengers,lags=10)


# In[23]:


tsa_plots.plot_pacf(df.Passengers)


# In[24]:


# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 
Train = df.head(80)
Test = df.tail(16)


# In[25]:


# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)


# In[26]:


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers)


# In[27]:


# Holt method 
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers)


# In[28]:


# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers)


# In[29]:


# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers)


# In[30]:


df

