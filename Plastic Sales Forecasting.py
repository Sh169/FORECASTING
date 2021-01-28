#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


plastic=pd.read_csv("PlasticSales.csv")
plastic.head()


# In[4]:


plastic.shape


# In[5]:


plastic.info()


# In[6]:


# PLotting the data
plastic.Sales.plot()


# In[7]:


plastic["Date"] = pd.to_datetime(plastic.Month,format="%b-%y")


# In[8]:


plastic["month"] = plastic.Date.dt.strftime("%b") # month extraction
plastic["year"] = plastic.Date.dt.strftime("%Y") # year extraction


# In[9]:


#Heatmap
plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=plastic,values="Sales",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values


# In[10]:


#Boxplot for ever
plt.figure(figsize=(8,6))
plt.subplot(211)
sns.boxplot(x="month",y="Sales",data=plastic)
plt.subplot(212)
sns.boxplot(x="year",y="Sales",data=plastic)


# In[14]:


#Preparing Dummy variables
month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
import numpy as np
p = plastic["Month"][0]
p[0:3]
plastic['months']= 0


# In[15]:


for i in range(60):
    p = plastic["Month"][i]
    plastic['months'][i]= p[0:3]


# In[16]:


month_dummies = pd.DataFrame(pd.get_dummies(plastic['months']))
plastic1 = pd.concat([plastic,month_dummies],axis = 1)


# In[17]:


t=np.arange(1,61)
plastic1['t']=t
t_square=plastic1['t']*plastic1['t']
plastic1['t_square']=t_square


# In[18]:


log_Sales=np.log(plastic1['Sales'])


# In[19]:


plastic1['log_Sales']=log_Sales


# In[20]:


# Lineplot
plt.figure(figsize=(12,3))
sns.lineplot(x="year",y="Sales",data=plastic)


# In[21]:


decompose_ts_add = seasonal_decompose(plastic.Sales,period=12)
decompose_ts_add.plot()
plt.show()


# In[22]:


#Splitting data
Train=plastic1.head(48)
Test=plastic1.tail(12)


# In[23]:


plastic1.Sales.plot()


# In[24]:


#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear


# In[25]:


#Exponential

Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[26]:


#Quadratic 

Quad = smf.ols('Sales~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


# In[27]:


#Additive seasonality 

add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[28]:


#Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Sales~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[29]:


##Multiplicative Seasonality

Mul_sea = smf.ols('log_Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[30]:


#Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


# In[31]:


#Compare the results 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# ##### Multiplicative Additive Seasonality has the least rmse value

# In[33]:


predict_data = pd.read_excel("New_PlasticSales.xlsx")


# In[34]:


predict_data


# In[35]:


#Build the model on entire data set
model_full = smf.ols('Sales~t+t_square',data=plastic1).fit()
pred_new  = pd.Series(model_full.predict(predict_data))
pred_new


# In[36]:


predict_data["forecasted_Sales"] = pd.Series(pred_new)
predict_data


# In[32]:


#### Here we  got the forecasted value for next 11 months along with t_square values

