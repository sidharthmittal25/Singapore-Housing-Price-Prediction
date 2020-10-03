#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
pd.options.display.float_format = '{:.4f}'.format
#arr_roi = []
import glob
import pmdarima as pm
from sklearn.metrics import mean_absolute_error


# In[5]:


path =r'F:\5106 GA\Private Raw data'
filenames = glob.glob(path + "/*.csv")
private_full = pd.DataFrame()
for filename in filenames:
    private_full = private_full.append([pd.read_csv(filename)])
private_full['month_year'] = pd.to_datetime(private_full['Sale Date']).dt.to_period('M')
private_monthly = private_full.groupby(['month_year']).agg(avg_transaction_price = ('Unit Price ($ psm)', 'mean'))
private_monthly.index = pd.date_range('2010-09','2020-10', freq = 'M')
private_monthly = private_monthly.loc[:'2020-08']
print(private_monthly)


# In[7]:


decom_private = seasonal_decompose(private_monthly['avg_transaction_price'])    
decom_private.plot() 


# In[14]:


train_private = private_monthly.loc[:'2017-08',:]
test_private = private_monthly.loc['2017-09':'2020-08',:]
model_train_p = pm.auto_arima(train_private['avg_transaction_price'], seasonal = True, m=12, suppress_warnings=True) # m=seasonal length
model_train_p.summary()


# In[15]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(model_train_p.resid(), lags=20)
plt.show()
plot_pacf(model_train_p.resid(), lags=20)
plt.show()


# In[68]:


pred_train_p, conf_train_p = model_train_p.predict(36, return_conf_int=True, alpha=0.05)
test_private_np = np.array(test_private['avg_transaction_price'])
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
mad_private = mean_absolute_error(test_private_np, pred_train_p)
mape_private = mape(test_private_np, pred_train_p)
print(mad_private)
print(mape_private)
print(conf_train_p)


# In[46]:


test_arima = pd.DataFrame()

test_arima['test_periods'] = pd.Series(pd.date_range('2017-09','2020-09', freq = 'M'))
test_arima['lower_bounds'] = [i[0] for i in conf_train_p]
test_arima['upper_bounds'] = [i[1] for i in conf_train_p]
test_arima['prediction'] = list(pred_train_p)

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(figsize=(12, 9))
ax1.plot_date(list(train_private.index), train_private, linestyle = 'solid', markeredgecolor = 'blue', markerfacecolor = 'blue', markersize = 3)
ax1.plot_date(test_arima['test_periods'], test_arima['prediction'], linestyle = 'solid', markeredgecolor = 'black', markerfacecolor = 'black', markersize = 3)
ax1.plot_date(test_arima['test_periods'], test_arima['upper_bounds'], linestyle = 'solid', markeredgecolor = 'orange', markerfacecolor = 'orange', markersize = 3)
ax1.plot_date(test_arima['test_periods'], test_arima['lower_bounds'], linestyle = 'solid', markeredgecolor = 'orange', markerfacecolor = 'orange', markersize = 3)
ax1.ticklabel_format(axis = 'y', style= 'sci',scilimits =(0,3))
ax1.set_xlabel('Year',fontsize=16)
ax1.set_ylabel('Avg. Unit Price ($ psm)',fontsize=16)
plt.tick_params(labelsize=18)
plt.show() 


# In[27]:


model_private = pm.auto_arima(private_monthly['avg_transaction_price'], seasonal = True, m=12, suppress_warnings=True) # m=seasonal length
model_private.summary()


# In[28]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(model_private.resid(), lags=20)
plt.show()
plot_pacf(model_private.resid(), lags=20)
plt.show()


# In[44]:


# Price prediction by postal district-D1
for i in range(28):
    i += 1
    dis = private_full[private_full['Postal District'] == i]
    try:
        dis_monthly = dis.groupby(['month_year']).agg(avg_transaction_price = ('Unit Price ($ psm)', 'mean'))
        dis_monthly = dis_monthly.loc[:'2020-08']
        model_dis = pm.auto_arima(dis_monthly, seasonal = True, m=12, suppress_warnings=True) # m=seasonal length
        predic_dis = model_dis.predict(36, return_conf_int=False, alpha=0.05)
        roi = ((predic_dis[-1] - dis_monthly.loc['2020-08'].values)/dis_monthly.loc['2020-08'].values)*100
        predic_dis = pd.Series(predic_dis)
        print('District {}'.format(i))
        print('ROI: {}'.format(roi))
        print(predic_dis)
        print()
    except ValueError:
        i += 1  


# In[58]:


# Price prediction by postal district-D1
for i in range(28):
    i += 1
    dis = private_full[private_full['Postal District'] == i]
    try:
        dis_monthly = dis.groupby(['month_year']).agg(avg_transaction_price = ('Unit Price ($ psm)', 'mean'))
        dis_monthly = dis_monthly.loc[:'2020-08']
        model_dis = pm.auto_arima(dis_monthly, seasonal = True, m=12, suppress_warnings=True) # m=seasonal length
        predic_dis = model_dis.predict(36, return_conf_int=False, alpha=0.05)
        roi = ((predic_dis[-1] - dis_monthly.loc['2020-08'].values)/dis_monthly.loc['2020-08'].values)*100
        predic_dis = pd.Series(predic_dis)
        print('District{} {}'.format(i,roi))
    except ValueError:
        i += 1 


# In[66]:


i=20
dis = private_full[private_full['Postal District'] == i]
dis_monthly = dis.groupby(['month_year']).agg(avg_transaction_price = ('Unit Price ($ psm)', 'mean'))
dis_monthly = dis_monthly.loc[:'2020-08']

d20_mean = dis_monthly['avg_transaction_price'].mean()
d20_std = dis_monthly['avg_transaction_price'].std()
d20_median = dis_monthly['avg_transaction_price'].median()
d20_cov = d20_std/d20_mean

print('mean: {}'.format(d20_mean))
print('std: {}'.format(d20_std))
print('median: {}'.format(d20_median))
print('Coefficient of Variation: {}'.format(d20_cov))


# In[ ]:




