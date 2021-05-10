import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import numpy as np
import statsmodels.tsa.stattools as stat

data = web.DataReader('000001.ss','yahoo', dt.datetime(2015,1,1),dt.datetime(2021,5,1))
subdata = data.iloc[:-60,:4]
data.head()

for i in range(4):
    pvalue = stat.adfuller(subdata.values[:,i], 1)[1]
    print("指标 ",data.columns[i]," 单位根检验的p值为：",pvalue)
    
subdata_diff1 = subdata.iloc[1:,:].values - subdata.iloc[:-1,:].values
for i in range(4):
    pvalue = stat.adfuller(subdata_diff1[:,i], 1)[1]
    print("指标 ",data.columns[i]," 单位根检验的p值为：",pvalue)
    

kldata=data.values[:,[2,3,1,0]] # 分别对应开盘价、收盘价、最低价和最高价
from pyecharts import options as opts
from pyecharts.charts import Kline

kobj = Kline().add_xaxis(data.index[-60:].strftime("%Y-%m-%d").tolist()).add_yaxis("上证-日K线图",kldata[-60:].tolist())
kobj.render("shangzheng.html")
    
rows, cols = subdata_diff1.shape
aicList = []
lmList = []

for p in range(1,11):
    baseData = None
    for i in range(p,rows):
        tmp_list = list(subdata_diff1[i,:]) + list(subdata_diff1[i-p:i].flatten())
        if baseData is None:
            baseData = [tmp_list]
        else:
            baseData = np.r_[baseData, [tmp_list]]
    X = np.c_[[1]*baseData.shape[0],baseData[:,cols:]]
    Y = baseData[:,0:cols]
    coefMatrix = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),Y)
    aic = np.log(np.linalg.det(np.cov(Y - np.matmul(X,coefMatrix),rowvar=False))) + 2*(coefMatrix.shape[0]-1)**2*p/baseData.shape[0]
    aicList.append(aic)
    lmList.append(coefMatrix)
    
pd.DataFrame({"P":range(1,11),"AIC":aicList})

p = np.argmin(aicList)+1
n = rows
preddf = None
for i in range(60):
    predData = list(subdata_diff1[n+i-p:n+i].flatten())
    predVals = np.matmul([1]+predData,lmList[p-1])
    predVals=data.iloc[n+i,:].values[:4]+predVals
    if preddf is None:
        preddf = [predVals]
    else:
        preddf = np.r_[preddf, [predVals]]
    subdata_diff1 = np.r_[subdata_diff1, [data.iloc[n+i+1,:].values[:4] - data.iloc[n+i,:].values[:4]]]
#(np.abs(preddf-data.iloc[-30:data.shape[0],:4]/data.iloc[-30:data.shape[0],:4]).describe()

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(range(60),data.iloc[-60:data.shape[0],i].values,'o-',c='black')
    plt.plot(range(60),preddf[:,i],'o--',c='gray')
    plt.ylim(3000,4000)
    plt.ylabel("$"+data.columns[i]+"$")
plt.show()
v = 100*(1 - np.sum(np.abs(preddf - data.iloc[-60:data.shape[0],:4]).values)/np.sum(data.iloc[-60:data.shape[0],:4].values))
print("Evaluation on test data: accuracy = %0.2f%% \n" % v)

kobj1 = Kline().add_xaxis(data.index[-60:].strftime("%Y-%m-%d").tolist()).add_yaxis("上证-日K线图预测",preddf.tolist())
kobj1.render("shangzhengpred.html")