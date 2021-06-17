#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[53]:


df = sns.load_dataset("planets")
df.head(10)


# In[54]:


df.info()


# In[55]:


df.describe()


# In[56]:


df.shape


# In[57]:


df["orbital_period"].min()


# In[58]:


df["orbital_period"].max()


# In[59]:


df["orbital_period"].std()


# In[60]:


df["orbital_period"].idxmin() 


# In[61]:


df.describe().T


# In[62]:


df.isnull().sum()


# In[63]:


sns.displot(x="distance", data = df)


# In[64]:


df.dropna(axis=0, how='any')


# In[65]:


df.dropna(axis=1, how='any')


# In[66]:


df.corr()


# In[67]:


df.corr()  # -1 ile 1 arası değişir. 0 olursa korelasyon yok. - lerde ise negatif + larda ise pozitif korelasyon vardır denilebilir.
# 1 ve -1 e yakın ise güçlü bir korelasyon vardır. Metamatikteki doğru ve ters orantı mantığı gibi düşünülebilir.


# In[68]:


df[["mass","distance"]].corr()


# In[69]:


df["method"].value_counts() # Unique değerlerin sayısını getirir. None değer göstermez


# In[70]:


### Groupby


# In[71]:


df.groupby("method")["orbital_period"].mean()  # method grubunu orbital_period a göre grupla ve ortalamasını getir


# In[72]:


df.groupby("method")[["orbital_period"]].mean()   # iki tane [[]] yapınca dataframe olarak getirir ve görselliği daha güzeldir.


# In[73]:


df.year.unique() # unique degerleri getirir


# In[74]:


df.year.nunique() # unique değerlerin sayısını verir


# In[119]:


data = {'Company':['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
       'Person':['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
       'Sales':[200, 120, 340, 124, 243, 350]}


# In[120]:


df1 = pd.DataFrame(data)


# In[121]:


df1


# In[78]:


df1.groupby("Company")[["Sales"]].mean()


# In[79]:


df1.groupby("Company")[["Sales"]].std()


# In[80]:


df1.groupby("Company")[["Sales"]].describe().T[["GOOG"]]


# In[81]:


df1.groupby("Company")[["Sales"]].describe()


# In[82]:


df2 = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df2.head()


# In[83]:


df2.info()


# In[84]:


df2.col2.unique()


# In[85]:


df2.col2.nunique()


# In[86]:


df2.col2.value_counts()


# In[87]:


newdf=df2[(df2.col1 > 2) & (df2.col2 == 444)]
newdf


# In[88]:


df2["col2"].sum()


# In[89]:


del df2["col1"]


# In[90]:


df2


# In[91]:


df2.columns


# In[92]:


df2.index


# In[93]:


df2.sort_values(by = "col2")


# In[94]:


df2.sort_values(by = "col2", ascending = False) # default değeri True'dur


# In[95]:


df2.sort_values(by = "col3")


# In[96]:


df2.sort_values(by = "col2")


# In[97]:


df3 = pd.DataFrame({'col1':[1, 2, 3, np.nan],
                   'col2':[np.nan, 555, 666, 444],
                   'col3':['abc', 'def', 'ghi', 'xyz']})
df3.head()


# In[98]:


df3.dropna()


# In[99]:


df3.dropna(how = "any")


# In[100]:


df3


# In[101]:


df3.dropna(how = "all")


# In[102]:


df3


# In[103]:


df3.fillna("FILL") # eğer df3[...] diye indexleme yapmas isek komple dataframe i oraya ne yazarsan onunla doldurur.


# In[104]:


df3["col2"].fillna(df3.col2.mean())
# df3.col2.mean() col2 nin ortalamasıdır. None olanları bunun ile doldurabiliriz


# In[105]:


df3[['col2']].mean()


# In[106]:


df3[["col2","col1"]].fillna(df3.col2.mean())


# In[107]:


data = {'Company':['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
       'Person':['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
       'Sales':[200, 120, 340, 124, 243, 350]}


# In[ ]:





# In[108]:


my_map = {"GOOG": "GOO", "MSFT": "MIC", "FB":"FACE"}


# In[109]:


df1


# In[110]:


df1


# In[111]:


df1["Company"] =  df1["Company"].map(my_map) # şirket isimlerini değiştirmek için map() kullandık


# In[112]:


df1


# In[ ]:





# In[113]:


df1


# In[114]:


df1["Company"] = df1["Company"].replace(to_replace = "GOO", value = "GOOGLE") 


# In[115]:


df1


# In[116]:


df1.Sales.astype(float)


# In[123]:


df1['Sales']=df1['Sales'].astype('str')


# In[124]:


df1["Sales"] = df1["Sales"].map(lambda x: x[:3] + "K")


# In[125]:


df1


# In[126]:


df4 = pd.DataFrame({'groups': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'var1': [10,23,33,22,11,99],
                   'var2': [100,253,333,262,111,969]})
df4


# In[127]:


df4.groupby("groups").mean()


# In[128]:


df4.groupby("groups").mean().T


# In[131]:


df4.groupby("groups").aggregate([np.min,np.median,np.max]).T


# In[133]:


df4.groupby('groups').agg(['min','max']).T


# In[134]:


df4.groupby('groups').agg({"var1" : "min", "var2": "max"}).T


# In[135]:


df4.groupby('groups').agg({"var1" : "mean", "var2": "std"}).T


# In[136]:


### APPLY


# In[137]:


df4


# In[142]:


df4.apply(np.sum)


# In[143]:


def clarus(x):
    return x*2


# In[144]:


df4["var1"] = df4["var1"].apply(clarus)


# In[145]:


df4


# In[146]:


titanic = sns.load_dataset("titanic")


# In[147]:


titanic.head()


# In[149]:


titanic.groupby("sex")["survived"].mean()


# In[154]:


titanic.groupby(["class","sex"])["survived"].mean().T


# In[ ]:




