#!/usr/bin/env python
# coding: utf-8

# In[1]:


# I imported numpy, pandas, matplotlib and seaborn libraries
# I called objects from each library individually
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# I have provided the dataset with this function
# I assigned it to the data variable. The dataset I imported
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv",sep=',',decimal='.')

# Looking at the first 5 data in the dataset 
data.head()


# In[4]:


# I saw that I have 21 attributes 
#I translated my attributes to Turkish for better analysis.
data.rename(columns={'customerID':'Müşteri ID','gender':'Cinsiyet','SeniorCitizen':'Yaşlı','Partner':'Evli',
                   'Dependents':'Ekonomik Bağımlı','tenure':'Abonelik Süresi','PhoneService':'Telefon Hizmeti',
                   'MultipleLines':'Birden Fazla Hat','InternetService':'İnternet Servisi',
                   'OnlineSecurity':'Çevrimiçi Güvenlik','OnlineBackup':'Çevrimiçi Yedekleme',
                   'DeviceProtection':'Cihaz Koruma','TechSupport':'Teknik Destek','StreamingTV':'Televizyon',
                   'StreamingMovies':'Film','Contract':'Sözleşme Süresi','PaperlessBilling':'Çevrimiçi Fatura',
                   'PaymentMethod':'Ödeme Yöntemi','MonthlyCharges':'Aylık Ödeme','TotalCharges':'Toplam Ödeme',
                   'Churn':'Müşteri Kaybı'}, inplace = True)


# In[7]:


data.head(8000)


# In[8]:



# While analyzing my dataset, I thought that I don't need 'Müşteri ID' at all and I removed it from my data variable.
data.drop('Müşteri ID', axis=1, inplace=True)


# In[9]:


# I changed the value of the 'No Loss of Customers' attribute of my dataset, made it understand better
data["Müşteri Kaybı"]= data["Müşteri Kaybı"].replace("No","Müşteri Kaybı Oluşmadı") 
data["Müşteri Kaybı"]= data["Müşteri Kaybı"].replace("Yes","Müşteri Kaybı Oluştu")


# In[10]:



# When examining my dataset, the value 0 or 1 for my 'old' attribute turned out to be meaningless. And I converted it to 'yes' or 'no' values.
# transfor the categori
data["Yaşlı"]= data["Yaşlı"].replace(0, "No") 
data["Yaşlı"]= data["Yaşlı"].replace(1, "Yes")


# In[11]:



# I wanted to see the data type of my attributes with the info function and called it.
 
# The data type of my 'Toplam Ödeme' attribute was supposed to be float64 rather than object.

data.info()


# In[29]:



# 'Toplam Ödeme' data type -> float64

data['Toplam Ödeme'] = pd.to_numeric(data['Toplam Ödeme'], errors='coerce')
data['Toplam Ödeme'] = data['Toplam Ödeme'].fillna(value=0)


# In[30]:


# Since I'm manipulating my 'old' attribute, I fixed the datatype to object just in case in [10]

data['Yaşlı'] = data['Yaşlı'].astype('object')


# In[31]:


# I looked at the numeric statistics of my numeric attributes. Using the describe function
data.describe()


# In[33]:


# 
# Using functions inside my sns object I saw a boxplot of my 'Müşteri Kaybı' attribute
sns.countplot(x = "Müşteri Kaybı", data = data)
data.loc[:, 'Müşteri Kaybı'].value_counts()


# Müşteri Kaybı Oluşmadı    5174
# Müşteri Kaybı Oluştu      1869


# In[34]:


# I checked to see if I had any data loss. And there was never

data.isnull().sum()


# In[35]:



#  In my dataset, my data was divided into two classes. These ; numerical and categorical

# But the target variable 'Müşteri Kaybı' was also categorical. I did not add this to the categorical part !!!!


Kategorik = data.select_dtypes(include='object').drop('Müşteri Kaybı', axis=1).columns.tolist()


Sayısal = data.select_dtypes(exclude='object').columns.tolist()


# I wanted to see it in categorical and numerical graphs to understand the dependence of my target variable on other variables.


# In[37]:


# Kategorik part ; 
for c in Kategorik:
    print('Öznitelik {} unique value: {}'.format(c, len(data[c].unique())))


# In[41]:


# It will be more efficient to work with the categorical part box plot.
plt.figure(figsize=(36,36))
for i,c in enumerate(Kategorik):
    plt.subplot(5,4,i+1)
    sns.countplot(data[c], hue=data['Müşteri Kaybı'])
    plt.title(c)
    plt.xlabel('')


# In[42]:



# It will be more efficient to see the numerical part with a line graph.

plt.figure(figsize=(20,5))
for i,c in enumerate(['Abonelik Süresi', 'Aylık Ödeme', 'Toplam Ödeme']):
    plt.subplot(1,3,i+1)
    sns.distplot(data[data['Müşteri Kaybı'] == 'Müşteri Kaybı Oluşmadı'][c], kde=True, color='blue', hist=False, kde_kws=dict(linewidth=2), label='Müşteri Kaybı Oluşmadı')
    sns.distplot(data[data['Müşteri Kaybı'] == 'Müşteri Kaybı Oluştu'][c], kde=True, color='Orange', hist=False, kde_kws=dict(linewidth=2), label='Müşteri Kaybı Oluştu')
    plt.title(c)


# In[43]:


data.head()


# In[44]:


data.info()


# In[45]:


#I used the box plot analysis you showed in the lecture to find our outliers.
import seaborn as sns

# This is 'Abonelik Süresi' outlier data graph

sns.boxplot(x = data['Abonelik Süresi'], y = data['Müşteri Kaybı'])


# In[46]:


import seaborn as sns

# This is 'Aylık Ödeme' outlier data graph

sns.boxplot(x = data['Aylık Ödeme'], y = data['Müşteri Kaybı'])


# In[47]:


import seaborn as sns

# This is 'Toplam Ödeme' outlier data graph
# Here we see that there is a lot of outlier data.
# The 'Toplam Ödeme' parameter actually gives us an info.

sns.boxplot(x = data['Toplam Ödeme'], y = data['Müşteri Kaybı'])


# In[48]:


from sklearn.preprocessing import LabelEncoder

# I used the sklearn library to clean up our outlier data.
# The data will become fully numeric with the labelencoder.
 
# Since there are contradictory data in the part where customer loss occurs, I will take action accordingly.

encoded = data.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)
encoded.head(8000)


# In[49]:


# 'Toplam Ödeme' outlier data cleared part 1


Müşteri_Kaybı_Yaşandı=encoded.loc[encoded['Müşteri Kaybı'].abs()>0]
Müşteri_Kaybı_Yaşandı


# In[50]:


# 'Toplam Ödeme' outlier data cleared part 2

Q1 = Müşteri_Kaybı_Yaşandı['Toplam Ödeme'].quantile(0.25)
Q3 = Müşteri_Kaybı_Yaşandı['Toplam Ödeme'].quantile(0.75)
IQR = Q3 - Q1
IQR


# In[51]:


Q=Q3+(1.5*IQR)
Q


# In[52]:


# 'Toplam Ödeme' outlier data cleared part 3
encoded_out = encoded[~((encoded['Toplam Ödeme'] < (Q3 + 1.5 * IQR)))&(encoded['Müşteri Kaybı']>0)]
encoded_out.head(8000)

# 109 


# In[56]:


# 'Toplam Ödeme' outlier data cleared part 4
encoded.drop(encoded[~((encoded['Toplam Ödeme'] < (Q3 + 1.5 * IQR)))&(encoded['Müşteri Kaybı']>0)].index, inplace=True)
encoded.head(8000)

 
# 109 data has outliers


# In[59]:


# 'Abonelik Süresi' outlier data cleared part 1
Q1_A = Müşteri_Kaybı_Yaşandı['Abonelik Süresi'].quantile(0.25)
Q3_A = Müşteri_Kaybı_Yaşandı['Abonelik Süresi'].quantile(0.75)
IQR_A = Q3_A - Q1_A
IQR_A


# In[60]:


# 'Abonelik Süresi' outlier data cleared part 2
Q_A=Q3_A+(1.5*IQR_A)
Q_A


# In[61]:


# 'Abonelik Süresi' outlier data cleared part 3
encoded_A_out = encoded[~((encoded['Abonelik Süresi'] < (Q3_A + 1.5 * IQR_A)))&(encoded['Müşteri Kaybı']>0)]
encoded_A_out.head(8000)


# In[62]:


# 'Abonelik Süresi' outlier data cleared part 4
encoded.drop(encoded[~((encoded['Abonelik Süresi'] < (Q3_A + 1.5 * IQR_A)))&(encoded['Müşteri Kaybı']>0)].index, inplace=True)
encoded.head(8000)


# I have removed all my outliers using formulas



# I have completed the preprocessing of my dataset on this site.


# In[ ]:


# TEST


# In[63]:


x = data.drop('Müşteri Kaybı', axis= 1)
y = data['Müşteri Kaybı']


# In[64]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.85, random_state = 400)

# That's how I defined the test ratios test_size = 0.85 and random_state = 400 for the most efficient result.


# In[65]:


x_test.head(8000)


# In[66]:


x_train.head(8000)


# In[67]:


y_test.head(8000)


# In[68]:


y_train.head(8000)


# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:





# In[ ]:




