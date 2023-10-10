#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('CVD_cleaned.csv')
print(df.head())


label_encoder = LabelEncoder()
df['Heart_Disease_Encoded'] = label_encoder.fit_transform(df['Heart_Disease'])


def convert_age_range(x):
    if '-' in x:
        return sum(map(int, x.split('-')))/2
    elif '+' in x:
        return int(x[:-1]) + 5  
    else:
        return int(x)

df['Age_Average'] = df['Age_Category'].apply(convert_age_range)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


sc = ax.scatter(df['Age_Average'], df['BMI'], df['Heart_Disease_Encoded'], c=df['Heart_Disease_Encoded'], cmap='coolwarm', marker='o')

ax.set_xlabel('Age_Average')
ax.set_ylabel('BMI')
ax.set_zlabel('Heart_Disease_Encoded')


cbar = fig.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Heart_Disease')

plt.title('3D Scatter Plot')
plt.show()


correlation_matrix = df.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




