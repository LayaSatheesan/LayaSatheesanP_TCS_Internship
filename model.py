
#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,accuracy_score,confusion_matrix,f1_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
# Import sys and warnings to ignore warning messages 
import sys
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# In[2]:


data=pd.read_csv(r'C:\Users\Dell\Downloads\salarydata.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# ### Understanding the dataset

# In[5]:


data.shape


# In[6]:


data.info()


# ### Data Cleaning

# ### 1. Missing Values

# In[7]:


#check the missing value
data.isnull().sum()


# **Above sum shows there are no null values in the dataset.**

# In[8]:


#we can see that there are some special characters in the data like ‘?’.
#Finding the special characters in the data frame
data.isin(['?']).sum(axis=0)


# In[9]:


#Handling missing values
# the code will replace the special character to nan  
data['native-country'] = data['native-country'].replace('?',np.nan)
data['workclass'] = data['workclass'].replace('?',np.nan)
data['occupation'] = data['occupation'].replace('?',np.nan)


# In[10]:


data.isnull().sum()


# In[11]:


#we will use the pandas DataFrame mode() method to fill the missing value.
data = data.fillna(data.mode().iloc[0])


# In[12]:


data.isnull().sum()


# ### 2. Remove duplicate data 

# In[13]:


#Checking for duplicated entries
sum(data.duplicated(data.columns))


# In[14]:


#Delete the duplicates and check that it worked
data = data.drop_duplicates(data.columns, keep='last')
sum(data.duplicated(data.columns))


# In[15]:


data.shape


# In[16]:


data.columns


# ### 3. Handling Outliers

# In[17]:


## checking outliers
for i in ['age',
       'capital-gain','capital-loss','hours-per-week'] :
    plt.title(i)
    sns.boxplot(data=data[i])
    plt.show()  


# ### Handling Outliers with age

# In[18]:


q1 = np.percentile(data['age'],25,interpolation='midpoint')
q3 = np.percentile(data['age'],75,interpolation='midpoint')

IQR = q3-q1
low_limit=q1-1.5*IQR
high_limit=q3+1.5*IQR

index=data['age'][(data['age']<low_limit)|(data['age']>high_limit)].index
data.drop(index,inplace=True)


# ### Handling Outliers with capital_gain

# In[19]:


q1 = np.percentile(data['hours-per-week'],25,interpolation='midpoint')
q3 = np.percentile(data['hours-per-week'],75,interpolation='midpoint')

IQR = q3-q1
low_limit=q1-1.5*IQR
high_limit=q3+1.5*IQR

index=data['hours-per-week'][(data['hours-per-week']<low_limit)|(data['hours-per-week']>high_limit)].index
data.drop(index,inplace=True)


# ### 5. Exploratory Data analysis

# In[20]:


sns.countplot(x=data["salary"])
plt.title("Countplot of salary")
plt.xlabel("salary")
plt.ylabel("Count")
plt.show()


# Most of the people got salary less than or equal to 50k

# In[21]:


sns.countplot(x=data["sex"])
plt.title("Countplot of sex")
plt.xlabel("sex")
plt.ylabel("Count")
plt.show()


# In[22]:


data["sex"].value_counts(normalize=True)


# In this dataset male count is more than female. Arount 67% are males.Others are females.

# In[23]:


#Checking race

data['race'].value_counts().plot(kind = 'pie')
plt.legend()
plt.rcParams['figure.figsize'] = (10,10)
plt.show()


# Most of the are from white race

# In[24]:


sns.countplot(x=data["relationship"])
plt.title("Countplot of relationship")
plt.xlabel("Relationship")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()


# Above 12k people in this dataset having a relationship as husband.More than 8k people under the category of not-in-family.

# In[25]:


#Checking race

data['occupation'].value_counts().plot(kind = 'pie')
plt.legend()
plt.rcParams['figure.figsize'] = (15,10)
plt.show()


# In[26]:


sns.countplot(x=data["marital-status"])
plt.title("Countplot of marital-status")
plt.xlabel("Marital-status")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()


# In[27]:


sns.countplot(x=data["education"])
plt.title("Countplot of education")
plt.xlabel("Education")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()


# In[28]:


sns.countplot(x=data["workclass"])
plt.title("Countplot of workclass")
plt.xlabel("Workclass")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()


# In[29]:


sns.lineplot(x=data['salary'],y=data['age'],marker='o')
plt.title('Lineplot of salary Vs age')
plt.show()


# In[30]:


plt.figure(figsize=(8,6))
sns.histplot(data["age"],binwidth=5)
plt.title("Distribution of age")
plt.show()


# In[31]:


plt.figure(figsize=(8,6))
sns.histplot(data["education-num"],binwidth=5)
plt.title("Distribution of education-num")
plt.show()


# In[32]:


plt.figure(figsize=(8,6))
sns.histplot(data["capital-gain"])
plt.title("Distribution of capital-gain")
plt.show()


# In[33]:


plt.figure(figsize=(8,6))
sns.histplot(data["capital-loss"])
plt.title("Distribution of capital-loss")
plt.show()


# In[34]:


plt.figure(figsize=(8,6))
sns.histplot(data["hours-per-week"],binwidth=5)
plt.title("Distribution of hours-per-week")
plt.show()


# In[35]:


#Plotting Salary vs age
sns.catplot(x="salary", y= "age", kind="box", data=data)
plt.show()


# In[36]:


# Plotting Salary vs fnlwgt
sns.catplot(x="age", y= "race", kind="box", data=data)


# In[37]:


# Plotting Salary vs hours_per_week
sns.catplot(x="sex", y= "hours-per-week", kind="violin", data=data)


# In[38]:


#setting the required plot style
plt.style.use('bmh')
#Defining a function graph()
def graph(x):
    sns.countplot(x = x,hue= 'salary', data = data )
    #adding labels for x and y axis
    plt.xlabel(x, fontsize = 15)
    plt.ylabel('Count', fontsize = 15)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
plt.figure(figsize=(20,20))
#plotting subplots using the function graph()
plt.subplot(421)
graph('age')
plt.subplot(422)
graph('education-num')
plt.subplot(423)
graph('capital-gain')
plt.subplot(424)
graph('capital-loss')
plt.subplot(425)
graph('hours-per-week')


# ### 6. Feature Reduction 

# - we can drop 'capital-gain'& 'capital-loss' both columns. 
# - The column,education-num is the numerical version of the column education, so we also drop it.

# In[39]:


data.drop(['capital-gain','capital-loss','education-num'], axis = 1,inplace = True)
data.head()


# In[40]:


data.shape


# Now, we need to convert the categorical values to numeric for modeling. Looking at the Marital-status col, there are nearly 6 different values which would mean the same as two values of being married ot no married, therefore we convert them into only two values.

# In[41]:


data.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent','Never-married','Separated','Widowed'],
             ['divorced','married','married','married','not married','not married','not married'], inplace = True)


# In[42]:


data['marital-status'].value_counts()


# Before we do further analysis, we will separate the data as numeric and categorical so that our analysis becomes easy.

# ## 7. Feature Engineering

# In[43]:


# education Category
data.education= data.education.replace(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th','10th', '11th', '12th'], 'school')
data.education = data.education.replace('HS-grad', 'high school')
data.education = data.education.replace(['Assoc-voc', 'Assoc-acdm', 'Prof-school', 'Some-college'], 'higher')
data.education = data.education.replace('Bachelors', 'undergrad')
data.education = data.education.replace('Masters', 'grad')
data.education = data.education.replace('Doctorate', 'doc')


# In[44]:


# Salary
data.Salary = data.salary.replace('<=50K', 0)
data.Salary = data.salary.replace('>50K', 1)


# In[45]:


data.corr()


# In[46]:


# Salary
data.Salary = data.Salary.replace( 0,'<=50K')
data.Salary = data.Salary.replace( 1,'>50K')


# In[47]:


data['salary'].value_counts()


# Dataset is unbalanced type

# In[48]:


#Covert workclass Columns Datatype To Category Datatype
data['workclass'] = data['workclass'].astype('category')


# ### 8.Label Encoding

# In[49]:


#apply label encoding
from sklearn.preprocessing import LabelEncoder
columns =["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
label_encoder = LabelEncoder()
for i in columns:
    data[i]=label_encoder.fit_transform(data[i])
data.head()


# In[50]:


data.head()


# In[51]:


#Moved the salary column to another variable
target_salary = data.pop('salary')
data.head()


# ## Sampling

# In[52]:


#pip install imblearn


# In[53]:


from imblearn.over_sampling import SMOTE


# In[56]:


oversample=SMOTE()
x,y=oversample.fit_resample(data,target_salary)
counter=Counter(y)
print(counter)


# In[57]:


plt.bar(counter.keys(),counter.values())


# In[58]:


#Standardisation
scale=StandardScaler()
X=scale.fit_transform(x)


# In[59]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=24) # 80% training and 20% test


# ## RANDOM FOREST

# In[60]:


rf_model=RandomForestClassifier()
rf_model.fit(X_train,Y_train)
Y_pred=rf_model.predict(X_test)
print('Accuracy on training data is:',rf_model.score(X_train,Y_train))
print('Accuracy is:',accuracy_score(Y_test,Y_pred))
print('Precision is:',precision_score(Y_test,Y_pred,average='weighted'))
print('Recall is:',recall_score(Y_test,Y_pred,average='weighted'))
print('f1 score is:',f1_score(Y_test,Y_pred,average='weighted'))
print(classification_report(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))


# In[61]:


# save the model
import pickle
filename = 'model.pkl'
pickle.dump(rf_model, open(filename, 'wb'))


# In[62]:


load_model = pickle.load(open(filename,'rb'))


# In[63]:


load_model.predict([[39,6,5,2,0,1,4,1,40,38]])


# In[ ]:




