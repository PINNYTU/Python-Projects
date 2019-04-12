#!/usr/bin/env python
# coding: utf-8

# In[1]:


#coding: utf-8
import pandas as pd
import datetime 
import time
import numpy as np
import matplotlib as mp
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


# In[2]:


#Custom scaler class >>> to create new table that dummies don't get to standardization
class CustomScaler (BaseEstimator, TransformerMixin):
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns=columns
        self.mean=None
        self.var=None
    def fit(self, X,y=None):
        self.scaler.fit(X[self.columns],y)
        self.mean=np.mean(X[self.columns])
        self.var=np.var(X[self.columns])
        return self
    def transform(self, X,y=None,copy=None):
        init_col_order=X.columns
        X_scale=pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scale=X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scale,X_scale],axis=1)[init_col_order]

#Main class 
class absenteeism_model():
    
    def __init__(self,model_file,scaler_file): #open predict model (Logistic regrssion, scaler) that has been saved
        with open ('model','rb') as model_file: #model=filename #wb=writebytes
            self.reg = pickle.load(model_file)
        with open ('scaler','rb') as scaler_file:
            self.scaler = pickle.load(scaler_file)
            self.data = None
   
    def Dataprocessing(self,data_file):
        df = pd.read_csv(data_file)
        #df=df.drop(columns='Unnamed: 0')
    #Store for further use  
        self.df_with_predictions = df.copy()
    #drop ID columns
        #df['Absenteeism Time in Hours'] = 'NaN'
    #Create new variables for each values in a column
        reason_group=pd.get_dummies(df["Reason for Absence"])    
    #group columns 
        reason1=reason_group.loc[:,1:14].max(axis=1)
        reason2=reason_group.loc[:,15:17].max(axis=1)
        reason3=reason_group.loc[:,18:21].max(axis=1)
        reason4=reason_group.loc[:,22:28].max(axis=1)
        new_df=pd.concat([df,reason1,reason2,reason3,reason4],axis=1)
     #change columns
    #Adjust datetime 
        day=[]
        month=[]
        new_df['Date']=pd.to_datetime(new_df["Date"],format='%d/%m/%Y')
        for d in new_df['Date'] :
            day.append(datetime.date.strftime(d, '%d' ))
            month.append(datetime.date.strftime(d, '%m'))

        day = pd.DataFrame(day)
        month = pd.DataFrame(month)
        print(day)
        print(month)
    #join table to data frame
        new_df=pd.concat([new_df,day,month],axis=1)
        new_df.drop(columns=["ID","Reason for Absence","Date","Absenteeism Time in Hours"],inplace= True)
    #Change value in column
        x=[]
        for d in new_df["Education"] :
            if d==1 :
                x.append(0)
            if d>1 :
                x.append(1) 
        Education=pd.DataFrame(x,columns=['Education'])
        new_df.drop(columns='Education',inplace=True)
        new_df=pd.concat([new_df,Education],axis=1)
    #Change order of columns
        new_df.columns= ['Transportation Expense','Distance to Work', 'Age', 'Daily Work Load Average','Body Mass Index', 'Children', 'Pets', 'Reason1', 'Reason2', 'Reason3','Reason4', 'Day of week', 'Month value','Education']
        new_df=new_df[['Reason1', 'Reason2','Reason3', 'Reason4', 'Month value', 'Day of week', 'Transportation Expense', 'Distance to Work',
       'Age', 'Daily Work Load Average', 'Body Mass Index','Education', 'Children',
       'Pets']]
        print(new_df)
    #Furthere uses
        self.preprocessed_data = new_df.copy()
        self.data = self.scaler.transform(new_df)
    
  

    def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
         
        # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
         
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
    def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data
        
    
    
    
    


# In[38]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




