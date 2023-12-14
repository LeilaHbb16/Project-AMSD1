
#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Variables: 
---------

data : diabetes health indicators original dataset
X : features dataset
Y : target labels
pred : list of predicted labels 

''';


# In[2]:


from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle


# In[3]:


def classifier(mat, model):
    
    if model=='Boosting':
        model = pickle.load(open("Boosting.pkl", "rb"))
        pred = model.predict(mat)
        
    elif model=='SVM':
        model = pickle.load(open("SVM.pkl", "rb"))
        pred = model.predict(mat)
        
    elif model=='RandomForest':
        model = pickle.load(open("RandomForest.pkl", "rb"))
        pred = model.predict(mat)
        
    else:
        raise Exception("Please select one of the three methods : SVM, RF, GBC")
    
    return pred


# In[4]:


# Import data
data = pd.read_csv('data/diabetes_health_validation.csv')
data['Diabetes_012'] = data['Diabetes_012'].astype(int)
# prepocessing de data
data= data.drop("Unnamed: 0", axis=1) #suppression de la variable Unnamed: 0
data = data.drop_duplicates() #suppression des instances duplique
data_standardized = pd.DataFrame(scaler.fit_transform(data), columns=df.columns) #standardiser les donnees
 # spliting 
X = data_standardized.drop(columns=['Diabetes_012'])
y = data_standardized['Diabetes_012']

# Predict labels using trained models
models = ['SVM', 'Boosting', 'RandomForest']
for model in models:

    # Make prediction
    pred = classifier(X, model)

    # Evaluate model results
    accuracy = accuracy_score(pred, y)
    f1 = f1_score(pred, y, average='macro', zero_division=True)

    # Print results
    print(f'Model: {model}\n-----\nAccuracy: {accuracy:.2f} \nF1_score: {f1:.2f} \n')

    # Confusion Matrix: 
    cm_normalized = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm_normalized)
