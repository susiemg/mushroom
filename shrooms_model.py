import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


mushrooms=pd.read_csv('mushrooms_clean.csv')


#feature encoding
df=mushrooms.copy()
target="class"
encode=list(df.loc[:,df.columns!="class"].columns)

for col in encode:
  dummy=pd.get_dummies(df[col],prefix=col)
  df=pd.concat([df,dummy],axis=1)
  del df[col]

target_mapper={"Edible":0, "Poisonous":1}
def target_encode(val):
  return target_mapper[val]

df["class"] = df["class"].apply(target_encode)

#separating X and Y
X=df.drop("class",axis=1)
Y=df["class"]

#Build RF model
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(X,Y)

#saving the model
import pickle
pickle.dump(clf,open("mushrooms_clf.pkl","wb"))


