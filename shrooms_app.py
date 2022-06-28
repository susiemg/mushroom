import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle



#import os
#os.getcwd()
#os.chdir("/Users/susanamaganga/Desktop/mushrooms")

#mushrooms=pd.read_csv('mushrooms.csv')
#list(mushrooms.loc[:,mushrooms.columns!="class"].columns)

#mushrooms.iloc[:,2].value_counts()

st.write('''
# To Eat Or Not To Eat?

This app predicts whether a mushroom is edible or poisonous.

To see the original data, please visit this [Kaggle link](https://www.kaggle.com/datasets/uciml/mushroom-classification)

''')

st.sidebar.header("User Input Feautures")

def user_input_features():
  cap_shape=st.sidebar.selectbox("Cap Shape",("Bell","Conical","Convex","Flat","Knobbed","Sunken"),key="1")
  cap_surface=st.sidebar.selectbox("Cap Surface",("Fibrous","Grooves","Scaly","Smooth"),key="2")
  cap_color=st.sidebar.selectbox("Cap Colour",("Brown","Buff","Cinnamon","Gray","Green","Pink","Purple","Red","White","Yellow"))
  bruises=st.sidebar.selectbox("Are bruises present?",("Yes","No"))
  odor=st.sidebar.selectbox("Odor",("Almond","Anise","Creosote","Fishy","Foul","Musty","Pungent","Spicy","None")) 
  gill_attachment=st.sidebar.selectbox("Gill Attachment",("Attached","Descending","Free","Notched"))
  gill_spacing=st.sidebar.selectbox("Gill Spacing",("Close","Crowded","Distant"))
  gill_size=st.sidebar.selectbox("Gill Size",("Broad","Narrow"))
  gill_color=st.sidebar.selectbox("Gill Colour",("Black","Brown","Buff","Chocolate","Gray","Green","Orange","Pink","Purple","Red","White","Yellow"))
  stalk_shape=st.sidebar.selectbox("Stalk Shape",("Enlarging","Tapering"))
  stalk_root=st.sidebar.selectbox("Stalk Root",("Bulbous","Club","Cup","Equal","Rhizomorphs","Rooted","Missing"))
  stalk_surface_above_ring=st.sidebar.selectbox("Stalk Surface Above Ring",("Fibrous","Scaly","Silky","Smooth"))
  stalk_surface_below_ring=st.sidebar.selectbox("Stalk Surface Below Ring",("Fibrous","Scaly","Silky","Smooth"))
  stalk_color_above_ring=st.sidebar.selectbox("Stalk Colour Above Ring",("Brown","Buff","Cinnamon","Gray","Orange","Pink","Red","White","Yellow"))
  stalk_color_below_ring=st.sidebar.selectbox("Stalk Colour Below Ring",("Brown","Buff","Cinnamon","Gray","Orange","Pink","Red","White","Yellow"))
  veil_type=st.sidebar.selectbox("Veil Type",("Partial","Universal"))
  veil_color=st.sidebar.selectbox("Veil Colour",("Brown","Orange","White","Yellow"))
  ring_number=st.sidebar.selectbox("Ring Number",("None","One","Two"))
  ring_type=st.sidebar.selectbox("Ring Type",("Evanescent","Flaring","Large","Pendant","Sheathing","Zone","None"))
  spore_print_color=st.sidebar.selectbox("Spore Print Colour",("Black","Brown","Buff","Chocolate","Gray","Green","Orange","Pink","Purple","Red","White","Yellow"))
  population=st.sidebar.selectbox("Population",("Abundant","Clustered","Numerous","Scattered","Several","Solitary"))
  habitat=st.sidebar.selectbox("Habitat",("Grasses","Leaves","Meadows","Paths","Urban","Waste","Woods"))
  
  data={'cap_shape':cap_shape,
        'cap_surface':cap_surface,
        'cap_color':cap_color,
        'bruises':bruises,
        'odor':odor,
        'gill_attachment':gill_attachment,
        'gill_spacing':gill_spacing,
        'gill_size':gill_size,
        'gill_color':gill_color,
        'stalk_shape':stalk_shape,
        'stalk_root':stalk_root,
        'stalk_surface_above_ring':stalk_surface_above_ring,
        'stalk_surface_below_ring':stalk_surface_below_ring,
        'stalk_color_above_ring':stalk_color_above_ring,
        'stalk_color_below_ring':stalk_color_below_ring,
        'veil_type':veil_type,
        'veil_color':veil_color,
        'ring_number':ring_number,
        'ring_type':ring_type,
        'spore_print_color':spore_print_color,
        'population':population,
        'habitat':habitat}
        
  features=pd.DataFrame(data,index=[0])
  return features

input_df= user_input_features()

mushrooms_raw=pd.read_csv('mushrooms_clean.csv')
mushrooms=mushrooms_raw.drop(columns=["class"]) #drop bc we're going to predict this
df=pd.concat([input_df,mushrooms],axis=0) #combine input features to dataset

#encoding categorical feautures
encode=list(df.columns)
for col in encode:
  dummy=pd.get_dummies(df[col],prefix=col)
  df=pd.concat([df,dummy],axis=1)
  del df[col]
  
df=df[:1] #selects only the first row

#st.subheader("User Input Features")
#st.write(df)

load_clf=pickle.load(open("mushrooms_clf.pkl","rb")) #reads saved classification model

#apply model to predict
prediction=load_clf.predict(df)
prediction_proba=load_clf.predict_proba(df)

st.subheader("Prediction")
mushrooms_class=np.array(["Edible","Poisonous"])
st.write(mushrooms_class[prediction])

st.subheader("Prediction Probability")
st.write(prediction_proba)

