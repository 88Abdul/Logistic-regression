# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    gender = st.sidebar.selectbox('Gender',('1','0'))
    insurance = st.sidebar.selectbox('Insurance',('1','0'))
    wearing_belt= st.sidebar.selectbox('SeatBelt',('1','0'))
    age = st.sidebar.number_input("Insert the Age")
    claim_amount = st.sidebar.number_input("Insert Loss")
    data = {'CLMSEX':gender,
            'CLMINSUR':insurance,
            'SEATBELT':wearing_belt,
            'CLMAGE':age,
            'LOSS':claim_amount}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

claimants = pd.read_csv("claimants.csv")
claimants.drop(["CASENUM"],inplace=True,axis = 1)
claimants = claimants.dropna()

X = claimants.iloc[:,[1,2,3,4,5]]
Y = claimants.iloc[:,0]
clf = LogisticRegression()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)