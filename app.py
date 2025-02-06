from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import pickle

model=tf.keras.models.load_model('model.h5')

with open('ONE_HOT','rb') as file:
    ONE_HOT=pickle.load(file)

with open('gender_encoder','rb') as file:
    gender_encoder=pickle.load(file)

with open('scaler','rb') as file:
    scaler=pickle.load(file)

##create a streamlit app

st.title("Customer Churn Prediction")

## user input

geography=st.selectbox('Geography',ONE_HOT.categories_[0])
gender=st.selectbox('Gender',gender_encoder.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])
st.write(gender)
st.write(gender_encoder.classes_)
st.write(gender_encoder.transform([gender]))
##craete the input data dict

input_data={
    "CreditScore": [credit_score],
    "Gender": [gender_encoder.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
}


geo_encoded=ONE_HOT.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=ONE_HOT.get_feature_names_out())

input_data = pd.concat([pd.DataFrame(input_data, index=[0]), geo_encoded_df], axis=1)

##scale the input data
input_data=scaler.transform(input_data)

##prediction churn

prediction=model.predict(input_data)
prediction_proba=prediction[0][0]

if prediction_proba>0.5:
    st.write("The Customer is likley to churn")
else:
    st.write("The customer will not churn")
