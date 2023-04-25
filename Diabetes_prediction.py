import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title('Diabetes Prediction App')
from PIL import Image
header_image = Image.open('medicine_photo.jpg')
st.image(header_image, caption= '', use_column_width = True)
#header_image = plt.imshow(r'D:\AI\Projects\Diabetes_prediction\medicine_photo.jpg')
st.write("This app uses Machine Learning Algorithm to study  \
    different features related to Diabetes and give you a prediction \
        of the cause of the disease. \t \n \
            To start follow the Instructions below")

#, Glucose, BloodPresure, SkinThickness, \
#    Insulin, BMI, DiabetesPedigreeFunction, Age
st.cache()
Pregnancies = st.slider(label= 'Please enter the number of pregnancies that you had', \
    min_value= 0, max_value= 100)
link = 'https://www.webmd.com/diabetes/guide/how-test-blood-glucose'
web_md = format(link)
st.write('Most people with diabetes need to check their blood sugar (glucose) levels \
            regularly. The results help you and your doctor manage those levels, \
                which helps you avoid diabetes complications. Getting your accurate blood \
                    sugar level can be a challenging task but it is a key feature\
                        in predicting the occurance of diabetes')
st.write('For more information use this link for a better understanding of your \
        glucose level ' + web_md)
Glucose = st.number_input(label= 'Enter your Glucose level', min_value=0, step=1)
#calculating Blood pressure
BloodPressure = st.number_input(label= 'Enter your Blood Pressure', min_value=0)
#calculating Insulin level
Insulin = st.number_input(label= 'Enter your Insulin Level', min_value=0)
#calculating bmi
weight = st.number_input(label= 'Enter your Weight in Kg', value= 10, min_value=1)
height_cm = st.number_input(label= 'enter your height in cm', value= 50, min_value=1)
height = height_cm / 100
bmi = weight / height**2
user_bmi = round(bmi, 2)
st.write('Your Body Mass Index is ' + str(user_bmi))
age = st.number_input(label= 'Enter your Age', min_value=1)       

#load the trained ml model
loaded_model = open('diabetes_alg.pkl', 'rb')
diabetes_alg = pickle.load(loaded_model)

#create a dataset for the user input
user_data = [Pregnancies, Glucose, BloodPressure, Insulin, bmi, age]
model_data = pd.DataFrame([user_data], columns=['Pregnancies', 'Glucose', 'BloodPresure', 
                                              'Insulin', 'BMI', 'Age'])

# use the pre-trained alg to predict on the user data
results = diabetes_alg.predict_proba(model_data)

#code to access the score of the final estimator
model_score= round(results[0][0] * 100)
model_prediction = results[0][1]
if model_prediction == 0:
    user_results = 'No diabetes'
else:
    user_results = 'no diabetes'
st.write('You have a ' + str(model_score) + ' % that you have ' + user_results )