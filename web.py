import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb


with open('xgb.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Postoperative prolonged hospital stay calculator for female patients undergoing gastrointestinal surgery')
st.subheader('Estimates the risk of postoperative prolonged hospital stay following gastrointestinal surgery in female patients.')
st.divider()
st.header('Pearls/Pitfalls')
st.write('This calculator is a data-driven risk stratification tool designed to estimate the likelihood of prolonged hospital stay after gastrointestinal surgery in female patients. It is based on the FDP-PONV randomized controlled trial which included 1101 female patients and integrates extreme gradient boosting algorithm to help identify high-risk individuals and optimize postoperative management.')
st.header('When to use')
st.write('Female patients undergoing gastrointestinal surgery, to estimate the risk of postoperative prolonged hospital stay.')
st.header('Why use')
st.write('Early identification of female patients at high risk for prolonged hospitalization enables clinicians to anticipate and allocate postoperative care resources more effectively, guide patient and family discussions about recovery expectations, minimize unnecessary hospital days and associated costs, and support personalized interventions and discharge planning. By leveraging comprehensive perioperative data, this tool facilitates objective risk stratification to enhance clinical outcomes, optimize resource utilization, and improve overall patient satisfaction.')
#st.write('A patient with detailed admission information can be more objectively risk stratified for their prognosis, quantify their risk, and potentially lead to shorter hospital stays, fewer inappropriate interventions, and more appropriate interventions.')
st.divider()

edu_input = st.selectbox('Education level',('Middle school or lower', 'High school', 'College or higher'))
DurationOfSurgery_input = st.text_input('Duration of surgery (min)','200')
Postoperative_albumin_infusion_input = st.selectbox('Postoperative albumin infusion',('No', 'Yes'))
HighestPainScore_input = st.text_input('Highest pain score (numeric rating scale) at mobility during 73-120 h after surgery','3')
TimeToFirstFlatus_input = st.text_input('Time to first flatus after surgery (h)','50')
TimeToFirstDefaecation_input = st.text_input('Time to first defaecation after surgery (h)','80')
PostoperativeInfection_input = st.selectbox('Postoperative infection',('No', 'Yes'))



edu_1 = np.where(edu_input=='Middle school or lower',1,0)
edu_2 = np.where(edu_input=='High school',1,0)
edu_3 = np.where(edu_input=='College or higher',1,0)
Postoperative_albumin_infusion = np.where(Postoperative_albumin_infusion_input=='No',0,1)
PostoperativeInfection = np.where(PostoperativeInfection_input=='No',0,1)



if st.button('Predict'):
    DurationOfSurgery = (float(DurationOfSurgery_input)-197.96493506)/75.7581255
    HighestPainScore = (float(HighestPainScore_input)-2.66623377)/1.25419922
    TimeToFirstFlatus = (float(TimeToFirstFlatus_input)-51.5974026)/29.449565
    TimeToFirstDefaecation = (float(TimeToFirstDefaecation_input)-76.66883117)/44.3180441
    print(type(DurationOfSurgery_input))
    features = np.array([DurationOfSurgery,HighestPainScore,TimeToFirstFlatus,TimeToFirstDefaecation,edu_2,edu_3,Postoperative_albumin_infusion,PostoperativeInfection]).reshape(1,-1)
    col1, col2 = st.columns(2)
    p = model.predict_proba(features)[:,1]*100
    col1.metric("Score", int(p), )
    #col2.metric("Probability of death from admission to 6 months", int(p), )
    #prediction = model.predict_proba(features)[:,1]
    #st.write(' Based on feature values, your risk score is : '+ str(int(prediction * 100)))
