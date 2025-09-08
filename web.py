import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb


with open('xgb.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Postoperative prolonged hospital stay calculator for patients undergoing gastrointestinal surgery')
st.subheader('Estimates the risk of postoperative prolonged hospital stay following gastrointestinal surgery in patients.')
st.divider()
st.header('Pearls/Pitfalls')
st.write('This calculator is a data-driven risk stratification tool designed to estimate the likelihood of prolonged hospital stay after gastrointestinal surgery in patients. It is based on the FDP-PONV randomized controlled trial which included 1141 patients and integrates extreme gradient boosting algorithm to help identify high-risk individuals and optimize postoperative management.')
st.header('When to use')
st.write('Patients undergoing gastrointestinal surgery, to estimate the risk of postoperative prolonged hospital stay.')
st.header('Why use')
st.write('Early identification of patients at high risk for prolonged hospitalization enables clinicians to anticipate and allocate postoperative care resources more effectively, guide patient and family discussions about recovery expectations, minimize unnecessary hospital days and associated costs, and support personalized interventions and discharge planning. By leveraging comprehensive perioperative data, this tool facilitates objective risk stratification to enhance clinical outcomes, optimize resource utilization, and improve overall patient satisfaction.')
#st.write('A patient with detailed admission information can be more objectively risk stratified for their prognosis, quantify their risk, and potentially lead to shorter hospital stays, fewer inappropriate interventions, and more appropriate interventions.')
st.divider()

edu_input = st.selectbox('Education level',('Middle school or lower', 'High school', 'College or higher'))
PreoperativeHypoalbuminemia_input = st.selectbox('Preoperative hypoalbuminemia',('No', 'Yes'))
PreoperativeInsufficientSleep_input = st.selectbox('Preoperative insufficient sleep',('No', 'Yes'))
TypeOfSurgery_input = st.selectbox('Type of surgery',('Gastrectomy or small intestinal resection', 'Colon resection', 'Rectum resection'))
DurationOfSurgery_input = st.text_input('Duration of surgery (min)','200')
Intraoperativebloodloss_input = st.text_input('Intraoperative blood loss (ml)','50')
Postoperativehypotension_input = st.selectbox('Postoperative hypotension',('No', 'Yes'))
Postoperative_albumin_infusion_input = st.selectbox('Postoperative albumin infusion',('No', 'Yes'))
HighestPainScore_input = st.text_input('Highest pain score (numeric rating scale) at mobility during 73-120 h after surgery','3')
TimeToFirstFlatus_input = st.text_input('Time to first flatus after surgery (h)','50')
TimeToFirstDefaecation_input = st.text_input('Time to first defaecation after surgery (h)','80')
PostoperativeInfection_input = st.selectbox('Postoperative infection',('No', 'Yes'))
PostoperativeAcuteKidneyInjury_input = st.selectbox('Postoperative acute kidney injury',('No', 'Yes'))




edu_1 = np.where(edu_input=='Middle school or lower',1,0)
edu_2 = np.where(edu_input=='High school',1,0)
edu_3 = np.where(edu_input=='College or higher',1,0)
Postoperative_albumin_infusion = np.where(Postoperative_albumin_infusion_input=='No',0,1)
PostoperativeInfection = np.where(PostoperativeInfection_input=='No',0,1)

PreoperativeHypoalbuminemia = np.where(PreoperativeHypoalbuminemia_input=='No',0,1)
PreoperativeInsufficientSleep = np.where(PreoperativeInsufficientSleep_input=='No',0,1)
TypeOfSurgery_1 = np.where(TypeOfSurgery_input=='Gastrectomy or small intestinal resection',1,0)
TypeOfSurgery_2 = np.where(TypeOfSurgery_input=='Colon resection',1,0)
TypeOfSurgery_3 = np.where(TypeOfSurgery_input=='Rectum resection',1,0)
Postoperativehypotension = np.where(Postoperativehypotension_input=='No',0,1)
PostoperativeAcuteKidneyInjury = np.where(PostoperativeAcuteKidneyInjury_input=='No',0,1)

if st.button('Predict'):
    DurationOfSurgery = (float(DurationOfSurgery_input)-198.90149626)/73.22723845
    Intraoperativebloodloss = (float(Intraoperativebloodloss_input)-101.66458853)/209.0399405
    HighestPainScore = (float(HighestPainScore_input)-2.78553616)/1.28443212
    TimeToFirstFlatus = (float(TimeToFirstFlatus_input)-54.19950125)/30.71666999
    TimeToFirstDefaecation = (float(TimeToFirstDefaecation_input)-77.60473815)/43.31145832
    
    print(type(DurationOfSurgery_input))
    features = np.array([DurationOfSurgery,Intraoperativebloodloss,HighestPainScore,TimeToFirstFlatus,TimeToFirstDefaecation,edu_2,edu_3,Postoperativehypotension,Postoperative_albumin_infusion,PostoperativeInfection,PreoperativeHypoalbuminemia,PreoperativeInsufficientSleep,PostoperativeAcuteKidneyInjury,TypeOfSurgery_2,TypeOfSurgery_3]).reshape(1,-1)
    col1, col2 = st.columns(2)
    p = model.predict_proba(features)[:,1]*100
    col1.metric("Score", int(p), )
    #col2.metric("Probability of death from admission to 6 months", int(p), )
    #prediction = model.predict_proba(features)[:,1]
    #st.write(' Based on feature values, your risk score is : '+ str(int(prediction * 100)))
