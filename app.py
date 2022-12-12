import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image


# Load  model a 
model = joblib.load(open("model-v1.joblib","rb"))

def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    df.arrest_boro = df.arrest_boro.map({'Bronx':0, 'Brooklyn':1, 'Manhattan':2,'Queens':3,'Staten Island':4})
    df.age_group = df.age_group.map({"18-24":0,"25-44":1, "45-64":2,"65+":3,"<18":4})     
    df.prep_race = df.prep_race.map({'WHITE':5, 'BLACK':2, 'WHITE HISPANIC':6, 'BLACK HISPANIC':3,
     'UNKNOWN':0,'ASIAN / PACIFIC ISLANDER':1, 'AMERICAN INDIAN/ALASKAN NATIVE':4})
    df.prep_sex = df.prep_sex.map({"Male":1,"Female":0})
    df.population = df.population.map({'1,424,948':1, '2,641,052':4, "1,576,876":2,'2,331,143':3, '493,494':0})
    df.household_income = df.household_income.map({"$39100":0,"$62230":1,"$86470":4,"$70470":2,"$83520":3})
    df.ofns_desc = df.ofns_desc.map({'ADMINISTRATIVE CODE': 0, 'ADMINISTRATIVE CODES': 1, 'AGRICULTURE & MRKTS LAW-UNCLASSIFIED': 2, 'ALCOHOLIC BEVERAGE CONTROL LAW': 3,
 'ANTICIPATORY OFFENSES': 4, 'ARSON': 5, 'ASSAULT 3 & RELATED OFFENSES': 6, "BURGLAR'S TOOLS": 7, 'BURGLARY': 8, 'CANNABIS RELATED OFFENSES': 9, 'CHILD ABANDONMENT/NON SUPPORT': 10,
 'CRIMINAL MISCHIEF & RELATED OF': 11, 'CRIMINAL TRESPASS': 12, 'DANGEROUS DRUGS': 13, 'DANGEROUS WEAPONS': 14, 'DISORDERLY CONDUCT': 15, 'DISRUPTION OF A RELIGIOUS SERV': 16,
 'ENDAN WELFARE INCOMP': 17, 'ESCAPE 3': 18, 'FELONY ASSAULT': 19, 'FOR OTHER AUTHORITIES': 20, 'FORGERY': 21, 'FRAUDS': 22, 'FRAUDULENT ACCOSTING': 23, 'GAMBLING': 24,
 'GRAND LARCENY': 25, 'GRAND LARCENY OF MOTOR VEHICLE': 26, 'HARRASSMENT 2': 27, 'HOMICIDE-NEGLIGENT,UNCLASSIFIE': 28, 'HOMICIDE-NEGLIGENT-VEHICLE': 29, 'INTOXICATED & IMPAIRED DRIVING': 30,
 'INTOXICATED/IMPAIRED DRIVING': 31, 'JOSTLING': 32, 'KIDNAPPING': 33, 'KIDNAPPING & RELATED OFFENSES': 34, 'LOITERING/GAMBLING (CARDS, DIC': 35, 'MISCELLANEOUS PENAL LAW': 36,
 'MOVING INFRACTIONS': 37, 'MURDER & NON-NEGL. MANSLAUGHTE': 38, 'NEW YORK CITY HEALTH CODE': 39, 'NYS LAWS-UNCLASSIFIED FELONY': 40, 'OFF. AGNST PUB ORD SENSBLTY &': 41,
 'OFFENSES AGAINST PUBLIC ADMINI': 42, 'OFFENSES AGAINST PUBLIC SAFETY': 43, 'OFFENSES AGAINST THE PERSON': 44, 'OFFENSES INVOLVING FRAUD': 45, 'OFFENSES RELATED TO CHILDREN': 46, 
 'OTHER OFFENSES RELATED TO THEF': 47, 'OTHER STATE LAWS': 48, 'OTHER STATE LAWS (NON PENAL LA': 49, 'OTHER STATE LAWS (NON PENAL LAW)': 50, 'OTHER TRAFFIC INFRACTION': 51, 'PARKING OFFENSES': 52,
  'PETIT LARCENY': 53, 'POSSESSION OF STOLEN PROPERTY': 54, 'PROSTITUTION & RELATED OFFENSES': 55, 'RAPE': 56, 'ROBBERY': 57, 'SEX CRIMES': 58, 'THEFT OF SERVICES': 59, 'THEFT-FRAUD': 60,
 'UNAUTHORIZED USE OF A VEHICLE': 61, 'UNLAWFUL POSS. WEAP. ON SCHOOL': 62, 'VEHICLE AND TRAFFIC LAWS': 63})
    return df

def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index = ['Felony','Infraction','Misdemeanor','Violation'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel(" Percentage(%) Confidence Level", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Crime Category ", labelpad=10, weight='bold', size=12)
    ax.set_title('Crime Prediction', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return

st.write("""
# Crime Prediction ML Web-App 
This app predicts the ** Category of crimes in all five boroughs of NYC **  using **Demography and socioeconomical features** input via the **side panel** 
""")

#read in wine image and render with streamlit
image = Image.open('AI_&_Crime.PNG')
st.image(image, caption='Smart Policing',use_column_width=True)

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe

    """
    arrest_boro = st.sidebar.selectbox("Select a Borough in NYC?", ("Bronx","Brooklyn" "Manhattan", "Queens", "Staten Island"))
    age_group = st.sidebar.selectbox("Age group?", ("<18","18-24","25-44" "45-64", "65+"))
    prep_race = st.sidebar.selectbox("Perpetrator's Race?", ('WHITE', 'BLACK', 'WHITE HISPANIC', 'BLACK HISPANIC',
       'ASIAN / PACIFIC ISLANDER', 'AMERICAN INDIAN/ALASKAN NATIVE','UNKNOWN'))
    prep_sex = st.sidebar.selectbox("Gender?", ("Male","Female"))
    population = st.sidebar.selectbox('population', ("1,424,948", "2,641,052", "1,576,876","2,331,143", "493,494"))
    household_income = st.sidebar.selectbox('Median Household Income?', ("$39100","$62230","$86470","$70470","$83520"))
    ofns_desc = st.sidebar.selectbox("Offense Decription?", ('ASSAULT 3 & RELATED OFFENSES', 'BURGLARY', 'JOSTLING', 'RAPE',
       'SEX CRIMES', 'FELONY ASSAULT', 'DANGEROUS WEAPONS', 'ARSON',
       'ROBBERY', 'OTHER STATE LAWS (NON PENAL LA', 'CRIMINAL TRESPASS',
       'INTOXICATED & IMPAIRED DRIVING', 'MISCELLANEOUS PENAL LAW',
       'VEHICLE AND TRAFFIC LAWS', 'OTHER TRAFFIC INFRACTION',
       'PETIT LARCENY', 'GRAND LARCENY', 'DANGEROUS DRUGS',
       'CRIMINAL MISCHIEF & RELATED OF', 'OFF. AGNST PUB ORD SENSBLTY &',
       'OFFENSES INVOLVING FRAUD', 'UNAUTHORIZED USE OF A VEHICLE',
       'FORGERY', 'FOR OTHER AUTHORITIES',
       'OTHER OFFENSES RELATED TO THEF', 'POSSESSION OF STOLEN PROPERTY',
       'OFFENSES AGAINST PUBLIC ADMINI', 'INTOXICATED/IMPAIRED DRIVING',
       'CANNABIS RELATED OFFENSES', 'MURDER & NON-NEGL. MANSLAUGHTE',
       'GRAND LARCENY OF MOTOR VEHICLE', 'DISORDERLY CONDUCT',
       'OFFENSES AGAINST THE PERSON', 'ALCOHOLIC BEVERAGE CONTROL LAW',
       'OTHER STATE LAWS', 'PROSTITUTION & RELATED OFFENSES',
       'THEFT OF SERVICES', 'THEFT-FRAUD', 'HARRASSMENT 2',
       'ENDAN WELFARE INCOMP', 'NYS LAWS-UNCLASSIFIED FELONY',
       'HOMICIDE-NEGLIGENT,UNCLASSIFIE', 'OFFENSES AGAINST PUBLIC SAFETY',
       'GAMBLING', 'HOMICIDE-NEGLIGENT-VEHICLE',
       'KIDNAPPING & RELATED OFFENSES', 'ADMINISTRATIVE CODE',
       "BURGLAR'S TOOLS", 'FRAUDS', 'MOVING INFRACTIONS',
       'AGRICULTURE & MRKTS LAW-UNCLASSIFIED', 'FRAUDULENT ACCOSTING',
       'ANTICIPATORY OFFENSES', 'NEW YORK CITY HEALTH CODE',
       'CHILD ABANDONMENT/NON SUPPORT', 'LOITERING/GAMBLING (CARDS, DIC',
       'ESCAPE 3', 'UNLAWFUL POSS. WEAP. ON SCHOOL', 'PARKING OFFENSES',
       'KIDNAPPING', 'OTHER STATE LAWS (NON PENAL LAW)',
       'ADMINISTRATIVE CODES', 'OFFENSES RELATED TO CHILDREN',
       'DISRUPTION OF A RELIGIOUS SERV'))


    features = {'arrest_boro': arrest_boro,
            'age_group': age_group,
            'prep_race': prep_race,
            'prep_sex': prep_sex,
            'population': population,
            'household_income': household_income,
            'ofns_desc': ofns_desc
            }
    data = pd.DataFrame(features,index=[0])

    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)

st.subheader('User Input parameters')
st.write(user_input_df)

prediction = model.predict(processed_user_input)
prediction_proba = model.predict_proba(processed_user_input)

visualize_confidence_level(prediction_proba)
