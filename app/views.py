from   django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd
reloadmodel=joblib.load('model.pkl')

symptoms_index={'Abdominal Pain': 39,
 'Abnormal Menstruation': 101,
 'Acidity': 8,
 'Acute Liver Failure': 44,
 'Altered Sensorium': 98,
 'Anxiety': 16,
 'Back Pain': 37,
 'Belly Pain': 100,
 'Blackheads': 123,
 'Bladder Discomfort': 89,
 'Blister': 129,
 'Blood In Sputum': 118,
 'Bloody Stool': 61,
 'Blurred and Distorted Vision': 49,
 'Breathlessness': 27,
 'Brittle nails': 72,
 'Bruising': 66,
 'Burning Micturution': 12,
 'Chest Pain': 56,
 'Chills': 5,
 'Cold Hands and Feets': 17,
 'Coma': 113,
 'Congestion': 55,
 'Constipation': 38,
 'Continuous Feel Of Urine': 91,
 'Continuous Sneezing': 3,
 'Cough': 24,
 'Cramps': 65,
 'Dark Urine': 33,
 'Dehydration': 29,
 'Depression': 95,
 'Diarrhoea': 40,
 'Dischromic Patches': 102,
 'Distention Of Abdomen': 115,
 'Dizziness': 64,
 'Drying And Tingling Lips': 76,
 'Enlarged Thyroid': 71,
 'Excessive Hunger': 74,
 'Extra Marital Contacts': 75,
 'Family History': 106,
 'Fast Heart Rate': 58,
 'Fatigue': 14,
 'Fluid Overload': 45,
 'Fluid Overload.1': 117,
 'Foul Smell Of Urine': 90,
 'Headache': 31,
 'High Fever': 25,
 'Hip Joint Pain': 79,
 'History Of Alcohol Consumption': 116,
 'Increased Appetite': 104,
 'Indigestion': 30,
 'Inflammatory Nails': 128,
 'Internal Itching': 93,
 'Irregular Sugar Level': 23,
 'Irritability': 96,
 'Irritation In Anus': 62,
 'Itching': 0,
 'Joint Pain': 6,
 'Knee Pain': 78,
 'Lack Of Concentration': 109,
 'Lethargy': 21,
 'Loss Of Appetite': 35,
 'Loss Of Balance': 85,
 'Loss Of Smell': 88,
 'Malaise': 48,
 'Mild Fever': 41,
 'Mood Swings': 18,
 'Movement Stiffness': 83,
 'Mucoid Sputum': 107,
 'Muscle Pain': 97,
 'Muscle Wasting': 10,
 'Muscle Weakness': 80,
 'Nausea': 34,
 'Neck Pain': 63,
 'Nodal Skin Eruptions': 2,
 'Obesity': 67,
 'Pain Behind The Eyes': 36,
 'Pain During Bowel Movements': 59,
 'Pain In Anal Region': 60,
 'Painful Walking': 121,
 'Palpitations': 120,
 'Passage Of Gases': 92,
 'Patches In Throat': 22,
 'Phlegm': 50,
 'Polyuria': 105,
 'Prominent Veins On Calf': 119,
 'Puffy Face And Eyes': 70,
 'Pus Filled Pimples': 122,
 'Receiving Blood Transfusion': 111,
 'Receiving Unsterile Injections': 112,
 'Red Sore Around Nose': 130,
 'Red Spots Over Body': 99,
 'Redness Of Eyes': 52,
 'Restlessness': 20,
 'Runny Nose': 54,
 'Rusty Sputum': 108,
 'Scurring': 124,
 'Shivering': 4,
 'Silver Like Dusting': 126,
 'Sinus Pressure': 53,
 'Skin Peeling': 125,
 'Skin Rash': 1,
 'Slurred Speech': 77,
 'Small Dents In Nails': 127,
 'Spinning Movements': 84,
 'Spotting Urination': 13,
 'Stiff Neck': 81,
 'Stomach Bleeding': 114,
 'Stomach Pain': 7,
 'Sunken Eyes': 26,
 'Sweating': 28,
 'Swelled Lymph Nodes': 47,
 'Swelling Joints': 82,
 'Swelling Of Stomach': 46,
 'Swollen Blood Vessels': 69,
 'Swollen Extremeties': 73,
 'Swollen Legs': 68,
 'Throat Irritation': 51,
 'Toxic Look(typhos)': 94,
 'Ulcers On Tongue': 9,
 'Unsteadiness': 86,
 'Visual Disturbances': 110,
 'Vomiting': 11,
 'Watering From Eyes': 103,
 'Weakness In Limbs': 57,
 'Weakness Of One Body Side': 87,
 'Weight Gain': 15,
 'Weight Loss': 19,
 'Yellow Crust Ooze': 131,
 'Yellow Urine': 42,
 'Yellowing Of Eyes': 43,
 'Yellowish Skin': 32}
description=pd.read_csv("des.csv")
description.set_index("Disease",drop=True,inplace=True)
des=description.to_dict(orient="index")

precaution=pd.read_csv("precaution.csv")
precaution.set_index("Disease",drop=True,inplace=True)
pre=precaution.to_dict(orient="index")

def fill_input(input,s):
    if(s=='Open this select menu'):
        return
    else:
        input[symptoms_index[s]]=1;
        return
def index(request):
    return render(request,'index.html')

def result(request):
    if request.method=='POST':
        print("The Disease is : \n")
        print(request.POST.get('s1'))
        input=np.zeros(132)
        fill_input(input,request.POST.get('s1'))
        fill_input(input,request.POST.get('s2'))
        fill_input(input,request.POST.get('s3'))
        fill_input(input,request.POST.get('s4'))
        fill_input(input,request.POST.get('s5'))
    ans=reloadmodel.predict([input])
    ans=ans[0]
    print("The diseas is : \n")
    print(ans)
    d=des[ans]['Description']
    p=pre[ans]
    p1=p["Precaution_1"]
    p2=p["Precaution_2"]
    p3=p["Precaution_3"]
    p4=p["Precaution_4"]
    print(p)
    is_zero=np.all(input==0)
    if(is_zero):
        return render(request,'result.html',{'ans':'You Have Not Given Us any Symptom!'})
    else:
        return render(request,'result.html',{'ans':ans,'des':d,'pre1':p1,'pre2':p2,'pre3':p3,'pre4':p4})
