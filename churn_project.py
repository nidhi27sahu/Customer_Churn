# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pickle


load = open('churn.pkl', 'rb')
model = pickle.load(load)



st.title('Telecom  Churn Predictor ')


def predict(voice_messages,intl_plan,day_mins,day_charge,
            customer_calls
            ):  
    pred = model.predict([[voice_messages,intl_plan,day_mins,day_charge,
                customer_calls]])
    return pred

def start():
    
    voice_messages = st.number_input('Number of Voicemail Messages', min_value=0, max_value=43)
    intl_plan = st.selectbox('Do you have International Plan?', ('Yes', 'No'))
    day_mins=st.number_input('Total minutes customer have used the service during daytime ?', min_value=0,max_value=351)
    day_charge = st.number_input('Total charges for day time ', min_value=0, max_value=55)
    customer_calls= st.number_input('Total number of calls to customer care :phone:', min_value=0, max_value=4)
    
    
    if st.button('Predict'):
        result = predict(voice_messages,intl_plan,day_mins,day_charge,
                    customer_calls)
        st.success('Will customer Churn? : {} '.format(result) )
        
if __name__== '__main__':
    start()
        
    
    