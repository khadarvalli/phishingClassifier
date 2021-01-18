#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

i = open("classifier.sav","rb")
classifier=pickle.load(i)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(having_IP_Address,URL_Length,Shortining_Service,having_At_Symbol,double_slash_redirecting,Prefix_Suffix,having_Sub_Domain,SSLfinal_State,Domain_registeration_length,Favicon,port,HTTPS_token,Request_URL,URL_of_Anchor,Links_in_tags,SFH,Submitting_to_email,Abnormal_URL,Redirect,on_mouseover,RightClick,popUpWidnow,Iframe,age_of_domain,DNSRecord,web_traffic,Page_Rank,Google_Index,Links_pointing_to_page,Statistical_report):
    
    prediction=classifier.predict([[having_IP_Address,URL_Length,Shortining_Service,having_At_Symbol,double_slash_redirecting,Prefix_Suffix,having_Sub_Domain,SSLfinal_State,Domain_registeration_length,Favicon,port,HTTPS_token,Request_URL,URL_of_Anchor,Links_in_tags,SFH,Submitting_to_email,Abnormal_URL,Redirect,on_mouseover,RightClick,popUpWidnow,Iframe,age_of_domain,DNSRecord,web_traffic,Page_Rank,Google_Index,Links_pointing_to_page,Statistical_report]])
    print(prediction)
    return prediction



def main():
    st.title("Phising Website ")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Phising Website ML App </h2>
    </div>
    """

 
    st.markdown(html_temp,unsafe_allow_html=True)
    having_IP_Address = st.text_input("having_IP_Address","Type Here")
    URL_Length = st.text_input("URL_Length","Type Here")
    Shortining_Service = st.text_input("Shortining_Service","Type Here")
    having_At_Symbol = st.text_input("having_At_Symbol","Type Here")
    double_slash_redirecting = st.text_input("double_slash_redirecting","Type Here")
    Prefix_Suffix = st.text_input("Prefix_Suffix","Type Here")
    having_Sub_Domain = st.text_input("having_Sub_Domain","Type Here")
    SSLfinal_State = st.text_input("SSLfinal_State","Type Here")
    Domain_registeration_length = st.text_input("Domain_registeration_length","Type Here")
    Favicon = st.text_input("Favicon","Type Here")
    port = st.text_input("port","Type Here")
    HTTPS_token = st.text_input("HTTPS_token","Type Here")
    Request_URL = st.text_input("Request_URL","Type Here")
    URL_of_Anchor = st.text_input("URL_of_Anchor","Type Here")
    Links_in_tags = st.text_input("Links_in_tags","Type Here") 
    SFH = st.text_input("SFH","Type Here")
    Submitting_to_email = st.text_input("Submitting_to_email","Type Here")
    Abnormal_URL = st.text_input("Abnormal_URL","Type Here")
    Redirect = st.text_input("Redirect","Type Here")
    on_mouseover = st.text_input("on_mouseover","Type Here")
    RightClick = st.text_input("RightClick","Type Here")
    popUpWidnow = st.text_input("popUpWidnow","Type Here")
    Iframe = st.text_input("Iframe","Type Here")
    age_of_domain = st.text_input("age_of_domain","Type Here")
    DNSRecord = st.text_input("DNSRecord","Type Here")
    web_traffic = st.text_input("web_traffic","Type Here")
    Page_Rank = st.text_input("Page_Rank","Type Here")
    Google_Index = st.text_input("Google_Index","Type Here")
    Links_pointing_to_page = st.text_input("Links_pointing_to_page","Type Here")
    Statistical_report = st.text_input("Statistical_report","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(having_IP_Address,URL_Length,Shortining_Service,having_At_Symbol,double_slash_redirecting,Prefix_Suffix,having_Sub_Domain,SSLfinal_State,Domain_registeration_length,Favicon,port,HTTPS_token,Request_URL,URL_of_Anchor,Links_in_tags,SFH,Submitting_to_email,Abnormal_URL,Redirect,on_mouseover,RightClick,popUpWidnow,Iframe,age_of_domain,DNSRecord,web_traffic,Page_Rank,Google_Index,Links_pointing_to_page,Statistical_report)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




