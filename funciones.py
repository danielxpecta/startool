import streamlit as st
import pickle as pk
import pandas as pd
import io
import base64
import json
import re
import uuid
import branca.colormap as cm
from datetime import datetime,date,timedelta
import numpy as np
import logging
import sys
import os
from sklearn.cluster import KMeans
import math

#Load the necessary files

file_name = "Modelos/prospecting_model.pkl"
final_model = pk.load(open(file_name, "rb"))
standard_model = pk.load(open('Modelos/standard_model.sav', 'rb'))

with open(f'Modelos/final_model.sav', 'rb') as f:
    final_old_model = pk.load(f)
hoy=datetime.today()

colormap = cm.LinearColormap(colors=['red', 'yellow', 'green'], vmin=0, vmax=1)
weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")

variables= ['avg_rebuy_months', 'avg_revisit_visit', 'ratio_compra', 'total_spent', 'median_income', 'acc_age', 'month_since_last_visit','month_since_last_offer', 'accountstatus', 'accountrating']
variables_ac=['avg_rebuy_months', 'avg_revisit_visit', 'ratio_compra', 'total_spent', 'median_income', 'acc_age', 'month_since_last_visit','month_since_last_offer', 'accountstatus', 'accountrating','account_id','state','offer_date','territory_name','acc_size','offer_day','total_offer','latitude','longitude','status']

prohibidos = ['NoOffer: Not Enough Scrap','Accepted','Rejected: Not enough scrap to sell','No:Offer Rejected offer too low']


@st.cache(allow_output_mutation=True)
def calcular_proba_noprospect(data,dates):
    """Calculate probability for existing accounts

    Args:
        data (DataFrame): Dataframe with the information of the accounts
        dates (list): List of dates

    Returns:
        DataFrame: DataFrame with account´s probabilities
    """
    firstDay = dates[0]
    lastDay = dates[1]
    delta = lastDay - firstDay
    probas=data[['id']]
    # Calculate probability for each day
    for i in range(delta.days + 1):
        day = firstDay + timedelta(days=i)
        weekDay=weekDays[day.weekday()]
        data['months_since_last_offer']=(pd.to_datetime(day)-pd.to_datetime(data['last_accepted_date']))/np.timedelta64(1, 'M')
        data['months_since_last_visit']=(pd.to_datetime(day)-pd.to_datetime(data['last_visit_date']))/np.timedelta64(1, 'M')
        X=data[['id','acc_age', 'avg_revisit_visit', 'months_since_last_offer','avg_rebuy_months', 'months_since_last_visit', 'total_spent','accountstatus','accountrating','rep_cluster','state_cluster']].set_index('id')
        X['accountstatus']=X['accountstatus'].fillna(100).astype(int).astype(str).replace("100","Other")
        X['accountrating']=X['accountrating'].fillna(100).astype(int).astype(str).replace("100","Other")
        X_dum=pd.get_dummies(X.rename(columns={'rep_cluster':'rep_group','state_cluster':'state_group'}))
        vars=['acc_age', 'avg_revisit_visit', 'months_since_last_offer','avg_rebuy_months', 'months_since_last_visit', 'total_spent','accountstatus_1', 'accountstatus_2','accountstatus_3', 'accountstatus_4', 'accountstatus_5','day_name_Friday', 'day_name_Monday','day_name_Thursday', 'day_name_Tuesday','day_name_Wednesday', 'accountrating_1','accountrating_2', 'accountrating_3', 'accountrating_4', 'rep_group_0','rep_group_1', 'rep_group_2', 'rep_group_3', 'rep_group_4','rep_group_5', 'rep_group_6', 'rep_group_Other', 'state_group_1','state_group_2', 'state_group_3', 'state_group_4']
        for col in vars:
            if col not in X_dum.columns:
                X_dum[col]=0
        X_dum=X_dum[vars]
        X_dum[['acc_age', 'avg_revisit_visit', 'months_since_last_offer','avg_rebuy_months', 'months_since_last_visit', 'total_spent']]=standard_model.transform(X_dum[['acc_age', 'avg_revisit_visit', 'months_since_last_offer','avg_rebuy_months', 'months_since_last_visit', 'total_spent']].fillna(X_dum[['acc_age', 'avg_revisit_visit', 'months_since_last_offer','avg_rebuy_months', 'months_since_last_visit', 'total_spent']].quantile(0.5)))
        if f'day_name_{weekDay}' in vars:
            X_dum[f'day_name_{weekDay}']=1
            X_dum = X_dum.loc[:,~X_dum.columns.duplicated()]
            X_dum[['0','1']]=final_old_model.predict_proba(X_dum)
            probas=probas.merge(X_dum.reset_index()[['id','1']].rename(columns={'1':f'{day}'}),how='left',on='id')
    #Get mean of probabilities
    probas.set_index('id',inplace=True)
    probas['1'] = probas.mean(numeric_only=True, axis=1)  
    datas=data.merge(probas.reset_index()[['id',"1"]],how='left',on='id')
    datas["Probability"]=np.round(datas["1"],2)
    return datas

def corregir_proba(proba,lv_outcome,lv_date,prospect):
    """Correct the probability if it is too soon to visit the account

    Args:
        proba (float): Buy Probability
        lv_outcome (string): Last visit outcome
        lv_date (datatime): last visit date
        prospect (int): Prospect

    Returns:
        float: Return the original probability if none of the conditions are satisfied, return -1 otherwise
    """
    x=proba
    if prospect==0:
        if lv_date<3 and lv_outcome =='NoOffer: Not Enough Scrap':
            x=-1
        elif lv_date<3 and lv_outcome =='Rejected: Not enough scrap to sell':
            x=-1
        elif lv_date<1 and lv_outcome =='NoOffer: Rejected offer to low':
            x=-1
        elif lv_date<5 and lv_outcome =='NoOffer: Committed to Competitor':
            x=-1
        elif lv_date<5 and lv_outcome =='NoOffer: Committed to competitor':
            x=-1
        elif lv_date<1.5 and lv_outcome =='NoOffer: Sends to corporate':
            x=-1
        elif lv_date<3 and lv_outcome =='NoOffer: Does Not Collect':
            x=-1
        elif lv_date<3 and lv_outcome =='Accepted':
            x=-1
        elif lv_date<2:
            x=-1
    else:
        if lv_date<1 and lv_outcome =='NoOffer: Not Enough Scrap':
            x=-1
        elif lv_date<1 and lv_outcome =='Rejected: Not enough scrap to sell':
            x=-1
        elif lv_date<1 and lv_outcome =='NoOffer: Rejected offer to low':
            x=-1
        elif lv_date<3 and lv_outcome =='NoOffer: Committed to Competitor':
            x=-1
        elif lv_date<3 and lv_outcome =='NoOffer: Committed to competitor':
            x=-1
        elif lv_date<1.5 and lv_outcome =='NoOffer: Sends to corporate':
            x=-1
        elif lv_date<3 and lv_outcome =='NoOffer: Does Not Collect':
            x=-1
        elif lv_date<3 and lv_outcome =='Accepted':
            x=-1
        elif lv_date<1:
            x=-1

    return x 


def get_center_latlong(df):
    """Calculate the center location of a dataframe that contains several locations

    Args:
        df (DataFrame): Dataframe with locations info. It has to contain latitude and longitude

    Returns:
       tuple: Latitude and Longitude center
    """
    # get the center of my map for plotting
    centerlat = (df['latitude'].max() + df['latitude'].min()) / 2
    centerlong = (df['longitude'].max() + df['longitude'].min()) / 2
    return centerlat, centerlong


def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    if pickle_it:
        try:
            object_to_download = pk.dumps(object_to_download)
        except pk.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            #object_to_download = object_to_download.to_csv(index=False)
            towrite = io.BytesIO()
            object_to_download = object_to_download.to_excel(towrite, encoding='utf-8', index=True, header=True)
            towrite.seek(0)

        else:
            object_to_download = json.dumps(object_to_download)

    try:
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(towrite.read()).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f"""
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: #e2141c;
                color: white;
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(0, 0, 0);
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: rgb(255, 255, 255);
                color: rgb(255, 255, 255);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(0, 0, 0);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">{button_text}</a><br></br>'

    st.markdown(dl_link, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def calcular_proba_prospect(data,dates):
    """Calculate probability for prospect accounts

    Args:
        data (DataFrame): Dataframe with the information of the accounts
        dates (list): List of dates

    Returns:
        DataFrame: DataFrame with account´s probabilities
    """

    firstDay = dates[0]
    lastDay = dates[1]
    delta = lastDay - firstDay
    probas=data[['id']]
    #First we calculate the buy probability of each day
    for i in range(delta.days + 1):
        day = firstDay + timedelta(days=i)
        data['months_between_firstvisit']=(day-pd.to_datetime(data['first_visit_date']).dt.date)/np.timedelta64(1, 'M')
        data['months_between_previousvisit']=(day-pd.to_datetime(data['last_visit_date']).dt.date)/np.timedelta64(1, 'M')
        
        X=data[['id','cum_prevvisits', 'months_between_firstvisit','months_between_previousvisit', 'avg_revisit_visit','rep_cluster', 'state_cluster']].set_index('id')
        X_dum=pd.get_dummies(X.rename(columns={'rep_cluster':'rep_group','state_cluster':'state_group'}))

        eliminar=[]
        for col in ['rep_group_Other','state_group_1', 'state_group_2','state_group_4', 'state_group_Other']:
            if col in X_dum.columns:
                X_dum=X_dum.drop(columns=[col])
        variables=['state_group_3','rep_group_0','rep_group_1','rep_group_2','rep_group_3','rep_group_4','rep_group_5','rep_group_6']
        for col in variables:
            if col not in X_dum.columns:
                X_dum[col]=0
        X_dum[['0','1']]=final_model.predict_proba(X_dum)
        probas=probas.merge(X_dum.reset_index()[['id','1']].rename(columns={'1':f'{day}'}),how='left',on='id')
    
    # Get mean of probabilities for each account
    probas.set_index('id',inplace=True)
    probas['1'] = probas.mean(numeric_only=True, axis=1)  
    datas=data.merge(probas.reset_index()[['id',"1"]],how='left',on='id')
    datas["Probability"]=np.round(datas["1"],2)
    return datas

@st.cache()
def find_k(max_value,data_geo):
    """This function calculate the optimal number of clusters for a dataframe 

    Args:
        max_value (int): Maximum number of clusters to try
        data_geo (DataFrame): Dataframe with latitude, longitude and probability

    Returns:
        int: Optimal number of clusters
    """
    dist_points_from_cluster_center = []
    K = range(1,max_value)
    for no_of_clusters in K:
        try:
            k_model = KMeans(n_clusters=no_of_clusters)
            k_model.fit(data_geo[['longitude','latitude','Probability']])
            dist_points_from_cluster_center.append(k_model.inertia_)
        except:
            pass
    
    def calc_distance(x1,y1,a,b,c):
        d= abs((a*x1+b*y1+c)) / (math.sqrt(a*a+b*b))
        return d
    
    a = dist_points_from_cluster_center[0]-dist_points_from_cluster_center[8]
    b = K[8]-K[0]
    c1 = K[0]*dist_points_from_cluster_center[8]
    c2 = K[8]*dist_points_from_cluster_center[0]
    c=c1-c2

    distance_of_point_from_line = []
    for k in range(9):
        distance_of_point_from_line.append(calc_distance(K[k],dist_points_from_cluster_center[k],a,b,c))
    
    return(distance_of_point_from_line.index(max(distance_of_point_from_line))+1)