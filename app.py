import streamlit as st
import pandas as pd
from streamlit_folium import folium_static 
import folium
from datetime import timedelta
import numpy as np
from sklearn.neighbors import DistanceMetric
from funciones import *
import random 
from querys import *
import matplotlib.pyplot as plt
from google_route import *

# Logos y tipografia se carga
random.seed(10)
st.set_page_config(page_title="X-SDR",page_icon='Imagenes/starlogoCrop.png')
st.sidebar.image("Imagenes/starlogo.png", use_column_width=True)
st.sidebar.header('Star App by Xpecta')
st.sidebar.subheader('Prospecting Edition')

# THIS FILE DISPLAYS THE WEB PAGE USING THE STREAMLIT PACKAGE. TO SEE THE WEBPAGE IT IS NECESSARY TO 
# INSTALL STREAMLIT PACKAGE AND RUN "streamlit run app.py" ON TERMINAL
dataframe=upload_data()
st.write(
    """
    # Accounts Detail
    > A map with available accounts will be displayed when you select a territory. A green point is an account with high probability to make a buy, while red is low probability.
    """
)

#SIDEBAR



check_account=st.sidebar.checkbox("Search for an account")
select_cluster='None'
# If the person wnat to center a route in one account or just one to see an account probability, the checkbox can be selected and the route is going to be centered in that account
if check_account:
    id_cuenta=st.sidebar.selectbox("Account ID",['Select one account ID']+dataframe['id'].astype(int).unique().tolist())
    rep='All'
    if id_cuenta != 'Select one account ID':
        nombre=dataframe.loc[dataframe['id']==id_cuenta,'name'].values[0]
        rep=st.sidebar.selectbox("Choose a Rep",[nombre])
else:
    id_cuenta='Select one account ID'
    rep=st.sidebar.selectbox("Choose a Rep",['All']+list(dataframe.loc[dataframe['inactive']==False,'name'].sort_values().unique()))

#In this section the person can filter the dataframe by rep, zip_code,state,territory_name,city and county
if rep != 'All':
    dataframe = dataframe[dataframe['name']==rep]
    rep_house = [dataframe['rep_lat'].unique()[0], dataframe['rep_lon'].unique()[0]]
    rephouse_name=f"{dataframe['name'].unique()[0]}'s house"
zip_code=st.sidebar.selectbox("Choose a Zip Code",['All']+list(dataframe['zip_left']))
if zip_code!='All':
    dataframe = dataframe[dataframe['zip_left']==zip_code]

state=st.sidebar.selectbox("Choose a State",['All']+list(dataframe['state'].unique()))
if state != 'All':
    dataframe = dataframe[dataframe['state']==state]

territory_name=st.sidebar.selectbox("Choose a Territory Name",['All']+list(dataframe['territory_name'].unique()))
if territory_name!='All':
    dataframe = dataframe[dataframe['territory_name']==territory_name]

city=st.sidebar.multiselect("Choose a City",['All']+sorted([str(x) for x in dataframe['city'].unique()]))
if city :
    dataframe = dataframe[dataframe['city'].isin(city)]

county=st.sidebar.selectbox("Choose a County",['All']+list(dataframe['county'].unique()))
if county!='All':
    dataframe = dataframe[dataframe['county']==county]

dates=st.sidebar.date_input("Visit range",[],min_value=hoy)
data=dataframe.copy()

# If dates were selected, the probabilities are calculated for prospect and no prospect account with the two differente models
if len(dates)==2:

    try:
        datas1=calcular_proba_prospect(data[data['prospect']==1],dates)
        datas1['Probability']=datas1['Probability'].replace(0.43,0)
    except:
        datas1=pd.DataFrame()
        st.warning("There are not prospect accounts.")
    try:
        datas2=calcular_proba_noprospect(data[data['prospect']==0],dates)
    except:
        datas2=pd.DataFrame()
        st.warning("There are not existing accounts.")

    datas=datas1.append(datas2,ignore_index=True)
    
    datas['last_visit_date']=pd.to_datetime(datas['last_visit_date']).dt.date
    datas['months_since_last_visit']=(pd.to_datetime(hoy.date())-pd.to_datetime(datas['last_visit_date']))/np.timedelta64(1, 'M')
    datas["Probability"]=datas[['Probability','last_visit_outcome','months_since_last_visit','prospect']].apply(lambda x: corregir_proba(x[0],x[1],x[2],x[3]),axis=1)
    outcomes_tiempo=['NoOffer: Not Enough Scrap','Rejected: Not enough scrap to sell','NoOffer: Rejected offer to low','NoOffer: Committed to Competitor','NoOffer: Committed to competitor','NoOffer: Sends to corporate','NoOffer: Does Not Collect','Accepted']
    
    datas['Buy Probability']=datas['Probability'].apply(lambda x: 'Low' if x<0.3 else ('Medium' if x<0.6 else 'High') )
    #datas['Buy Probability']=np.round(datas['Probability']*100,1).astype(str)+'%'
    datas['last_accepted_date']=datas['last_accepted_date'].apply(lambda x: str(x)[:10] if x!=np.nan else '')
   
#MAPS AND TABLES

#MAP
    if (territory_name!='All') | (state!='All') :
        data_geo=datas.dropna(subset=['latitude','longitude'])
        center = get_center_latlong(data_geo)

        #Clustering
        k = find_k(10,data_geo)
        k_model=KMeans(n_clusters=k).fit(data_geo[['longitude','latitude','Probability']])

        data_geo['Cluster']=k_model.predict(data_geo[['longitude','latitude','Probability']]).astype(int)
        data_geo['Cluster label']= 'Cluster '+ (data_geo['Cluster']).astype(str)
        try:
            st.write("## Map with probabilities")
            map1 = folium.Map(location=center, zoom_start=10,tiles='Stamen Toner')
            for i in range(len(data_geo)):
                prac_name=data_geo.iloc[i][['practice_name','Buy Probability']].values[0]
                prob=data_geo.iloc[i][['practice_name','Buy Probability']].values[1]

                html = f"Practice Name: {prac_name} <br>Buy Probability: {prob}"
                iframe = folium.IFrame(html)
                popup = folium.Popup(iframe,min_width=180,max_width=190)
                folium.Circle(location=[data_geo.iloc[i]['latitude'], data_geo.iloc[i]['longitude']],popup=popup,radius=150,fill=True,color=colormap(data_geo.iloc[i]['Probability']),fill_opacity=0.8).add_to(map1)
            #Add rep house
            try:
                
                html = f"Practice Name: {rephouse_name} "
                iframe = folium.IFrame(html)
                popup = folium.Popup(iframe,min_width=180,max_width=190)
                folium.Circle(location=[data_geo['rep_lat'].unique()[0], data_geo['rep_lon'].unique()[0]],popup=popup,radius=150,fill=True,color="blue").add_to(map1)
            except:
                st.error("The house of the rep is not in this state")
            folium_static(map1)

            st.write("## Map with clusters")
            st.write(f"**Optimal Number of Clusters:** {k}")

            map2 = folium.Map(location=center, zoom_start=10,tiles='Stamen Toner')
            rainbow = ['red','yellow','blue','white','purple','pink']

            #Para cada cluster se le asigna y se plotea en el mapa
            for cluster in range(0,k): 
                group = folium.FeatureGroup(name='<span style=\\"color: {0};\\">{1}</span>'.format(rainbow[cluster-1],cluster))
                for lat, lon,prob, label in zip(data_geo['latitude'], data_geo['longitude'],data_geo['Probability'], data_geo['Cluster']):
                    if int(label) == cluster: 
                        label = folium.Popup('Cluster ' + str(cluster) + ' Probability '+ str(prob), parse_html=True)
                        folium.CircleMarker(
                            (lat, lon),
                            radius=2,
                            popup=label,
                            color=rainbow[cluster-1],
                            fill=True,
                            fill_color=rainbow[cluster-1],
                            fill_opacity=0.7).add_to(group)
                group.add_to(map2)
            folium.map.LayerControl('topright', collapsed=False).add_to(map2)
            folium_static(map2)

            st.write("### Cluster Statistics")
            # Calculate statistics for each cluster and displayes a table
            resumen_stats=pd.DataFrame(index=sorted(data_geo['Cluster label'].unique()),columns=['Not visited (Last 2 years)','Never Bought (Last 2 years)','Avg Buy','Total Accounts','Territory not covered (%)'])
            for cluster in resumen_stats.index:
                resumen_stats.loc[cluster,'Not visited (Last 2 years)']=data_geo.loc[(data_geo['Cluster label']==cluster)&((data_geo['cum_prevvisits']==0)|(data_geo['months_since_last_visit']>24))].shape[0]
                resumen_stats.loc[cluster,'Never Bought (Last 2 years)']=int(data_geo.loc[(data_geo['Cluster label']==cluster)&((data_geo['cum_prevbuys']==0)|(data_geo['months_since_last_offer']>24))].shape[0])
                resumen_stats.loc[cluster,'Total Accounts']=data_geo.loc[(data_geo['Cluster label']==cluster)].shape[0]
                resumen_stats.loc[cluster,'Avg Buy']=data_geo.loc[(data_geo['Cluster label']==cluster),'total_spent'].sum()/data_geo.loc[(data_geo['Cluster label']==cluster),'cum_prevbuys'].sum()
                resumen_stats.loc[cluster,'Territory not covered (%)'] = (resumen_stats.loc[cluster,'Not visited (Last 2 years)']/data_geo[data_geo['Cluster label']==cluster].shape[0])*100
            st.write(resumen_stats)
            select_cluster=st.selectbox("Select the cluster that you wan to cover",['None']+sorted(data_geo['Cluster label'].unique()))
        except Exception as e:
            st.warning("There are no accounts with location info")
            st.warning(str(e))
            select_cluster='None'
        
        
    else:
        st.warning("To view the probability map please select a state or territory.")
    
    st.write("""
        ### Accounts List
    """)
    sort= st.radio(
        "Sort probability values:",
        ('Descending','Ascending')
    )
    if select_cluster!='None':
        datas=datas[datas['id'].isin(data_geo.loc[data_geo['Cluster label']==select_cluster,'id'].to_list())]
    datas[['last_accepted_date','last_visit_date']]=datas[['last_accepted_date','last_visit_date']].fillna('')
    datas['last_visit_outcome']=datas['last_visit_outcome'].fillna('')
    datas['last_visit_date']=datas['last_visit_date'].fillna('')
    datas['last_accepted_date']=datas['last_accepted_date'].replace('None',np.nan)
    datas['last_accepted_date']=datas['last_accepted_date'].fillna('')
    
    #TABLES 
    data_visual=datas[['id','name','practice_name','Buy Probability','city','zip_left','phone','last_accepted_date','last_visit_date','last_visit_outcome','statusandrating','Probability','prospect']].rename(columns={'practice_name':'Practice Name','zip_left':'Zip','last_accepted_date':'Last Buy Date','last_visit_date':'Last Visit Date','last_visit_outcome':'LV Outcome','statusandrating':'Status - Rating'})
    if id_cuenta != 'Select one account ID':
        st.write("**Target account:**")
        st.write(data_visual[data_visual['id']==id_cuenta].astype(str)).drop(columns='Probability')
    if sort=='Ascending':
        data_visual=data_visual.sort_values(by='Probability',ascending=True)
        st.write(data_visual.reset_index(drop=True).drop(columns='Probability').fillna('').astype(str))
    elif sort=='Descending':
        data_visual=data_visual.sort_values(by='Probability',ascending=False)
        st.write(data_visual.reset_index(drop=True).drop(columns='Probability').fillna('').astype(str))
    download_button(data_visual.astype(str), f'accountsProbabilities.xlsx', f'Download Table', pickle_it=False)

    # SUGGESTED DRIVE LIST COMPUTATION
    dist = DistanceMetric.get_metric('haversine')
    distancia_maxima = st.slider("Select the maximum desired distance between accounts (miles)",0,100,20)

    
    temporal_distancias=datas.copy()
    st.write("""
        ### Suggested Drive List
    """)
    firstDay = dates[0]
    lastDay = dates[1]
    delta = lastDay - firstDay
    probas=data[['id']]

    # We create a drive list for each selected day
    for i in range(delta.days + 1):
        day = firstDay + timedelta(days=i)
        weekDay=weekDays[day.weekday()]
        if weekDay in ['Monday','Tuesday','Thursday','Wednesday','Friday']:
            st.write(
                """**{} - {}**""".format(weekDay,day)
            )

            # First we select the center of the drive list,  this should be an account with more than 5% Buy probability
            distan_max=temporal_distancias[temporal_distancias['Probability']>0.05].sort_values(by='Probability',ascending=False)[['id','practice_name','prospect','longitude','latitude']].set_index('id').dropna()
            if distan_max.shape[0]==0:
                distan_max=temporal_distancias[temporal_distancias['Probability']>=0].sort_values(by='Probability',ascending=False)[['id','practice_name','prospect','longitude','latitude']].set_index('id').dropna()

            #If the user is looking for a specific account we choose this one to be the center 
            if id_cuenta != 'Select one account ID':
                if id_cuenta not in distan_max.index:
                    distan_max=distan_max.append(datas.loc[datas['id']==id_cuenta,['id','practice_name','prospect','longitude','latitude']].set_index('id'))
            distan_max['latitude'] = np.radians(distan_max['latitude'])
            distan_max['longitude'] = np.radians(distan_max['longitude'])
            try:
                try:
                    #First we are going to fill the drive list with the existing accounts. cuentas_distancia contains all the existing accounts
                    cuentas_distancia = distan_max.loc[distan_max['prospect']==0,['latitude','longitude']]
                    if id_cuenta != 'Select one account ID':
                        if id_cuenta not in cuentas_distancia.index:
                            cuentas_distancia = cuentas_distancia.append(distan_max.loc[id_cuenta,['latitude','longitude']])
                    #The distance matriz is created
                    distance_matrix=pd.DataFrame(dist.pairwise(cuentas_distancia.to_numpy())*6373,  columns=distan_max[distan_max['prospect']==0].index, index=distan_max[distan_max['prospect']==0].index)
                except:
                    cuentas_distancia = distan_max[['latitude','longitude']]
                    if id_cuenta != 'Select one account ID':
                        if id_cuenta not in cuentas_distancia.index:
                            cuentas_distancia = cuentas_distancia.append(distan_max.loc[id_cuenta,['latitude','longitude']])
                    distance_matrix=pd.DataFrame(dist.pairwise(cuentas_distancia)*6373,  columns=distan_max.index, index=distan_max.index)
                #We set as the center account the one with more accounts with a distance less than distancia_maxima
                distance_matrixx=distance_matrix.apply(lambda x: np.where(x < distancia_maxima,1,0))
                if id_cuenta != 'Select one account ID':
                    idx_maximo=id_cuenta
                else:
                    idx_maximo=distance_matrixx.sum().idxmax()
                
                center_acc={'id':idx_maximo,'practice_name': temporal_distancias.loc[temporal_distancias['id']==idx_maximo,'practice_name'],'longitude': temporal_distancias.loc[temporal_distancias['id']==idx_maximo,'longitude'].values[0],'latitude': temporal_distancias.loc[temporal_distancias['id']==idx_maximo,'latitude'].values[0]}
                #Then we create the dataframe with the new accounts
                distan_news=temporal_distancias.loc[(temporal_distancias['prospect']==1)&(temporal_distancias['Probability']>=0),['id','practice_name','longitude','latitude']]
                
                try:
                    distan_news=distan_news.append(center_acc,ignore_index=True).set_index('id').dropna()
                    distan_news['latitude'] = np.radians(distan_news['latitude'])
                    distan_news['longitude'] = np.radians(distan_news['longitude'])
                    #This is the distance matriz of the new accounts
                    distance_matrix_news=pd.DataFrame(dist.pairwise(distan_news[['latitude','longitude']].to_numpy())*6373,  columns=distan_news.index, index=distan_news.index)
                    #We select up to 19 existing accounts that are near the center and then we select the remaining of new accounts
                    idx_max=distance_matrix.loc[distance_matrix[idx_maximo]<=distancia_maxima,idx_maximo].sort_values().head(19).index.tolist()
                    cant_nuevas = 25-len(idx_max)+1
                    idx_new=distance_matrix_news.loc[distance_matrix_news[idx_maximo]<distancia_maxima,idx_maximo].sort_values().iloc[1:cant_nuevas].index.tolist()
                    indices=idx_new+idx_max
                except Exception as e:
                    print('No hay suficientes prospect')
                    indices=idx_max
                
                drive_list=datas.sort_values(by='Probability',ascending=False).loc[datas['id'].isin(indices),['id','practice_name','Buy Probability','last_visit_outcome','city','zip_left','phone','last_accepted_date','last_visit_date','latitude','longitude','Probability']].rename(columns={'practice_name':'Practice Name','zip_left':'Zip','last_accepted_date':'Last Buy Date','last_visit_date':'Last Visit Date'}).reset_index(drop=True)
                drive_list['Last Buy Date']=drive_list['Last Buy Date'].replace('nan','')
                
                drive_list.index=drive_list.index+1
                drive_list.loc[0]=[0,rephouse_name,'','','','','','','',rep_house[0],rep_house[1],0]
                drive_list=drive_list.sort_index()

                
            except Exception as e:
                st.error(f'There are no enough high probability accounts to visit this day. \n {str(e)}')
            try:
                data = create_google_data(drive_list)
                solution,manager,routing = solve_route(data)
                if solution:
                    f = print_solution(data,manager,routing,solution,drive_list[['id','Practice Name','Buy Probability','city','Zip','phone','last_visit_outcome','Last Buy Date','Last Visit Date']])
                    st.write(f.astype(str))
                    download_button(f, f'accountsProbabilities.xlsx', f'Download Suggested Drive List', pickle_it=False)
                else:
                    st.write("Error happened while creating the route")
            except:
                st.write(drive_list.astype(str))
                download_button(drive_list, f'accountsProbabilities.xlsx', f'Download Suggested Drive List', pickle_it=False)
            #st.write(drive_list.drop(columns=['latitude','longitude','Probability']).astype(str))
            
            
            temporal_distancias.drop(temporal_distancias[temporal_distancias['id'].isin(indices)].index,inplace=True)

else:
    st.error("Select a date range to view the probabilities.")