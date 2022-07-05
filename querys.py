from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from datetime import datetime
import streamlit as st

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
BigQuery_client = bigquery.Client(credentials=credentials)
# THIS FILE PERFORMS THE ETL OF THE DATA HOSTED IN BIG QUERY AND DOWNLOADS THE FINAL FILE IN THE DATA FOLDER

group3_st=['IN','MI','MD','MA','OH','PA','IL','NJ','FL','TX','CA','GA']
group1_st=['WA','CO','NC','MN','NY','OR','AZ','VA']
group2_st=['SC','TN','UT','NV','AK','MO','LA','KY','DC','NE']
group4_st=['OK','RI','KS','CT','WI']

hoy=datetime.today()

def state_group(state):
    if state in group1_st:
        return "1"
    elif state in group2_st:
        return "2"
    elif state in group3_st:
        return "3"
    elif state in group4_st:
        return "4"
    else:
        return "Other"

group0=['Aaron Domke', 'Alex Aguilar', 'Andrew Rogers', 'Andrew Stein','Christiana Merdaa', 'Daniel Lombardo', 'David Gulakowski','Debra Morris', 'Demery Moody', 'Eran Even-Kesef', 'Gary Schwartz','James Loomis', 'Jason Chew', 'Jason Mankevich', 'Ladeen Mccray-Davis','Laura Rowe', 'Lucy Shockney', 'Nicholas Wilsey', 'Patrick Markowski','Paul Kanengiser', 'Ricky Rodemacher', 'Robert Wolfe', 'Ryan Weaver','Shannon Rhodes', 'Shay Bar Nissim', 'Stephen Blanchfill', 'Troy C-Rep','Vanessa Berrueta', 'Vicky Barnett', 'Whitney Ross']
group1=['Amy Teague', 'Brian Hunter', 'Brian Sundberg', 'Brien Hebb','Carol Monaco', 'Crystal Benner', 'Dawn Devereux', 'Eric Moskow','Farah Elturk', 'Fernando Andrews', 'Gene Spinner', 'Jim Ryan','Kim Stotsky', 'Ray Patterson', 'Rudi Lovell', 'Warren Sabo']
group2=['Elizabeth Reynoso', 'Judd Stewart', 'Lisa Buzonik']
group3=['Brent Hicks', 'Chris Charbonneau', 'Jim Harness', 'Joseph Bonham']
group4=['Alan Veronese', 'Anthony Canton', 'Brett Buckingham','Christi Finnell', 'Dawn Wagner', 'Jon Dowling', 'Mark Howard','Misty Boodt', 'Nancy Braier', 'Nicole Hernandez', 'Robin Harp','Robin Reinhold', 'Scott Watkins', 'Tom Leverone']
group5=['Sammie Harris']
group6=['Alex Atanassov', 'Karen Embly']
def rep_cluster(name):
    if name in group0:
        return "0"
    elif name in group1:
        return "1"
    elif name in group2:
        return "2"
    elif name in group3:
        return "3"
    elif name in group4:
        return "4"
    elif name in group5:
        return "5"
    elif name in group6:
        return "6"
    else:
        return "Other"

query="""
WITH avg_revisit_visit AS
(
    SELECT account_id as id, AVG(DATE_DIFF(DATE(offer_date), DATE(PriorDate),MONTH)) as avg_revisit_visit, CAST(MIN(offer_date) as DATE) as first_visit_date,CAST(MAX(offer_date) as DATE) as last_visit_date
    FROM (
    SELECT
        account_id,
        offer_date,
        status,
        LAG(offer_date) OVER (PARTITION BY account_id ORDER BY offer_date) as PriorDate
    FROM `star-big-data.star_us_rds.offers`
    WHERE EXTRACT(YEAR FROM offer_date) > 2018)
    WHERE status <> 'NoVisit'
    GROUP BY account_id
    ORDER BY account_id 
),
avg_rebuy_months AS
(
    SELECT account_id as id, AVG(DATE_DIFF(DATE(offer_date), DATE(PriorDate),MONTH)) as avg_rebuy_months
    FROM (
    SELECT
        account_id,
        offer_date,
        LAG(offer_date) OVER (PARTITION BY account_id ORDER BY offer_date) as PriorDate
    FROM `star-big-data.star_us_rds.offers`
    WHERE EXTRACT(YEAR FROM offer_date) > 2018 AND status='Accepted') 
    GROUP BY account_id
    ORDER BY account_id
),
acc_age AS
(
    SELECT account_id as id,DATE_DIFF(DATE(CURRENT_DATE()), DATE(first_order_date),MONTH) as acc_age
    FROM(
    select account_id, MIN(offer_date) as first_order_date
    from `star-big-data.star_us_rds.offers`
    where EXTRACT(YEAR FROM offer_date) > 2018
    group by account_id
    order by account_id)
),
total_spent AS
(
    SELECT account_id as id, IFNULL(SUM(total_offer), 0 ) as total_spent
    FROM `star-big-data.star_us_rds.offers`
    where EXTRACT(YEAR FROM offer_date) > 2018 AND status='Accepted'
    group by account_id
    order by account_id
),
accounts AS
(
    SELECT c.id,c.specialty,c.accountstatus,c.phone,c.territory_id,c.last_visit_outcome,c.lastoffer_id,c.zip_left,c.last_offer_date,c.last_accepted_date,c.place_id,t.name as territory_name,c.accountrating,c.rep_id,c.practice_name,c.last_visit_date,p.city,p.longitude,p.latitude,p.county,p.state,r.name,r.inactive,
    CASE
        WHEN c.accountstatus = 0 then 'Lead'
        WHEN c.accountstatus = 1 then 'Prospect'
        WHEN c.accountstatus = 2 then 'Opportunity'
        WHEN c.accountstatus = 3 then 'Lost'
        WHEN c.accountstatus = 4 then 'Deleted'
        WHEN c.accountstatus = 5 then 'NPS'
    END as status_txt,
    CASE
        WHEN c.accountrating = 0 then 'Best'
        WHEN c.accountrating = 1 then 'Good'
        WHEN c.accountrating = 2 then 'Fair'
        WHEN c.accountrating = 3 then 'Poor'
        WHEN c.accountrating = 4 then 'Unqualified'
    END as rating_txt,
    FROM `star-big-data.star_us_rds.accounts` c
    LEFT JOIN `star-big-data.star_us_rds.places` p ON c.place_id=p.id 
    LEFT JOIN `star-big-data.star_us_rds.reps` r ON c.rep_id = r.id
    LEFT JOIN `star-big-data.star_us_rds.Territory` t ON c.territory_id = t.id
    WHERE c.accountstatus IS NOT NULL and c.accountrating IS NOT NULL
),offers AS
(
    SELECT o.id as offer_id, o.longitude,o.latitude,o.outcome_details,o.status,o.rep_id,r.name,r.inactive,o.total_offer,o.account_id,CAST(o.offer_date as DATE) as offer_date,  o.territory_id,t.name as territory_name,
    CASE
        WHEN status='Accepted' then 1
        WHEN status<>'Accepted' then 0
    END as buy,1 as visit , CAST(lag(o.offer_date) over (partition by o.account_id order by o.offer_date) as DATE) AS previous_visit_date
    FROM `star-big-data.star_us_rds.offers` o
    LEFT JOIN `star-big-data.star_us_rds.reps` r
    ON o.rep_id = r.id
    LEFT JOIN `star-big-data.star_us_rds.Territory` t
    ON o.territory_id = t.id
    WHERE EXTRACT(YEAR FROM offer_date) > 2018 and o.offer_date IS NOT NULL and o.account_id IS NOT NULL and r.name <> 'General Pot' and r.name <> 'Nancy Braier' and o.status <> 'NoVisit'
    ORDER BY o.offer_date ASC
),cummulative AS
(
    SELECT offer_id ,account_id,SUM(visit) OVER (PARTITION BY account_id ORDER BY offer_id) - 1 as cum_visits, SUM(buy) OVER (PARTITION BY account_id ORDER BY offer_id) - buy as cum_buys
    FROM offers     
    WHERE account_id IS NOT NULL
),cummulative_data AS
(
    SELECT account_id, max(cum_visits) as cum_prevvisits,max(cum_buys) as cum_prevbuys
    FROM cummulative
    GROUP BY account_id
),
acc_statusrating as
(
    select id , CONCAT(status_txt,' ',rating_txt) as statusandrating
    FROM accounts
)
SELECT a.id, a.specialty, a.accountstatus, a.phone, a.territory_id, a.last_visit_outcome, a.lastoffer_id,a.zip_left,CAST(a.last_offer_date as DATE) as last_offer_date,CAST(a.last_accepted_date as DATE) as last_accepted_date,a.place_id, a.territory_name,a.accountrating,a.rep_id,a.practice_name,CAST(a.last_visit_date as DATE) as last_visit_date,a.city,a.longitude,a.latitude,a.county,a.state,a.name,a.inactive,ts.total_spent,aa.acc_age,arm.avg_rebuy_months,arv.avg_revisit_visit, arv.first_visit_date, DATE_DIFF(DATE(CURRENT_DATE()), DATE(arv.first_visit_date),MONTH) as months_between_firstvisit,
CASE
    WHEN DATE_DIFF(DATE(CURRENT_DATE()), DATE(a.last_accepted_date),MONTH) < 24 then DATE_DIFF(DATE(CURRENT_DATE()), DATE(a.last_accepted_date),MONTH)
    WHEN DATE_DIFF(DATE(CURRENT_DATE()), DATE(a.last_accepted_date),MONTH) >= 24 then NULL
END as months_since_last_offer, 
CASE
    WHEN DATE_DIFF(DATE(CURRENT_DATE()), DATE(a.last_visit_date),MONTH) < 24 then DATE_DIFF(DATE(CURRENT_DATE()), DATE(a.last_visit_date),MONTH) 
    WHEN DATE_DIFF(DATE(CURRENT_DATE()), DATE(a.last_visit_date),MONTH) >= 24 then NULL 
END as months_since_last_visit, asr.statusandrating, cd.cum_prevvisits, cd.cum_prevbuys,
CASE
    WHEN (cum_prevbuys=0 or cum_prevbuys IS NULL) or DATE_DIFF(DATE(CURRENT_DATE()), DATE(a.last_visit_date),MONTH) >= 24 then 1
    ELSE 0
END as prospect
FROM accounts a
LEFT JOIN acc_age aa ON aa.id=a.id
LEFT JOIN avg_rebuy_months arm ON arm.id=a.id
LEFT JOIN avg_revisit_visit arv ON arv.id=a.id
LEFT JOIN total_spent ts ON a.id=ts.id
LEFT JOIN acc_statusrating asr ON asr.id=a.id
LEFT JOIN cummulative_data cd ON cd.account_id = a.id
WHERE( a.specialty<>'Private Buys' or a.specialty IS NULL)  and statusandrating <> 'Deleted Poor' and statusandrating <> 'NPS Poor' and statusandrating <> 'Lead Unqualified' and statusandrating <> 'Lost Poor' 
"""

@st.cache(ttl=28800,allow_output_mutation=True)
def upload_data():
    """It connects to BigQuery and get all the data needed

    Returns:
        DataFrame: DataFrame with all the data
    """
    df = (BigQuery_client.query(query).result().to_dataframe(create_bqstorage_client=True,))
    reps = pd.read_excel('data/repHouses.xlsx',engine="openpyxl").rename(columns={'lat':'rep_lat','lon':'rep_lon'})
    df['state_cluster']=df['state'].apply(state_group)
    df['rep_cluster']=df["name"].apply(rep_cluster)
    df = df.merge(reps[['name','rep_lat','rep_lon']],how='left',on='name')
    return df
