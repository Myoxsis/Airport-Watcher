#%%

import pandas as pd
import json
import plotly.graph_objects as go
import os
import numpy as np
from datetime import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns

#%%

with open('aircraft_master_data.json') as file:
        aircraft_type_list = json.load(file)

#%%

def check_duplicated(df, col):
    ids = df[col]
    print(df[ids.isin(ids[ids.duplicated()])].sort_values(col))

def get_flight_info(x):
    flight_id = x['flight']['identification']['id']
    flight_row = x['flight']['identification']['row']
    flight_name = x['flight']['identification']['number']['default']

    l_flight_id.append(flight_id)
    l_flight_row.append(flight_row)
    l_flight_name.append(flight_name)

def get_airport_info(airport_data, destination_airport):
    if airport_data is not None:
        airport_dep_code = airport_data['flight']['airport']['origin']['code']['iata']
        airport_dep_name = airport_data['flight']['airport']['origin']['name']
    else:
        airport_dep_code = "N/A"
        airport_dep_name = "N/A"

    airport_arr_code = destination_airport
    travel_segment = airport_dep_code + " - " + airport_arr_code

    if destination_airport == 'CDG':
        airport_arr_name = "Paris Charles De Gaulle Airport"
    elif destination_airport == 'ORY':
        airport_arr_name = "Paris Orly"
    elif destination_airport == 'LBG':
        airport_arr_name = "Paris Le Bourget"
    else:
        airport_arr_name = "Unknown destination airport"

    l_travel_segment.append(travel_segment)
    l_airport_departure.append(airport_dep_name)
    l_airport_arrival.append(airport_arr_name)

def get_airline_info(x):
    try:
        airline_name = scope['flight']['airline']['name']
    except:
        airline_name = "N/A"

    l_airline_name.append(airline_name)

def get_time_info(x):
    sdt_i = x['flight']['time']['scheduled']['departure']
    sat_i = x['flight']['time']['scheduled']['arrival']
    rdt_i = x['flight']['time']['real']['departure']
    rat_i = x['flight']['time']['real']['arrival']
    edt_i = x['flight']['time']['estimated']['departure']
    eat_i = x['flight']['time']['estimated']['arrival']
    
    sdt_i_l.append(sdt_i)
    sat_i_l.append(sat_i)
    rdt_i_l.append(rdt_i)
    rat_i_l.append(rat_i)
    edt_i_l.append(edt_i)
    eat_i_l.append(eat_i)

def timestamp_to_datetime(x):
    try:
        r = datetime.fromtimestamp(x)
    except:
        r = np.nan
    return r

def check_unknown_aircraft_type(x):
    tmp = x.loc[x['aircraft_fam_u'] == 'Not Found', :]
    r = tmp['aircraft_type_code'].value_counts().index.to_list()
    return r

def get_aircraft_info(aircraft_typ):
        try:
            d = aircraft_type_list[aircraft_typ]['desc']
        except:
            d = "Not Found"
        try:
            f = aircraft_type_list[aircraft_typ]['fam']
        except:
            f = "Not Found"
        return d, f

#%%

li = []

for file_location in os.listdir('./data'):
    print(file_location)
    with open(f'./data/{file_location}') as file:
        data = json.load(file)
        DESTINATION_AIRPORT = data['result']['request']['code']

    l_flight_id, l_flight_name, l_flight_row, l_airline_name, l_aircraft_type, l_aircraft_type_code, l_aircraft_registration, l_travel_segment, l_airport_departure, l_airport_arrival = ([] for i in range(10))
    sdt_i_l, sat_i_l, rdt_i_l, rat_i_l, edt_i_l, eat_i_l = ([] for i in range(6))
    lists_l = l_flight_id, l_flight_name, l_flight_row, l_airline_name, l_travel_segment, l_airport_departure, l_airport_arrival, l_aircraft_type, l_aircraft_type_code, l_aircraft_registration, sdt_i_l, sat_i_l, rdt_i_l, rat_i_l, edt_i_l, eat_i_l
    list_colnames = ["flight_id", "flight_name", "flight_row", "Airline", "segment", "airport_departure", "airport_arrival", "aircraft_type", "aircraft_type_code", "aircraft_reg", "sdt", "sat", "rdt", "rat", "edt", "eat"]


    full_scope = data['result']['response']['airport']['pluginData']['schedule']['arrivals']['data']

    for scope in full_scope:
    
        get_flight_info(scope)
        get_airport_info(scope, DESTINATION_AIRPORT)
        get_airline_info(scope)
        get_time_info(scope)

        try:
            aircraft_type = scope['flight']['aircraft']['model']['text']
        except:
            aircraft_type = "N/A"
        try:
            aircraft_type_code = scope['flight']['aircraft']['model']['code']
        except:
            aircraft_type_code = "N/A"
        try:
            aircraft_registration = scope['flight']['aircraft']['registration']
        except:
            aircraft_registration = "N/A"
        try:
            aircraft_msn = scope['flight']['aircraft']['serialNo']
        except:
            aircraft_msn = "N/A"
        
        l_aircraft_type.append(aircraft_type)
        l_aircraft_type_code.append(aircraft_type_code)
        l_aircraft_registration.append(aircraft_registration)
        
    df = pd.DataFrame(lists_l).T
    df.columns = list_colnames
    li.append(df)

master = pd.concat(li, axis=0, ignore_index=True)

master['sat_d'] = master['sat'].apply(lambda x : timestamp_to_datetime(x))
master['sdt_d'] = master['sdt'].apply(lambda x : timestamp_to_datetime(x))
master['rat_d'] = master['rat'].apply(lambda x : timestamp_to_datetime(x))
master['rdt_d'] = master['rdt'].apply(lambda x : timestamp_to_datetime(x))
master['eat_d'] = master['eat'].apply(lambda x : timestamp_to_datetime(x))
master['edt_d'] = master['edt'].apply(lambda x : timestamp_to_datetime(x))

master['comments'] = master["Airline"].apply(lambda x : re.findall(r'\((.*?)\)',x))
master['comments'] = master['comments'].apply(lambda x : x[0] if len(x) > 0 else "")
master['comments'] = master['comments'].apply(lambda x : x.upper())

#%%

# drop duplicates

ids = master["flight_row"]
dubs = master[ids.isin(ids[ids.duplicated()])].sort_values("flight_row")
master['flight_id'] = master['flight_id'].fillna(0)

for id in dubs['flight_row'].unique():
    t_dubs = master.loc[master['flight_row'] == id, :]
    idx_to_delete = t_dubs.loc[t_dubs['flight_id'] == 0, :].index
    master = master.drop(index=idx_to_delete)

master.reset_index(drop=True, inplace=True)

# %%

master['aircraft_model_u'], master['aircraft_fam_u'] = zip(*master['aircraft_type_code'].apply(lambda x : get_aircraft_info(x)))

# %%

master['aircraft_fam_u'].value_counts()

# %%

check_unknown_aircraft_type(master)


# %%

master.head()

#%%

#master.to_csv('INITIAL_BDD_INTEGRATED_NO_CLEANSING.csv')

#%%

master['SAT_DATE'] = master['sat_d'].dt.date
master['SAT_DAY'] = master['sat_d'].dt.day_name()
master['SAT_HOUR'] = master['sat_d'].dt.hour

hr_pa = master.groupby(['SAT_HOUR', 'SAT_DAY'])['flight_row'].count()
hr_pa = hr_pa.unstack(level=0)
hr_pa = hr_pa.fillna(0)

f, ax = plt.subplots(figsize=(10, 6))
f.set_facecolor("white")

reorderlist = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.heatmap(hr_pa.reindex(reorderlist), cmap='YlOrBr')

plt.xlabel('Hour')
plt.ylabel('Day')

plt.title('Heatmap of Landings per day')

#%%

plt.plot(master[''])

#%%

master.head()
# %%

master.loc[master['aircraft_fam_u'] == "Not Found", :]

#%%

t1 = master.loc[master['aircraft_fam_u'] == "Not Found", :]
for ac_typ in t1['aircraft_type_code'].unique():
    t2 = t1.loc[t1['aircraft_type_code'] == ac_typ,:]
    print(ac_typ)
    print(t2['aircraft_type'].unique())

#%%


#/////////////
master.loc[master['aircraft_type_code'] == "BE40", :]


#////////////







# %%

master.loc[master['aircraft_reg'].str.startswith("F-HTY") == True, :]

# %%

# %%

# EDA

fig = plt.figure(figsize = (10, 5))

plt.bar(master['airport_arrival'].value_counts().index, 
        master['airport_arrival'].value_counts().values)
 
plt.xlabel("Airport")
plt.ylabel("No. of flights")
plt.title("Flights captured per airport")
plt.show()

fig = plt.figure(figsize = (10, 5))

plt.bar(master['aircraft_fam_u'].value_counts()[:10].index, 
        master['aircraft_fam_u'].value_counts()[:10].values)

plt.xticks(rotation = 25)
 
plt.xlabel("Aircraft Families")
plt.ylabel("No. of flights")
plt.title("Flights captured per aircraft families")
plt.show()

# %%

master['aircraft_fam_u'].value_counts()[:10]

# %%

master.head()

# %%

timespan = master.groupby("SAT_DATE")['flight_id'].count()
timespan = timespan.reset_index(drop=False)
timespan.head()

plt.plot(timespan['SAT_DATE'], timespan['flight_id'])
plt.xticks(rotation = 25)

# %%



# %%

#%%

# Outlier Detection

from sklearn.ensemble import IsolationForest

#%%

scoped_df = master.loc[master['airport_arrival'] == "Paris Charles De Gaulle Airport", :]

#%%

x1 = 'flight_name'; x2 ="Airline"; x3 = "segment"; x4="aircraft_fam_u"; x5 = "comments"
X = scoped_df[[x1, x2, x3, x4, x5]]

#%%
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
enc = Pipeline(steps=[
    ("encoder", preprocessing.OrdinalEncoder()),
    ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
])
X = enc.fit_transform(X)


# %%

clf = IsolationForest(max_samples='auto', random_state = 1, contamination= 0.01)
preds = clf.fit_predict(X)
scoped_df['isoletionForest_outliers'] = preds
scoped_df['isoletionForest_outliers'] = scoped_df['isoletionForest_outliers'].astype(str)
scoped_df['isoletionForest_scores'] = clf.decision_function(X)
print(scoped_df['isoletionForest_outliers'].value_counts())

# %%

scoped_df.head()

# %%

scoped_df.loc[scoped_df['isoletionForest_outliers'] == "-1", :].sort_values(by="SAT_DATE", ascending=False)

#%%

scoped_df.loc[scoped_df['isoletionForest_outliers'] == "-1", :]['SAT_DATE'].value_counts()

# %%

len(master.loc[master['flight_id'] == 0, :])

#%%

master = master.loc[master['flight_id'] != 0, :]

# %%

# %%

# %%

for file_location in os.listdir('./data'):
    print(file_location)
    with open(f'./data/{file_location}') as file:
        data = json.load(file)
        DESTINATION_AIRPORT = data['result']['request']['code']
        print(DESTINATION_AIRPORT)

# %%



# %%

# %%

# %%
