# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc #The dash_core_components library includes a component called Graph.
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_table
from collections import OrderedDict
from plotly import tools
from plotly.subplots import make_subplots
from datetime import datetime as dt
from psycopg2 import connect
from sqlalchemy import create_engine
import scipy.io as sio
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
import psycopg2
import math
import os
import sys
import pyproj
from dateutil.relativedelta import relativedelta
import time
import base64
from datetime import date
from dash_extensions import Download
import io
from pathlib import Path
from scipy import stats

# DATA VISUALIZATION WITH RATES INTEGRATED

mapbox_access_token = open("mapbox_token.txt").read()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#Access to database
myFile = open('dbConfig_real.txt')
connStr = myFile.readline()
data_conn = connStr.split(" ",2)
dbname = data_conn[0].split("=",1)[1]
username = data_conn[1].split("=",1)[1]
password = data_conn[2].split("=",1)[1]
conn = connect(connStr)
cur = conn.cursor()
engine = create_engine('postgresql://'+username+':'+password+'@localhost:5432/'+dbname)

# Retrive receivers data
ggm_table = pd.read_sql_query(""" select * from ggm """,engine)
ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
#lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
enu = pyproj.Proj(proj='utm',zone='32N',ellps='WGS84', datum='WGS84')
#lon, lat, alt = pyproj.transform(ecef, lla, ggm_table['a_priori_x'].values, ggm_table['a_priori_y'].values,  ggm_table['a_priori_z'].values, radians=False)
ggm_table['lon'] = ggm_table['a_priori_x']
ggm_table['lat'] = ggm_table['a_priori_y']
#ggm_table['alt'] = alt
receivers=ggm_table['short_name_4ch']

# Tropospheric params
types = ['ZTD','ZWD','PWV','PRESSURE','TEMPERATURE','HUMIDITY']

# Load all the tropo values of the first receiver at 5 minutes from the database and create the first dataframe in the dictionary df_tropo
df_tropo= {"%s"%receivers[0]+"_"+tf+'_'+str(300): pd.read_sql_query(""" select epoch_time, data_val , flag_tropo from gnsstropo where flag_tropo=0 and epoch_time-floor(epoch_time/%s)*%s=12  and id_tropo_table in 
    (select id_tropo_table from troposet 
	  where type= %s 
    and troposet.id_result=(select id_result from ggm where short_name_4ch=%s ) ) order by epoch_time
    """,engine,params=(300,300,tf,receivers[0]))for tf in types }

for key in df_tropo.items():
  df_tropo[key[0]]= df_tropo[key[0]].drop_duplicates(subset=['epoch_time'], keep='last')  
for key in df_tropo.items():
  df_tropo[key[0]]['date']=pd.to_datetime(df_tropo[key[0]]['epoch_time'],unit='s')
  
  
# Function to load from the db the receiver selected from the user based on rate value and add its dataframe the to the dictionary df_tropo
list_key=[]
def get_db (menu_list,rate_val):
  for rec in list(df_tropo.keys()):
    list_key.append( rec.split('_')[0]+'_'+rec.split('_')[2])
    imported_rec=list(set(list_key))
  print(imported_rec)
  for selected_dv in menu_list:
    if selected_dv+'_'+str(rate_val) in imported_rec:
      print('already imported {}'.format(selected_dv+'_'+str(rate_val)))
    else:
      if rate_val !=1:         
        df_update= {"%s"%selected_dv+"_"+tf+'_'+str(rate_val): pd.read_sql_query(""" select epoch_time, data_val ,flag_tropo from gnsstropo where flag_tropo=0   and id_tropo_table in 
          (select id_tropo_table from troposet 
          where type= %s
          and troposet.id_result=(select id_result from ggm where short_name_4ch=%s ) ) order by epoch_time
          """,engine,params=(tf,selected_dv)) for tf in types}
        for key in df_update.items():
          df_update[key[0]]= df_update[key[0]].drop_duplicates(subset=['epoch_time'], keep='last')
        for key in df_update.items():
          df_update[key[0]]['date'] = pd.to_datetime(df_update[key[0]]['epoch_time'], unit='s')
        df_tropo.update(df_update)
      else:
        df_update= {"%s"%selected_dv+"_"+tf+'_'+str(rate_val): pd.read_sql_query(""" select epoch_time, data_val ,flag_tropo from gnsstropo where flag_tropo=0  and epoch_time-floor(epoch_time/%s)*%s=0  and id_tropo_table in 
          (select id_tropo_table from troposet 
          where type= %s
          and troposet.id_result=(select id_result from ggm where short_name_4ch=%s ) ) order by epoch_time
          """,engine,params=(rate_val,rate_val,tf,selected_dv)) for tf in types}
        for key in df_update.items():
          df_update[key[0]]= df_update[key[0]].drop_duplicates(subset=['epoch_time'], keep='last')
        for key in df_update.items():
          df_update[key[0]]['date'] = pd.to_datetime(df_update[key[0]]['epoch_time'], unit='s')
        df_tropo.update(df_update)        
        
  return(df_tropo)

# Same function to load from the db the receiver selected from the user from the points on the map based on rate value
def get_db_points (selectData,rate_val):
  for rec in list(df_tropo.keys()):
    list_key.append( rec.split('_')[0]+'_'+rec.split('_')[2])
    imported_rec=list(set(list_key))
  print(imported_rec)
  for sd in range(int(len(selectData['points']))):
    if selectData['points'][sd]['text']+'_'+str(rate_val) in imported_rec:
      print('already imported')
    else:
      if rate_val !=1:
        df_update= {"%s"%selectData['points'][sd]['text']+"_"+tf+'_'+str(rate_val): pd.read_sql_query(""" select epoch_time, data_val, flag_tropo from gnsstropo where flag_tropo=0  and id_tropo_table in 
          (select id_tropo_table from troposet 
          where type= %s
          and troposet.id_result=(select id_result from ggm where short_name_4ch=%s ) ) order by epoch_time
          """,engine,params=(tf,selectData['points'][sd]['text'])) for tf in types}
        for key in df_update.items():
          df_update[key[0]]= df_update[key[0]].drop_duplicates(subset=['epoch_time'], keep='last')
        for key in df_update.items():
          df_update[key[0]]['date'] = pd.to_datetime(df_update[key[0]]['epoch_time'], unit='s')
        df_tropo.update(df_update)
      else:
        df_update= {"%s"%selectData['points'][sd]['text']+"_"+tf+'_'+str(rate_val): pd.read_sql_query(""" select epoch_time, data_val, flag_tropo from gnsstropo where flag_tropo=0 and epoch_time-floor(epoch_time/%s)*%s=0  and id_tropo_table in 
          (select id_tropo_table from troposet 
          where type= %s
          and troposet.id_result=(select id_result from ggm where short_name_4ch=%s ) ) order by epoch_time
          """,engine,params=(rate_val,rate_val,tf,selectData['points'][sd]['text'])) for tf in types}
        for key in df_update.items():
          df_update[key[0]]= df_update[key[0]].drop_duplicates(subset=['epoch_time'], keep='last')
        for key in df_update.items():
          df_update[key[0]]['date'] = pd.to_datetime(df_update[key[0]]['epoch_time'], unit='s')
        df_tropo.update(df_update)
        
  return(df_tropo)

pos_col=['pos_x', 'pos_y', 'pos_z']

# For each receiver extract position values at 5 minutes
# In this case all the coord values together cause data are few
start=time.time()
df_pos = {"%s"%sn+"_pos_300": pd.read_sql_query(""" select * from gnssposition where flag_pos=0  and id_pos_table in 
  (select id_pos_table from positionset 
  where positionset.id_result=(select id_result from ggm where short_name_4ch=%s ) ) order by pos_time 
  """,engine,params=(sn,)) for sn in ggm_table['short_name_4ch'] }
end=time.time()
print('pos extract  time is {}'.format( end-start))
for key in df_pos.items():
  df_pos[key[0]]= df_pos[key[0]].drop_duplicates(subset=['pos_time'], keep='last')
for key in df_pos.items():
  d=pd.DataFrame()
  d=df_pos[key[0]].copy()
  for i in pos_col:
    z_scores = stats.zscore(d[i])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = abs_z_scores < 3
    d = d[filtered_entries]
  df_pos[key[0]]=d
for key in df_pos.items():
  df_pos[key[0]]['date']=pd.to_datetime(df_pos[key[0]]['pos_time'],unit='s')

  

# Function to load from the db the receiver selected from the user based on rate value and add its dataframe the to the dictionary df_pos
start=time.time()
def get_pos(rate_val):
  df_update_pos = {"%s"%sn+"_pos_"+str(rate_val): pd.read_sql_query(""" select * from gnssposition where flag_pos=0  and id_pos_table in 
    (select id_pos_table from positionset 
    where positionset.id_result=(select id_result from ggm where short_name_4ch=%s ) ) order by pos_time 
    """,engine,params=(sn,)) for sn in ggm_table['short_name_4ch'] }
  for key in df_update_pos.items():
    df_update_pos[key[0]]= df_update_pos[key[0]].drop_duplicates(subset=['pos_time'], keep='last')
    for i in pos_col:
      df_update_pos[key[0]]=df_update_pos[key[0]][np.abs(stats.zscore(df_update_pos[key[0]][i])<2)]
  for key in df_update_pos.items():
    df_update_pos[key[0]]['date']=pd.to_datetime(df_update_pos[key[0]]['pos_time'],unit='s')  
  df_pos.update(df_update_pos)
  return(df_pos)
end=time.time()
print('pos extract  time is {}'.format( end-start))


# Set layout colors
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


# Function that extracts the indeces of the dataframe of interest to be used as condition
#for the extraction of the axis values in the extract values funcion
#parameters are df=name of the dataframe, param=tropo param,rate_vale= sample rate, time_param=time column of the dataframe
#start_date,end_date= boundaries of the rangeslider
start=time.time()
def extract_index(df,name_rec,param,rate_val,time_param,start_date,end_date):

  tmp_time = df["%s"%name_rec+"_"+param+'_'+str(rate_val)][time_param]
  start_index= df["%s"%name_rec+"_"+param+'_'+str(rate_val)].loc[(tmp_time>=start_date),[time_param]].index[0]
  end_index=df["%s"%name_rec+"_"+param+'_'+str(rate_val)].loc[(tmp_time<=end_date),[time_param]].index[-1]

  return start_index,end_index
end=time.time()
print('total extract index time is {}'.format(end-start))

# Function that extracts the axis values, respectively time and value
#parameters are df=name of the dataframe, param=tropo param,rate_vale= sample rate,time_param=time column of the dataframe
#value_param= y axis variable, start_index,end_index= boundaries of the rangeslider
def extract_x_axis_values(df,name_rec,param,rate_val,time_param,start_index,end_index):
    # return epoch time for time axis
    x= df["%s"%name_rec+"_"+param+'_'+str(rate_val)].loc[start_index:end_index][time_param]
    return x
def extract_y_axis_values(df,name_rec,param,rate_val,value_param,start_index,end_index):
    # return values for y axes
    y= df["%s"%name_rec+"_"+param+'_'+str(rate_val)].loc[start_index:end_index][value_param]
    return y
              

logo_gred = './logos/GReD_logo.png' # replace with your own image
encoded_image_gred = base64.b64encode(open(logo_gred, 'rb').read())
logo_polimi = './logos/poli_logo.png' # replace with your own image
encoded_image_polimi = base64.b64encode(open(logo_polimi, 'rb').read())

# Set the layout of the page
app.layout = html.Div([
    html.Div([
      html.Img(src='data:image/png;base64,{}'.format(encoded_image_polimi.decode()), 
      style={'float': 'left','padding':'0 0 0 100px'}),
      html.Img(src='data:image/png;base64,{}'.format(encoded_image_gred.decode()), 
      style={'float': 'right', 'padding':'0 100px 0 0'}),
    #Title
      html.H1( 'LAMPO Water Vapor Monitoring',
      style={'textAlign': 'center','color': '#004876','padding':'20px 0 50px 200px','background-color':'#50D2FF','height':20,'font-style': 'oblique'}
    )
    ]),

    html.Div([
      # START MAP 
      # Map layer with buttons to change style + table
      html.Div([
        html.Details([
          html.Summary('Map'),
          html.Div(['Change style',
            dcc.RadioItems(
              id='style_map',
              options=[{'label': i, 'value': i} for i in ['Satellite', 'OpenStreetMap','Dark']],
              value='Satellite',
              labelStyle={'display': 'inline-block'}
          )],style={'textAlign':'left','padding':'0px 0px 0px 100px'}),
          dcc.Loading(id='loading_map', children=[
          dcc.Graph( 
            id='map'
          )],type="circle")
          ],style={'backgroundColor': colors['background'],'textAlign': 'left','color': colors['text'],'padding':'20px 0 0px 50px'},open=True),
      html.Div([
        html.Details([
          html.Summary('Download'),
            html.Div([
              html.Summary('Select input for the download'),

                dcc.Dropdown(
                  id='down_rec',
                  multi=True,
                  placeholder="Select one or more receivers",
                  options=[{'label': i, 'value': i} for i in receivers],
                  style={'width':'60%'}
                ),
                dcc.Dropdown(
                  id='down_param',
                  multi=True,
                  placeholder="Select one or more parameters",
                  options=[{'label': i, 'value': i} for i in ['ZWD', 'ZTD', 'ZHD', 'pos', 'PWV', 'HUMIDITY','TEMPERATURE','PRESSURE']],
                  style={'width':'60%'}
                ),
                dcc.Dropdown(
                  id='down_timestamp',
                  placeholder="Select one timestamp",
                  options=[{'label': i, 'value': i} for i in ['5min', '1min', '30sec']],
                  style={'width':'60%'}
                ),
                html.Div(id='output-container-param')
          ],),

          html.Div([
            html.Summary('Date'),
              dcc.DatePickerRange(
                id='my-date-picker-range',
                display_format='MMM Do, YY',
                start_date_placeholder_text='MMM Do, YY' 
              ),
            html.Div(id='output-container-date-picker-range')
          ]),
       html.Div([html.Button("Download csv", id="btn", n_clicks=0), html.Div(id='output-container-download')]),
      ],style={'backgroundColor': colors['background'],'textAlign': 'left','color': colors['text'],'padding':'20px 0 0px 50px'}), ]),
      
      ],style={'textAlign': 'center','color': colors['text'],'display':'inline-block','padding':'0px 20px 0 0'}),
      # END MAP

      # START TEMPORAL SERIES
      # Plot wv values with type selector and list receivers
      html.Div([
        html.Details([
          html.Summary('Tropospheric delays temporal series'),
          html.Div([
            dcc.RadioItems(
              id='rate',
              options=[{'label': '5 min', 'value': 300}, {'label': '1 min', 'value': 60},{'label': '30 sec', 'value': 1}],
              value=300,
              labelStyle={'display': 'inline-block'}
          )],style={'textAlign':'left','padding':'0px 0px 0px 100px'}),
          html.Div(['Change parameters',
            dcc.Checklist(
              id='change_type',
              options=[{'label': i, 'value': i} for i in ['ZWD', 'ZTD', 'ZHD']],
              value=['ZWD'],
              labelStyle={'display': 'inline-block'},
              style={'width':'40%',
              'float':'left',
              'padding':'20px 0px 0px 0px'
              }

            ),
            dcc.Dropdown(
              id='select_receiver_menu',
              options=[{'label': i, 'value': i} for i in receivers],
              value=[receivers[0]],
              multi=True,
              placeholder="Select a receiver",
              style={'width':'60%',
              'float':'right'}
            )
          ],style={'textAlign': 'left','color': colors['text'],'padding':'0px 50px 50px 0px'}),

          # Temporal series of tropo values ztd zwd and zhd
          dcc.Loading(id='loading_temp_series', children=[
          dcc.Graph(
            id='temporal_series',
            config={'displaylogo':False,
              'displayModeBar':'hover',
              'modeBarButtonsToRemove': ['hoverCompareCartesian','hoverClosestCartesian','toggleSpikelines']}
          )],type="circle")

          ],style={'backgroundColor': colors['background'],'textAlign': 'left','color': colors['text'],'padding':'0px 40px 0px 50px' } ,
         open= True),

        # Temporal series for position set values
        html.Details([
          html.Summary('Coordinates temporal series'),
          html.Div([
            dcc.Checklist(
              id = 'remove_median',
              options=[{'label': 'Remove the median', 'value': 'True'}],
              value=['True'],
              labelStyle={'display': 'inline-block'},
              style={'width':'40%',
              'float':'left'
              }

            ),
            dcc.Dropdown(
              id = 'select_ref_receiver',
              options=[{'label': g, 'value': g} for g in receivers],
              placeholder="Select a reference receiver",
              style={'width':'60%', 'float':'right'}
          )],style={'backgroundColor': colors['background'],'textAlign': 'left','color': colors['text'],'padding':'0px 60px 50px 0px' }),
          html.Div([
            dcc.Loading(id='loading_coord_series', children=[
            dcc.Graph(
              id='coord_series',
              config={'displaylogo':False,
                'displayModeBar':'hover',
                'modeBarButtonsToRemove': ['hoverCompareCartesian','hoverClosestCartesian','toggleSpikelines']}

          )],type="circle")],style={'backgroundColor': colors['background'],'textAlign': 'left','color': colors['text'] })
          ],style={'backgroundColor': colors['background'],'textAlign': 'left','color': colors['text'],'padding':'0px 40px 0px 50px' }),

        # Range Slider
        html.Div([
          dcc.RangeSlider(
              id = 'range_slider',
              step = 1
        )],style={'width':700,'backgroundColor': colors['background']}),
        # Temporal series pwv
        html.Details([
          html.Summary('PWV temporal series'),
          dcc.Loading(id='loading_pwv_series', children=[
          dcc.Graph(
            id='pwv_temp_series',
            config={'displaylogo':False,
              'displayModeBar':'hover',
              'modeBarButtonsToRemove': ['hoverCompareCartesian','hoverClosestCartesian','toggleSpikelines']}

        )],type="circle")],style={'backgroundColor': colors['background'],'textAlign': 'left','color': colors['text'],'padding':'20px 0 20px 50px' }),

        # Temporal series for others tropo types values
        html.Details([
          html.Summary('Weather stations series PTH'),
          dcc.Loading(id='loading_tropo_series', children=[
          dcc.Graph(
            id='tropo_series',
            config={'displaylogo':False,
              'displayModeBar':'hover',
              'modeBarButtonsToRemove': ['hoverCompareCartesian','hoverClosestCartesian','toggleSpikelines']}

        )],type="circle")],style={'backgroundColor': colors['background'],'textAlign': 'left','color': colors['text'],'padding':'20px 0 20px 50px' })

      ],style={'textAlign': 'center','color': colors['text'],'width': '50%','float': 'right', 'display': 'inline-block','padding':'20px 0 0 0'})
      # END TEMPORAL SERIES
    ])
], style={'backgroundColor': colors['background']})



# CALLSBACK
# Loaders
@app.callback(Output("loading_map", "children"))
@app.callback(Output("loading_temp_series", "children"))
@app.callback(Output("loading_coord_series", "children"))
@app.callback(Output("loading_pwv_series", "children"))
@app.callback(Output("loading_tropo_series", "children"))

# Callback range slider
@app.callback(
  [dash.dependencies.Output('range_slider','min'),
  dash.dependencies.Output('range_slider','max'),
  dash.dependencies.Output('range_slider','value'),
  dash.dependencies.Output('range_slider','marks')],
  [dash.dependencies.Input('select_receiver_menu','value'),
  dash.dependencies.Input('map','selectedData'),
  dash.dependencies.Input('rate','value')])
def update_rangeslider(selected_dropdown_value,selectData,rate_val):
  print('the rate val is:{}'.format(rate_val))
  if selectData != None and selectData['points'] !=[]:
    get_db_points(selectData,rate_val)
    print(selectData)
    print('the points are {}'.format(selectData['points']))
    print(selectData['points'][0]['text'])
    for sd in range(int(len(selectData['points']))):

      min_val=df_tropo["%s"%selectData['points'][sd]['text']+"_ZTD_"+str(rate_val)]['date'].min().month
      max_val=df_tropo["%s"%selectData['points'][sd]['text']+"_ZTD_"+str(rate_val)]['date'].max().month
      marks_val = {date.month:{'label':"%s"%date.strftime("%B")+" "+str(date.year),'style':{'color':'white','font-size':'120%','font-weight': 'bold'}} for date in df_tropo["%s"%selectData['points'][sd]['text']+"_ZTD_"+str(rate_val)]['date'].dt.date.unique()}

      #value = [min_val,max_val]
      marks = marks_val
      value= [max_val -1,max_val]

      return min_val,max_val,value,marks

  # Check if any receiver is selected
  if selected_dropdown_value != []:
    get_db(selected_dropdown_value,rate_val)
    # For each selected receiver
    for rec in selected_dropdown_value:
      min_val=df_tropo["%s"%rec+"_ZTD_"+str(rate_val)]['date'].min().month
      max_val=df_tropo["%s"%rec+"_ZTD_"+str(rate_val)]['date'].max().month
      marks_val = {date.month:{'label':"%s"%date.strftime("%B")+" "+str(date.year),'style':{'color':'white','font-size':'120%','font-weight': 'bold'}} for date in df_tropo["%s"%rec+"_ZTD_"+str(rate_val)]['date'].dt.date.unique()}

    #value = [min_val,max_val]
    marks = marks_val
    value= [max_val -1,max_val]

    return min_val,max_val,value,marks
  # If no receiver has been selected
  else:
    min_val = 0
    max_val = 0
    value = []
    marks = {0:{'label': 'Select a receiver', 'style':{'color':'white','font-size':'120%','font-weight': 'bold'} }}
    return min_val,max_val,value,marks


# Callback of ztd,zwd,pvw temporal values
@app.callback(
    dash.dependencies.Output('temporal_series', 'figure'),
    [dash.dependencies.Input('select_receiver_menu', 'value'),
    dash.dependencies.Input('change_type','value'),
    dash.dependencies.Input('range_slider','value'),
    dash.dependencies.Input('map','selectedData'),
    dash.dependencies.Input('rate','value')])
def update_time_series(selected_dropdown_value,selected_type,selected_range,selectData,rate_val):
  print(selected_range)
  trace=[]
  start_date=""
  end_date=""
  start_index=""
  end_index=""

   # Check if any point is selected
  if selectData != None and selectData['points'] !=[]:
    get_db_points(selectData,rate_val)
    min_y=df_tropo["%s"%selectData['points'][0]['text']+"_ZTD_"+str(rate_val)]['date'].min().year
    min_d=df_tropo["%s"%selectData['points'][0]['text']+"_ZTD_"+str(rate_val)]['date'].min().day
    max_y=df_tropo["%s"%selectData['points'][0]['text']+"_ZTD_"+str(rate_val)]['date'].max().year
    max_d=df_tropo["%s"%selectData['points'][0]['text']+"_ZTD_"+str(rate_val)]['date'].max().day
    if selected_range is not None:
      if selected_range != []:
        start_date=dt(min_y,selected_range[0],min_d)
        end_date=dt(max_y,selected_range[1],max_d)
        print(min_y,min_d,max_y,max_d,start_date,end_date)
        start_index,end_index=extract_index(df_tropo,selectData['points'][0]['text'],'ZTD',rate_val,'date',start_date,end_date)

    # Check if a delay is selected
    if selected_type != []:
      # For each selected receiver
      for sd in range(int(len(selectData['points']))):
        # For each selected delay
        for tropo_type in selected_type:
          if tropo_type == 'ZHD':
            y_ztd = extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],'ZTD',rate_val,'data_val',start_index,end_index)
            y_zwd = extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],'ZWD',rate_val,'data_val',start_index,end_index)
            trace.append(go.Scattergl(
              x = extract_x_axis_values(df_tropo,selectData['points'][sd]['text'],'ZTD',rate_val,'date',start_index,end_index),
              y = (y_ztd.astype(float) - y_zwd.astype(float))*100,
              mode= 'markers',
              name= "%s"%selectData['points'][sd]['text']+"_ZHD"
            ))
          else:

            trace.append(go.Scattergl(
              x = extract_x_axis_values(df_tropo,selectData['points'][sd]['text'],tropo_type,rate_val,'date',start_index,end_index),
              y = (extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],tropo_type,rate_val,'data_val',start_index,end_index).astype(float))*100,
              mode= 'markers',
              name= "%s"%selectData['points'][sd]['text']+"_"+tropo_type
            ))


      figure={
        'data': trace,
        'layout': go.Layout(
              title= {
              'text': 'TROPOSPHERIC DELAYS',
              'xanchor': 'center',
              'yanchor': 'bottom',
              'x': 0.5,
              'y': 1,
              'xref': 'paper',
              'yref': 'paper',
              'font' :{'size':15, 'color' : '#7FDBFF'}
              },
              xaxis={
                'title': 'Date', 
                'automargin':True,
                'rangeselector': {'buttons': list([
                  {'count': 1, 'label': '1M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 3, 'label': '3M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 6, 'label': '6M', 
                   'step': 'month',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': '1Y', 
                   'step': 'year',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': 'YTD', 
                   'step': 'year', 
                   'stepmode': 'todate'},
                  ])},
                  'type': 'date'},
              yaxis={'title': '[cm]', 'rangemode':'nonnegative'},
              hovermode='closest',
              plot_bgcolor= colors['background'],
              paper_bgcolor= colors['background'],
              font= {'color': colors['text']},
              showlegend = True,
              height=250,
              margin= {
                'l': 40,
                'r': 10,
                'b': 0,
                't': 15
              },
              width=750

        )
      }

      return  figure


    # If no delay has been selected
    else:
      figure={
      'layout': go.Layout(
            title= {
              'text': 'Please select a delay',
              'xanchor': 'center',
              'yanchor': 'bottom',
              'x': 0.5,
              'y': 1,
              'xref': 'paper',
              'yref': 'paper',
              'font' :{'size':15, 'color' : '#7FDBFF'}
              },
            xaxis={
                'title': 'Date', 
                'automargin':True,
                'rangeselector': {'buttons': list([
                  {'count': 1, 'label': '1M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 3, 'label': '3M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 6, 'label': '6M', 
                   'step': 'month',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': '1Y', 
                   'step': 'year',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': 'YTD', 
                   'step': 'year', 
                   'stepmode': 'todate'},
                  ])},
                  'type': 'date'},

            yaxis={'title': '[cm]', 'rangemode':'nonnegative'},
            hovermode='closest',
            plot_bgcolor= colors['background'],
            paper_bgcolor= colors['background'],
            height=250,
              margin= {
                'l': 40,
                'r': 10,
                'b': 0,
                't': 15
              },
              width=750

      )
    }
    return figure


  # Check if any receiver is selected
  if selected_dropdown_value != []:
    get_db(selected_dropdown_value,rate_val)
    start_plot_fig=time.time()
    min_y=df_tropo["%s"%selected_dropdown_value[0]+"_ZTD_"+str(rate_val)]['date'].min().year
    min_d=df_tropo["%s"%selected_dropdown_value[0]+"_ZTD_"+str(rate_val)]['date'].min().day
    max_y=df_tropo["%s"%selected_dropdown_value[0]+"_ZTD_"+str(rate_val)]['date'].max().year
    max_d=df_tropo["%s"%selected_dropdown_value[0]+"_ZTD_"+str(rate_val)]['date'].max().day
    if selected_range is not None:
      if selected_range != []:
        start_date=dt(min_y,selected_range[0],min_d)
        end_date=dt(max_y,selected_range[1],max_d)
        print(min_y,min_d,max_y,max_d,start_date,end_date)
        start_time=time.time()
        start_index,end_index=extract_index(df_tropo,selected_dropdown_value[0],'ZTD',rate_val,'date',start_date,end_date)
        end_time=time.time()
        print('the func time is {}'.format(end_time-start_time))
        print(start_index,end_index)
        print('time until here {}'.format(end_time-start_plot_fig))
    # Check if a delay is selected
    if selected_type != []:
      # For each selected receiver
      for rec in selected_dropdown_value:
        # For each selected delay
        for tropo_type in selected_type:
          if tropo_type == 'ZHD':
            start_zhd=time.time()
            y_ztd = extract_y_axis_values(df_tropo,rec,'ZTD',rate_val,'data_val',start_index,end_index)
            end_zhd=time.time()
            print('the extract y values is {}'.format(end_zhd-start_zhd))
            y_zwd = extract_y_axis_values(df_tropo,rec,'ZWD',rate_val,'data_val',start_index,end_index)
            trace.append(go.Scattergl(
              x = extract_x_axis_values(df_tropo,rec,'ZTD',rate_val,'date',start_index,end_index),
              y = (y_ztd.astype(float) - y_zwd.astype(float))*100,
              mode= 'markers',
              name= "%s"%rec+"_ZHD"
            ))
          else:
            start_append=time.time()
            trace.append(go.Scattergl(
              x = extract_x_axis_values(df_tropo,rec,tropo_type,rate_val,'date',start_index,end_index),
              y = (extract_y_axis_values(df_tropo,rec,tropo_type,rate_val,'data_val',start_index,end_index).astype(float))*100,
              mode= 'markers',
              name= "%s"%rec+"_"+tropo_type
            ))
            end_append=time.time()
            print('THE APPEND TIME IS {}'.format(end_append-start_append))

      figure={
        'data': trace,
        'layout': go.Layout(
            title= {
              'text': 'TROPOSPHERIC DELAYS',
              'xanchor': 'center',
              'yanchor': 'bottom',
              'x': 0.5,
              'y': 1,
              'xref': 'paper',
              'yref': 'paper',
              'font' :{'size':15, 'color' : '#7FDBFF'}
              },
              xaxis={
                'title': 'Date', 
                'automargin':True,
                'rangeselector': {'buttons': list([
                  {'count': 1, 'label': '1M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 3, 'label': '3M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 6, 'label': '6M', 
                   'step': 'month',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': '1Y', 
                   'step': 'year',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': 'YTD', 
                   'step': 'year', 
                   'stepmode': 'todate'},
                  ])},
                  'type': 'date'},


              yaxis={'title': '[cm]', 'rangemode':'nonnegative'},
              hovermode='closest',
              plot_bgcolor= colors['background'],
              paper_bgcolor= colors['background'],
              font= {'color': colors['text']},
              showlegend = True,
              height=250,
              margin= {
                'l': 40,
                'r': 10,
                'b': 0,
                't': 15
              },
              width=750
        )
      }
      end_plot_fig=time.time()
      print('THE PLOT TIME IS {}'.format(end_plot_fig-start_plot_fig))

      return figure

    # If no delay has been selected
    else:
      figure={
      'layout': go.Layout(
            title= {
              'text': 'Please select a delay',
              'xanchor': 'center',
              'yanchor': 'bottom',
              'x': 0.5,
              'y': 1,
              'xref': 'paper',
              'yref': 'paper',
              'font' :{'size':15, 'color' : '#7FDBFF'}
              },
            xaxis={
                'title': 'Date', 
                'automargin':True,
                'rangeselector': {'buttons': list([
                  {'count': 1, 'label': '1M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 3, 'label': '3M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 6, 'label': '6M', 
                   'step': 'month',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': '1Y', 
                   'step': 'year',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': 'YTD', 
                   'step': 'year', 
                   'stepmode': 'todate'},
                  ])},
                  'type': 'date'},

            yaxis={'title': '[cm]', 'rangemode':'nonnegative'},
            hovermode='closest',
            plot_bgcolor= colors['background'],
            paper_bgcolor= colors['background'],
            font= {'color': colors['text']},
            height=250,
              margin= {
                'l': 40,
                'r': 10,
                'b': 0,
                't': 15
              },
              width=750


      )
    }
    return figure
  # If no receiver has been selected
  else:
    figure={
      'layout': go.Layout(
            title= {
              'text': 'Please select a receiver',
              'xanchor': 'center',
              'yanchor': 'bottom',
              'x': 0.5,
              'y': 1,
              'xref': 'paper',
              'yref': 'paper',
              'font' :{'size':15, 'color' : '#7FDBFF'}
              },
            xaxis={
                'title': 'Date', 
                'automargin':True,
                'rangeselector': {'buttons': list([
                  {'count': 1, 'label': '1M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 3, 'label': '3M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 6, 'label': '6M', 
                   'step': 'month',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': '1Y', 
                   'step': 'year',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': 'YTD', 
                   'step': 'year', 
                   'stepmode': 'todate'},
                  ])},
                  'type': 'date'},

            yaxis={'title': '[cm]', 'rangemode':'nonnegative'},
            hovermode='closest',
            plot_bgcolor= colors['background'],
            paper_bgcolor= colors['background'],
            font= {'color': colors['text']},
            height=250,
              margin= {
                'l': 40,
                'r': 10,
                'b': 0,
                't': 15
              },
              width=750

      )
    }

  return figure




# Callback pwv
@app.callback(
    dash.dependencies.Output('pwv_temp_series', 'figure'),
    [dash.dependencies.Input('select_receiver_menu', 'value'),
    dash.dependencies.Input('range_slider','value'),
    dash.dependencies.Input('map','selectedData'),
    dash.dependencies.Input('rate','value')])
def update_pwv_series(selected_dropdown_value,selected_range,selectData,rate_val):
  trace_pwv=[]
  start_date=""
  end_date=""
  start_index=""
  end_index=""
  # Check if any point is selected
  if selectData != None and selectData['points'] !=[]:
    get_db_points(selectData,rate_val)
    if df_tropo["%s"%selectData['points'][0]['text']+"_PWV_"+str(rate_val)].empty != True:
      min_y=df_tropo["%s"%selectData['points'][0]['text']+"_PWV_"+str(rate_val)]['date'].min().year
      min_d=df_tropo["%s"%selectData['points'][0]['text']+"_PWV_"+str(rate_val)]['date'].min().day
      max_y=df_tropo["%s"%selectData['points'][0]['text']+"_PWV_"+str(rate_val)]['date'].max().year
      max_d=df_tropo["%s"%selectData['points'][0]['text']+"_PWV_"+str(rate_val)]['date'].max().day
      if selected_range is not None:
        if selected_range != []:
          start_date=dt(min_y,selected_range[0],min_d)
          end_date=dt(max_y,selected_range[1],max_d)
          print(min_y,min_d,max_y,max_d,start_date,end_date)
          start_index,end_index=extract_index(df_tropo,selectData['points'][0]['text'],'PWV',rate_val,'date',start_date,end_date)
      # For each selected receiver
    for sd in range(int(len(selectData['points']))):
      if df_tropo["%s"%selectData['points'][sd]['text']+"_PWV_"+str(rate_val)].empty == True:
        trace_pwv.append(go.Scattergl(
          x = [2],
          y = [2],
          mode = 'markers+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name= "%s"%selectData['points'][sd]['text']+"_PWV"
        ))
      else:
        trace_pwv.append(go.Scattergl(
          x = extract_x_axis_values(df_tropo,selectData['points'][sd]['text'],'PWV',rate_val,'date',start_index,end_index),
          y = (extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],'PWV',rate_val,'data_val',start_index,end_index).astype(float))*1000,
          mode= 'markers',
          name= "%s"%selectData['points'][sd]['text']+"_PWV"
        ))
    figure_pwv={
      'data': trace_pwv,
      'layout': go.Layout(
            title= {
              'text':  'PWV',
              'xanchor': 'center',
              'yanchor': 'bottom',
              'x': 0.5,
              'y': 1,
              'xref': 'paper',
              'yref': 'paper',
              'font' :{'size':15, 'color' : '#7FDBFF'}
              },


            xaxis={
                'title': 'Date', 
                'automargin':True,
                'rangeselector': {'buttons': list([
                  {'count': 1, 'label': '1M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 3, 'label': '3M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 6, 'label': '6M', 
                   'step': 'month',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': '1Y', 
                   'step': 'year',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': 'YTD', 
                   'step': 'year', 
                   'stepmode': 'todate'},
                  ])},
                  'type': 'date'},

            yaxis={'title': '[mm]', 'rangemode':'nonnegative'},
            hovermode='closest',
            plot_bgcolor= colors['background'],
            paper_bgcolor= colors['background'],
            font= {'color': colors['text']},
            showlegend = True,
              height=250,
              margin= {
                'l': 40,
                'r': 10,
                'b': 0,
                't': 15
              }

      )
    }
    return figure_pwv

   # Check if any receiver is selected
  if selected_dropdown_value != []:
    get_db(selected_dropdown_value,rate_val)
    if df_tropo["%s"%selected_dropdown_value[0]+"_PWV_"+str(rate_val)].empty != True:
      min_y=df_tropo["%s"%selected_dropdown_value[0]+"_PWV_"+str(rate_val)]['date'].min().year
      min_d=df_tropo["%s"%selected_dropdown_value[0]+"_PWV_"+str(rate_val)]['date'].min().day
      max_y=df_tropo["%s"%selected_dropdown_value[0]+"_PWV_"+str(rate_val)]['date'].max().year
      max_d=df_tropo["%s"%selected_dropdown_value[0]+"_PWV_"+str(rate_val)]['date'].max().day
      if selected_range is not None:
        if selected_range != []:
          start_date=dt(min_y,selected_range[0],min_d)
          end_date=dt(max_y,selected_range[1],max_d)
          print(min_y,min_d,max_y,max_d,start_date,end_date)
          start_index,end_index=extract_index(df_tropo,selected_dropdown_value[0],'ZTD',rate_val,'date',start_date,end_date)
    # For each selected receiver
    for rec in selected_dropdown_value:
      if df_tropo["%s"%rec+"_PWV_"+str(rate_val)].empty == True:
        trace_pwv.append(go.Scattergl(
          x = [2],
          y = [2],
          mode = 'markers+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35}
        ))
      else:
        trace_pwv.append(go.Scattergl(
          x = extract_x_axis_values(df_tropo,rec,'PWV',rate_val,'date',start_index,end_index),
          y = (extract_y_axis_values(df_tropo,rec,'PWV',rate_val,'data_val',start_index,end_index).astype(float))*1000,
          mode= 'markers',
          name= "%s"%rec+"_PWV"
        ))
    figure_pwv={
      'data': trace_pwv,
      'layout': go.Layout(
            title= {
              'text': 'PWV',
              'xanchor': 'center',
              'yanchor': 'bottom',
              'x': 0.5,
              'y': 1,
              'xref': 'paper',
              'yref': 'paper',
              'font' :{'size':15, 'color' : '#7FDBFF'}
              },

            xaxis={
                'title': 'Date', 
                'automargin':True,
                'rangeselector': {'buttons': list([
                  {'count': 1, 'label': '1M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 3, 'label': '3M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 6, 'label': '6M', 
                   'step': 'month',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': '1Y', 
                   'step': 'year',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': 'YTD', 
                   'step': 'year', 
                   'stepmode': 'todate'},
                  ])},
                  'type': 'date'},

            yaxis={'title': '[mm]', 'rangemode':'nonnegative'},
            hovermode='closest',
            plot_bgcolor= colors['background'],
            paper_bgcolor= colors['background'],
            font= {'color': colors['text']},
            showlegend = True,
              height=250,
              margin= {
                'l': 40,
                'r': 10,
                'b': 0,
                't': 15
              }

      )
    }
    return figure_pwv
  # If no receiver has been selected
  else:
    figure_pwv={
      'layout': go.Layout(

            title= {
              'text': 'Please select a receiver',
              'xanchor': 'center',
              'yanchor': 'bottom',
              'x': 0.5,
              'y': 1,
              'xref': 'paper',
              'yref': 'paper',
              'font' :{'size':15, 'color' : '#7FDBFF'}
              },
            xaxis={
                'title': 'Date', 
                'automargin':True,
                'rangeselector': {'buttons': list([
                  {'count': 1, 'label': '1M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 3, 'label': '3M', 
                   'step': 'month', 
                   'stepmode': 'backward'},
                  {'count': 6, 'label': '6M', 
                   'step': 'month',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': '1Y', 
                   'step': 'year',
                   'stepmode': 'backward'},
                  {'count': 1, 'label': 'YTD', 
                   'step': 'year', 
                   'stepmode': 'todate'},
                  ])},
                  'type': 'date'},

            yaxis={'title': '[mm]', 'rangemode':'nonnegative'},
            hovermode='closest',
            plot_bgcolor= colors['background'],
            paper_bgcolor= colors['background'],
            font= {'color': colors['text']},
      )
    }

  return figure_pwv

# Callback map
@app.callback(
  dash.dependencies.Output('map','figure'),
  [dash.dependencies.Input('style_map','value')])
def update_map(style_layer):
  trace_1=go.Scattermapbox(
    name = 'Receiver',
    lat = ggm_table['lat'],
    lon = ggm_table['lon'],
    text = ggm_table['short_name_4ch'],
    hoverlabel = {'namelength':0,
      'bgcolor':'white',
      'font':{'color':'black'}},
    hovertemplate =
            "<b>Lon: %{lon}<br>" +
            "<b>Lat: %{lat}<br>" +
            "<b>Receiver: %{text}<br>"
            ,
    marker = {
      'color' : '#7FDBFF',
      'size': 20
    },
    selected = {
    'marker': {
    'color' : '#FF0EAF'
    }
    }
    )

  if style_layer=='Satellite':
    layout=go.Layout(
      height=500,
      width=700,
      mapbox= {
        'zoom': 6.7,
        'center':{'lon':9.78,'lat':45.8082}, 
        'accesstoken': 'pk.eyJ1Ijoic2FyYW1hZmZpb2xpOTUiLCJhIjoiY2sxZzZkMmFlMDhtMDNocXB1aTgzam04NiJ9._N6Pg4KwL7jRjTCNm6BjOA',
        'style': 'mapbox://styles/mapbox/satellite-v9'
        },
      showlegend=True,
      plot_bgcolor= colors['background'],
      paper_bgcolor= colors['background'],
      font= {'color': colors['text']},
      margin= {
        'l': 20,
        'r': 20,
        'b': 0,
        't': 15
      })
  elif style_layer=='OpenStreetMap':
    layout=go.Layout(
      #title='MAP',
      height=500,
      width=700,
      mapbox= {
        'zoom': 6.7,
        'center':{'lon':9.78,'lat':45.8082},
        'style':'open-street-map'
        },
      showlegend=True,
      plot_bgcolor= colors['background'],
      paper_bgcolor= colors['background'],
      font= {'color': colors['text']},
      margin= {
        'l': 20,
        'r': 20,
        'b': 0,
        't': 15
      })
  elif style_layer=='Dark':
    layout=go.Layout(
      height=500,
      width=700,
      mapbox= {
        'zoom': 6.7,
        'center':{'lon':9.78,'lat':45.8082},
        'accesstoken': 'pk.eyJ1Ijoic2FyYW1hZmZpb2xpOTUiLCJhIjoiY2sxZzZkMmFlMDhtMDNocXB1aTgzam04NiJ9._N6Pg4KwL7jRjTCNm6BjOA',
        'style': 'mapbox://styles/saramaffioli95/ck2m0kuo300cc1ctfnpqg6dum'
        },
      showlegend=True,
      plot_bgcolor= colors['background'],
      paper_bgcolor= colors['background'],
      font= {'color': colors['text']},
      margin= {
        'l': 20,
        'r': 20,
        'b': 0,
        't': 15
      })

  fig = go.Figure(data = trace_1, layout = layout)
  return fig

# Callback tropo series
@app.callback(
  dash.dependencies.Output('tropo_series','figure'),
  [dash.dependencies.Input('select_receiver_menu','value'),
  dash.dependencies.Input('range_slider','value'),
  dash.dependencies.Input('map','selectedData'),
  dash.dependencies.Input('rate','value')])
def update_tropo_series(selected_dropdown_value,selected_range,selectData,rate_val):
  fig_tropo = make_subplots(rows=3, cols=1, specs=[[{}], [{}],[{}]],shared_xaxes=True,vertical_spacing=0.08,row_width=[15, 15, 15],
    subplot_titles=("PRESSURE","TEMPERATURE","HUMIDITY"))
  start_date=""
  end_date=""
  start_index=""
  end_index=""
  # Check if any point is selected
  if selectData != None and selectData['points'] !=[]:
    get_db_points(selectData,rate_val)
    if df_tropo["%s"%selectData['points'][0]['text']+"_PRESSURE_"+str(rate_val)].empty != True:
      min_y=df_tropo["%s"%selectData['points'][0]['text']+"_PRESSURE_"+str(rate_val)]['date'].min().year
      min_d=df_tropo["%s"%selectData['points'][0]['text']+"_PRESSURE_"+str(rate_val)]['date'].min().day
      max_y=df_tropo["%s"%selectData['points'][0]['text']+"_PRESSURE_"+str(rate_val)]['date'].max().year
      max_d=df_tropo["%s"%selectData['points'][0]['text']+"_PRESSURE_"+str(rate_val)]['date'].max().day
      print('TROPO DATE{}'.format(min_y,min_d,max_y,max_d))
      if selected_range is not None:
        if selected_range != []:
          start_date=dt(min_y,selected_range[0],min_d)
          end_date=dt(max_y,selected_range[1],max_d)
          print(min_y,min_d,max_y,max_d,start_date,end_date)
          start_index,end_index=extract_index(df_tropo,selectData['points'][0]['text'],'PRESSURE',rate_val,'date',start_date,end_date)

    for sd in range(int(len(selectData['points']))):
            # Pressure
      if df_tropo["%s"%selectData['points'][sd]['text']+"_PRESSURE_"+str(rate_val)].empty == True:
        fig_tropo.append_trace(go.Scattergl(
          x = [2],
          y = [2],
          mode = 'markers+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name='Pressure {}'.format(selectData['points'][sd]['text'])
        ), 1, 1)

      else:
        fig_tropo.append_trace(go.Scattergl(
          x = extract_x_axis_values(df_tropo,selectData['points'][sd]['text'],'PRESSURE',rate_val,'date',start_index,end_index),
          y = extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],'PRESSURE',rate_val,'data_val',start_index,end_index),
          mode='markers',
          name='Pressure {}'.format(selectData['points'][sd]['text'])
        ), 1, 1)

      # Temperature
      if df_tropo["%s"%selectData['points'][sd]['text']+"_TEMPERATURE_"+str(rate_val)].empty == True:
        fig_tropo.append_trace(go.Scattergl(
          x = [2],
          y = [2],
          mode = 'markers+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name='Temperature {}'.format(selectData['points'][sd]['text'])
        ), 2, 1)

      else:
        fig_tropo.append_trace(go.Scattergl(
          x = extract_x_axis_values(df_tropo,selectData['points'][sd]['text'],'TEMPERATURE',rate_val,'date',start_index,end_index),
          y = extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],'TEMPERATURE',rate_val,'data_val',start_index,end_index),
          mode='markers',
          name='Temperature {}'.format(selectData['points'][sd]['text'])
        ), 2, 1)

      # Humidity
      if df_tropo["%s"%selectData['points'][sd]['text']+"_HUMIDITY_"+str(rate_val)].empty == True:
        fig_tropo.append_trace(go.Scattergl(
          x = [2],
          y = [2],
          mode = 'markers+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name='Humidity {}'.format(selectData['points'][sd]['text'])
        ), 3, 1)

      else:
        fig_tropo.append_trace(go.Scattergl(
          x = extract_x_axis_values(df_tropo,selectData['points'][sd]['text'],'HUMIDITY',rate_val,'date',start_index,end_index),
          y = extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],'HUMIDITY',rate_val,'data_val',start_index,end_index),
          mode='markers',
          name='Humidity {}'.format(selectData['points'][sd]['text'])
        ), 3, 1)

      # Update axis properties and layout
    fig_tropo.update_xaxes(title_text="Date",gridcolor='#2D2D2D', gridwidth=0.2,row=3, col=1)
    fig_tropo.update_xaxes(gridcolor='#2D2D2D', gridwidth=0.2,row=1, col=1)
    fig_tropo.update_xaxes(gridcolor='#2D2D2D', gridwidth=0.2,row=2, col=1)
    fig_tropo.update_yaxes(title_text="[mbar]",gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False, row=1, col=1)
    fig_tropo.update_yaxes(title_text="[°C]",gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False, row=2, col=1)
    fig_tropo.update_yaxes(title_text="[%]", gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False,row=3, col=1)
    fig_tropo['layout'].update(height=600, paper_bgcolor  ='#111111',plot_bgcolor='#111111', font={'color':'#7FDBFF'},
                margin= {
                'l': 30,
                'r': 10,
                'b': 0,
                't': 15
              })
    fig_tropo.update_xaxes( rangeselector=dict(
      buttons=list([
          dict(count=1,
               label="1m",
               step="month",
               stepmode="backward"),
          dict(count=3,
               label="3m",
               step="month",
               stepmode="backward"),
          dict(count=6,
               label="6m",
               step="month",
               stepmode="backward"),
          dict(count=1,
               label="1y",
               step="year",
               stepmode="backward"),          
          dict(count=1,
               label="YTD",
               step="year",
               stepmode="todate")
      ])),
      type="date",
      row=1,
      col=1
    )

    return fig_tropo

  # Check if any receiver is selected
  if selected_dropdown_value != []:
    get_db(selected_dropdown_value,rate_val)
    if df_tropo["%s"%selected_dropdown_value[0]+"_PRESSURE_"+str(rate_val)].empty != True:
      min_y=df_tropo["%s"%selected_dropdown_value[0]+"_PRESSURE_"+str(rate_val)]['date'].min().year
      min_d=df_tropo["%s"%selected_dropdown_value[0]+"_PRESSURE_"+str(rate_val)]['date'].min().day
      max_y=df_tropo["%s"%selected_dropdown_value[0]+"_PRESSURE_"+str(rate_val)]['date'].max().year
      max_d=df_tropo["%s"%selected_dropdown_value[0]+"_PRESSURE_"+str(rate_val)]['date'].max().day
      if selected_range is not None:
        if selected_range != []:
          start_date=dt(min_y,selected_range[0],min_d)
          end_date=dt(max_y,selected_range[1],max_d)
          print(min_y,min_d,max_y,max_d,start_date,end_date)
          start_index,end_index=extract_index(df_tropo,selected_dropdown_value[0],'PRESSURE',rate_val,'date',start_date,end_date)
    # For each selected receiver
    for rec in selected_dropdown_value:

      # Pressure
      if df_tropo["%s"%rec+"_PRESSURE_"+str(rate_val)].empty == True:
        fig_tropo.append_trace(go.Scattergl(
          x = [2],
          y = [2],
          mode = 'markers+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name='Pressure {}'.format(rec)
        ), 1, 1)

      else:
        fig_tropo.append_trace(go.Scattergl(
          x = extract_x_axis_values(df_tropo,rec,'PRESSURE',rate_val,'date',start_index,end_index),
          y = extract_y_axis_values(df_tropo,rec,'PRESSURE',rate_val,'data_val',start_index,end_index),
          mode='markers',
          name='Pressure {}'.format(rec)
        ), 1, 1)

      # Temperature
      if df_tropo["%s"%rec+"_TEMPERATURE_"+str(rate_val)].empty == True:
        fig_tropo.append_trace(go.Scattergl(
          x = [2],
          y = [2],
          mode = 'markers+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name='Temperature {}'.format(rec)
        ), 2, 1)

      else:
        fig_tropo.append_trace(go.Scattergl(
          x = extract_x_axis_values(df_tropo,rec,'TEMPERATURE',rate_val,'date',start_index,end_index),
          y = extract_y_axis_values(df_tropo,rec,'TEMPERATURE',rate_val,'data_val',start_index,end_index),
          mode='markers',
          name='Temperature {}'.format(rec)
        ), 2, 1)

      # Humidity
      if df_tropo["%s"%rec+"_HUMIDITY_"+str(rate_val)].empty == True:
        fig_tropo.append_trace(go.Scattergl(
          x = [2],
          y = [2],
          mode = 'markers+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name='Humidity {}'.format(rec)
        ), 3, 1)

      else:
        fig_tropo.append_trace(go.Scattergl(
          x = extract_x_axis_values(df_tropo,rec,'HUMIDITY',rate_val,'date',start_index,end_index),
          y = extract_y_axis_values(df_tropo,rec,'HUMIDITY',rate_val,'data_val',start_index,end_index),
          mode='markers',
          name='Humidity {}'.format(rec)
        ), 3, 1)

      # Update axis properties and layout
    fig_tropo.update_xaxes(title_text="Date",gridcolor='#2D2D2D', gridwidth=0.2,row=3, col=1)
    fig_tropo.update_xaxes(gridcolor='#2D2D2D', gridwidth=0.2,row=1, col=1)
    fig_tropo.update_xaxes(gridcolor='#2D2D2D', gridwidth=0.2,row=2, col=1)
    fig_tropo.update_yaxes(title_text="[mbar]",gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False, row=1, col=1)
    fig_tropo.update_yaxes(title_text="[°C]",gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False, row=2, col=1)
    fig_tropo.update_yaxes(title_text="[%]", gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False,row=3, col=1)
    fig_tropo['layout'].update(height=600,paper_bgcolor  ='#111111',plot_bgcolor='#111111', font={'color':'#7FDBFF'},
               margin= {
                'l': 30,
                'r': 10,
                'b': 0,
                't': 15
              })
    fig_tropo.update_xaxes( rangeselector=dict(
      buttons=list([
          dict(count=1,
               label="1m",
               step="month",
               stepmode="backward"),
          dict(count=3,
               label="3m",
               step="month",
               stepmode="backward"),
          dict(count=6,
               label="6m",
               step="month",
               stepmode="backward"),
          dict(count=1,
               label="1y",
               step="year",
               stepmode="backward"),          
          dict(count=1,
               label="YTD",
               step="year",
               stepmode="todate")
      ])),
      type="date",
      row=1,
      col=1
    )

    return fig_tropo
  # If no receiver has been selected
  else:
    fig_tropo ={
      'layout': go.Layout(
            title= {
              'text': 'Please select a receiver',
              'xanchor': 'center',
              'x': 0.5,
              'xref': 'paper',
              'font' :{'size':15, 'color' : '#7FDBFF'}
              },
            xaxis={'title': 'Date', 'automargin':True},
            hovermode='closest',
            plot_bgcolor= colors['background'],
            paper_bgcolor= colors['background'],
            font= {'color': colors['text']},
            height= 250
      )
    }

  return fig_tropo

# Callback coord series
@app.callback(
  dash.dependencies.Output('coord_series','figure'),
  [dash.dependencies.Input('select_receiver_menu','value'),
  dash.dependencies.Input('remove_median','value'),
  dash.dependencies.Input('select_ref_receiver','value'),
  dash.dependencies.Input('range_slider','value'),
  dash.dependencies.Input('map','selectedData'),
  dash.dependencies.Input('rate','value')])
def update_coord_series(selected_dropdown_value,values,ref,selected_range,selectData,rate_val):

  fig_coord = make_subplots(rows=3, cols=1, specs=[[{}],[{}],[{}]],shared_xaxes=True,vertical_spacing=0.08,subplot_titles=('EAST', 'NORTH', 'UP'),
                          row_width=[15, 15, 15])
  start_date=""
  end_date=""
  start_index=""
  end_index=""
  # Check if any point is selected
  if selectData != None and selectData['points'] !=[]:
    get_pos(rate_val)  
    min_y=df_pos["%s"%selectData['points'][0]['text']+"_pos_"+str(rate_val)]['date'].min().year
    min_d=df_pos["%s"%selectData['points'][0]['text']+"_pos_"+str(rate_val)]['date'].min().day
    max_y=df_pos["%s"%selectData['points'][0]['text']+"_pos_"+str(rate_val)]['date'].max().year
    max_d=df_pos["%s"%selectData['points'][0]['text']+"_pos_"+str(rate_val)]['date'].max().day
    if selected_range is not None:
      if selected_range != []:
        start_date=dt(min_y,selected_range[0],min_d)
        end_date=dt(max_y,selected_range[1],max_d)
        print(min_y,min_d,max_y,max_d,start_date,end_date)
        start_index,end_index=extract_index(df_pos,selectData['points'][0]['text'],'pos',rate_val,'date',start_date,end_date)


    # For each selected receiver
    for sd in range(int(len(selectData['points']))):
      east, north, up = pyproj.transform(ecef, enu, df_pos["%s"%selectData['points'][sd]['text']+"_pos_"+str(rate_val)]['pos_x'].values, df_pos["%s"%selectData['points'][sd]['text']+"_pos_"+str(rate_val)]['pos_y'].values, \
      df_pos["%s"%selectData['points'][sd]['text']+"_pos_"+str(rate_val)]['pos_z'].values, radians=False)
      df_pos["%s"%selectData['points'][sd]['text']+"_pos_"+str(rate_val)]['east'] = east
      df_pos["%s"%selectData['points'][sd]['text']+"_pos_"+str(rate_val)]['north'] = north
      df_pos["%s"%selectData['points'][sd]['text']+"_pos_"+str(rate_val)]['up'] = up

      # Check reference receiver to be substracted
      if ref is not None:
        east, north, up = pyproj.transform(ecef, enu, df_pos["%s"%ref+"_pos_"+str(rate_val)]['pos_x'].values, df_pos["%s"%ref+"_pos_"+str(rate_val)]['pos_y'].values, df_pos["%s"%ref+"_pos_"+str(rate_val)]['pos_z'].values, radians=False)
        df_pos["%s"%ref+"_pos_"+str(rate_val)]['east'] = east
        df_pos["%s"%ref+"_pos_"+str(rate_val)]['north'] = north
        df_pos["%s"%ref+"_pos_"+str(rate_val)]['up'] = up
        # If remove median equal to True
        if 'True' in values:
        # EAST
          y_east_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'east',start_index,end_index).astype(float)
          y_east_ref = extract_y_axis_values(df_pos,ref,'pos',rate_val,'east',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'date',start_index,end_index),
              y = y_east_sd - y_east_sd.median() - y_east_ref + y_east_ref.median(),
              mode = 'markers',
              name = "%s"%selectData['points'][sd]['text']+" "+"EAST"
            ), 1, 1)
          # NORTH
          y_north_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'north',start_index,end_index).astype(float)
          y_north_ref = extract_y_axis_values(df_pos,ref,'pos',rate_val,'north',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'date',start_index,end_index),
              y = y_north_sd - y_north_sd.median() - y_north_ref + y_north_ref.median(),
              mode = 'markers',
              name = "%s"%selectData['points'][sd]['text']+" "+"NORTH"
            ), 2, 1)
          # UP
          y_up_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'up',start_index,end_index).astype(float)
          y_up_ref = extract_y_axis_values(df_pos,ref,'pos',rate_val,'up',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'date',start_index,end_index),
              y = y_up_sd - y_up_sd.median() - y_up_ref + y_up_ref.median(),
              mode = 'markers',
              name = "%s"%selectData['points'][sd]['text']+" "+"UP"
            ), 3, 1)

        # If remove median is False
        else:
          # EAST
          y_east_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'east',start_index,end_index).astype(float)
          y_east_ref = extract_y_axis_values(df_pos,ref,'pos',rate_val,'east',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'date',start_index,end_index),
              y = y_east_sd - y_east_ref ,
              mode = 'markers',
              name = "%s"%selectData['points'][sd]['text']+" "+"EAST"
            ), 1, 1)
          # NORTH
          y_north_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'north',start_index,end_index).astype(float)
          y_north_ref = extract_y_axis_values(df_pos,ref,'pos',rate_val,'north',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'date',start_index,end_index),
              y = y_north_sd - y_north_ref,
              mode = 'markers',
              name = "%s"%selectData['points'][sd]['text']+" "+"NORTH"
            ), 2, 1)
          # UP
          y_up_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'up',start_index,end_index).astype(float)
          y_up_ref = extract_y_axis_values(df_pos,ref,'pos',rate_val,'up',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'date',start_index,end_index),
              y = y_up_sd - y_up_ref ,
              mode = 'markers',
              name = "%s"%selectData['points'][sd]['text']+" "+"UP"
            ), 3, 1)

      # If no reference receiver is selected
      else:
        # Check value remove median box
        if 'True' in values:
          # EAST
          y_east_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'east',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'date',start_index,end_index),
              y = y_east_sd - y_east_sd.median() ,
              mode = 'markers',
              name = "%s"%selectData['points'][sd]['text']+" "+"EAST"
            ), 1, 1)
          # NORTH
          y_north_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'north',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'date',start_index,end_index),
              y = y_north_sd - y_north_sd.median(),
              mode = 'markers',
              name = "%s"%selectData['points'][sd]['text']+" "+"NORTH"

            ), 2, 1)
          # UP
          y_up_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'up',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'date',start_index,end_index),
              y = y_up_sd - y_up_sd.median() ,
              mode = 'markers',
              name = "%s"%selectData['points'][sd]['text']+" "+"UP"
            ), 3, 1)
        # If remove median not selected
        else:
          # EAST
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'date',start_index,end_index),
              y = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'east',start_index,end_index).astype(float),
              mode = 'markers',
              name = "%s"%selectData['points'][sd]['text']+" "+"EAST"
            ), 1, 1)
          # NORTH
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'date',start_index,end_index),
              y = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'north',start_index,end_index).astype(float),
              mode = 'markers',
              name = "%s"%selectData['points'][sd]['text']+" "+"NORTH"

            ), 2, 1)
          # UP
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'date',start_index,end_index),
              y = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos',rate_val,'up',start_index,end_index).astype(float),
              mode = 'markers',
              name = "%s"%selectData['points'][sd]['text']+" "+"UP"
            ), 3, 1)

     # Update axes properties and layout
    fig_coord.update_xaxes(title_text="Date",gridcolor='#2D2D2D', gridwidth=0.2,row=3, col=1)
    fig_coord.update_xaxes(gridcolor='#2D2D2D', gridwidth=0.2,row=1, col=1)
    fig_coord.update_xaxes(gridcolor='#2D2D2D', gridwidth=0.2,row=2, col=1)
    fig_coord.update_yaxes(title_text="[m]",gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False, row=1, col=1)
    fig_coord.update_yaxes(title_text="[m]",gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False, row=2, col=1)
    fig_coord.update_yaxes(title_text="[m]", gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False,row=3, col=1)
    fig_coord['layout'].update(height=600,width=750,paper_bgcolor  ='#111111',plot_bgcolor='#111111', font={'color':'#7FDBFF'},
              margin= {
                'l': 40,
                'r': 5,
                'b': 0,
                't': 15
              })
    fig_coord.update_xaxes( rangeselector=dict(
        buttons=list([
          dict(count=1,
               label="1m",
               step="month",
               stepmode="backward"),
          dict(count=3,
               label="3m",
               step="month",
               stepmode="backward"),
          dict(count=6,
               label="6m",
               step="month",
               stepmode="backward"),
          dict(count=1,
               label="1y",
               step="year",
               stepmode="backward"),          
          dict(count=1,
               label="YTD",
               step="year",
               stepmode="todate")
            ])),
      type="date",
      row=1,
      col=1
    )

    return fig_coord

  # Check if any receiver is selected
  if selected_dropdown_value != []:
    get_pos(rate_val)   
    if df_pos["%s"%selected_dropdown_value[0]+"_pos_"+str(rate_val)].empty != True:
      min_y=df_pos["%s"%selected_dropdown_value[0]+"_pos_"+str(rate_val)]['date'].min().year
      min_d=df_pos["%s"%selected_dropdown_value[0]+"_pos_"+str(rate_val)]['date'].min().day
      max_y=df_pos["%s"%selected_dropdown_value[0]+"_pos_"+str(rate_val)]['date'].max().year
      max_d=df_pos["%s"%selected_dropdown_value[0]+"_pos_"+str(rate_val)]['date'].max().day
      if selected_range is not None:
        if selected_range != []:
          start_date=dt(min_y,selected_range[0],min_d)
          end_date=dt(max_y,selected_range[1],max_d)
          print(min_y,min_d,max_y,max_d,start_date,end_date)
          start_index,end_index=extract_index(df_pos,selected_dropdown_value[0],'pos',rate_val,'date',start_date,end_date)
    # For each receiver
    for rec in selected_dropdown_value:
      east, north, up = pyproj.transform(ecef, enu, df_pos["%s"%rec+"_pos_"+str(rate_val)]['pos_x'].values, df_pos["%s"%rec+"_pos_"+str(rate_val)]['pos_y'].values, \
       df_pos["%s"%rec+"_pos_"+str(rate_val)]['pos_z'].values, radians=False)
      df_pos["%s"%rec+"_pos_"+str(rate_val)]['east'] = east
      df_pos["%s"%rec+"_pos_"+str(rate_val)]['north'] = north
      df_pos["%s"%rec+"_pos_"+str(rate_val)]['up'] = up

      # Check reference receiver to be substracted
      if ref is not None:
        east, north, up = pyproj.transform(ecef, enu, df_pos["%s"%ref+"_pos_"+str(rate_val)]['pos_x'].values, df_pos["%s"%ref+"_pos_"+str(rate_val)]['pos_y'].values, df_pos["%s"%ref+"_pos_"+str(rate_val)]['pos_z'].values, radians=False)
        df_pos["%s"%ref+"_pos_"+str(rate_val)]['east'] = east
        df_pos["%s"%ref+"_pos_"+str(rate_val)]['north'] = north
        df_pos["%s"%ref+"_pos_"+str(rate_val)]['up'] = up
        # If remove median equal to True
        if 'True' in values:
        # EAST

          y_east_sd = extract_y_axis_values(df_pos,rec,'pos',rate_val,'east',start_index,end_index).astype(float)
          y_east_ref = extract_y_axis_values(df_pos,ref,'pos',rate_val,'east',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,rec,'pos',rate_val,'date',start_index,end_index),
              y = y_east_sd - y_east_sd.median() - y_east_ref + y_east_ref.median(),
             mode = 'markers',
              name = "%s"%rec+" "+"EAST"
            ), 1, 1)
          # NORTH
          y_north_sd = extract_y_axis_values(df_pos,rec,'pos',rate_val,'north',start_index,end_index).astype(float)
          y_north_ref = extract_y_axis_values(df_pos,ref,'pos',rate_val,'north',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,rec,'pos',rate_val,'date',start_index,end_index),
              y = y_north_sd - y_north_sd.median() - y_north_ref + y_north_ref.median(),
              mode = 'markers',
              name = "%s"%rec+" "+"NORTH"
            ), 2, 1)
          # UP
          y_up_sd = extract_y_axis_values(df_pos,rec,'pos',rate_val,'up',start_index,end_index).astype(float)
          y_up_ref = extract_y_axis_values(df_pos,ref,'pos',rate_val,'up',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,rec,'pos',rate_val,'date',start_index,end_index),
              y = y_up_sd - y_up_sd.median() - y_up_ref + y_up_ref.median(),
              mode = 'markers',
              name = "%s"%rec+" "+"UP"
            ), 3, 1)

        # If remove median is False
        else:
          # EAST
          y_east_sd = extract_y_axis_values(df_pos,rec,'pos',rate_val,'east',start_index,end_index).astype(float)
          y_east_ref = extract_y_axis_values(df_pos,ref,'pos',rate_val,'east',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,rec,'pos',rate_val,'date',start_index,end_index),
              y = y_east_sd - y_east_ref ,
              mode = 'markers',
              name = "%s"%rec+" "+"EAST"
            ), 1, 1)
          # NORTH
          y_north_sd = extract_y_axis_values(df_pos,rec,'pos',rate_val,'north',start_index,end_index).astype(float)
          y_north_ref = extract_y_axis_values(df_pos,ref,'pos',rate_val,'north',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,rec,'pos',rate_val,'date',start_index,end_index),
              y = y_north_sd - y_north_ref ,
              mode = 'markers',
              name = "%s"%rec+" "+"NORTH"
            ), 2, 1)
          # UP
          y_up_sd = extract_y_axis_values(df_pos,rec,'pos',rate_val,'up',start_index,end_index).astype(float)
          y_up_ref = extract_y_axis_values(df_pos,ref,'pos',rate_val,'up',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,rec,'pos',rate_val,'date',start_index,end_index),
              y = y_up_sd  - y_up_ref ,
              mode = 'markers',
              name = "%s"%rec+" "+"UP"
            ), 3, 1)

      # If no reference receiver is selected
      else:
        # Check value remove median box
        if 'True' in values:
          # EAST
          y_east_sd = extract_y_axis_values(df_pos,rec,'pos',rate_val,'east',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,rec,'pos',rate_val,'date',start_index,end_index),
              y = y_east_sd - y_east_sd.median() ,
              mode = 'markers',
              name = "%s"%rec+" "+"EAST"
            ), 1, 1)
          # NORTH
          y_north_sd = extract_y_axis_values(df_pos,rec,'pos',rate_val,'north',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,rec,'pos',rate_val,'date',start_index,end_index),
              y = y_north_sd - y_north_sd.median() ,
              mode = 'markers',
              name = "%s"%rec+" "+"NORTH"

            ), 2, 1)
          # UP
          y_up_sd = extract_y_axis_values(df_pos,rec,'pos',rate_val,'up',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,rec,'pos',rate_val,'date',start_index,end_index),
              y = y_up_sd - y_up_sd.median() ,
              mode = 'markers',
              name = "%s"%rec+" "+"UP"
            ), 3, 1)
        # If remove median not selected
        else:
          # EAST
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,rec,'pos',rate_val,'date',start_index,end_index),
              y = extract_y_axis_values(df_pos,rec,'pos',rate_val,'east',start_index,end_index).astype(float),
              mode = 'markers',
              name = "%s"%rec+" "+"EAST"
            ), 1, 1)
          # NORTH
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,rec,'pos',rate_val,'date',start_index,end_index),
              y = extract_y_axis_values(df_pos,rec,'pos',rate_val,'north',start_index,end_index).astype(float),
              mode = 'markers',
              name = "%s"%rec+" "+"NORTH"

            ), 2, 1)
          # UP
          fig_coord.append_trace(go.Scattergl(
              x = extract_x_axis_values(df_pos,rec,'pos',rate_val,'date',start_index,end_index),
              y = extract_y_axis_values(df_pos,rec,'pos',rate_val,'up',start_index,end_index).astype(float),
              mode = 'markers',
              name = "%s"%rec+" "+"UP"
            ), 3, 1)

     # Update axes properties and layout
    fig_coord.update_xaxes(title_text="Date",gridcolor='#2D2D2D', gridwidth=0.2,row=3, col=1)
    fig_coord.update_xaxes(gridcolor='#2D2D2D', gridwidth=0.2,row=1, col=1)
    fig_coord.update_xaxes(gridcolor='#2D2D2D', gridwidth=0.2,row=2, col=1)
    fig_coord.update_yaxes(title_text="[m]",gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False, row=1, col=1)
    fig_coord.update_yaxes(title_text="[m]",gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False, row=2, col=1)
    fig_coord.update_yaxes(title_text="[m]", gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False,row=3, col=1)
    fig_coord['layout'].update(height=600,width=750,paper_bgcolor  ='#111111',plot_bgcolor='#111111', font={'color':'#7FDBFF'},
              margin= {
                'l': 40,
                'r': 5,
                'b': 0,
                't': 15
              })
    fig_coord.update_xaxes( rangeselector=dict(
        buttons=list([
          dict(count=1,
               label="1m",
               step="month",
               stepmode="backward"),
          dict(count=3,
               label="3m",
               step="month",
               stepmode="backward"),
          dict(count=6,
               label="6m",
               step="month",
               stepmode="backward"),
          dict(count=1,
               label="1y",
               step="year",
               stepmode="backward"),          
          dict(count=1,
               label="YTD",
               step="year",
               stepmode="todate")
        ])),
      type="date",
      row=1,
      col=1
    )

    return fig_coord
  # If no receiver has been selected
  else:
    fig_coord={
      'layout': go.Layout(
            title= {
              'text': 'Please select a receiver',
              'xanchor': 'center',
              'x': 0.5,
              'xref': 'paper',
              'font' :{'size':15, 'color' : '#7FDBFF'}
              },
            xaxis={'title': 'Date', 'automargin':True},
            yaxis={'title': '[m]', 'rangemode':'nonnegative'},
            hovermode='closest',
            plot_bgcolor= colors['background'],
            paper_bgcolor= colors['background'],
            font= {'color': colors['text']},
            height=250
      )
    }

  return fig_coord

# Callback download
@app.callback(
    dash.dependencies.Output('output-container-date-picker-range', 'children'),
    [dash.dependencies.Input('my-date-picker-range', 'start_date'),
     dash.dependencies.Input('my-date-picker-range', 'end_date')])
def update_output_date(start_date, end_date):
    string_prefix = 'You have selected: '
    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        print('the start_date_object is {}'.format(start_date_object))
        start_date_string = start_date_object.strftime('%B %d, %Y')
        print('the start_date_string is {}'.format(start_date_string))
        string_prefix = string_prefix + 'Start Date: ' + start_date_string + ' | '
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'End Date: ' + end_date_string
    if len(string_prefix) == len('You have selected: '):
        return 'Please select initial and final date'
    else:
        return string_prefix

@app.callback(
    dash.dependencies.Output('output-container-param', 'children'),
    [dash.dependencies.Input('down_rec', 'value'),
     dash.dependencies.Input('down_param', 'value'),
     dash.dependencies.Input('down_timestamp', 'value')])
def update_output_param(d_rec, d_param, d_timestamp):
    string_prefix = 'You have selected: '
    string_rec = 'Receivers: '
    string_param = 'Parameters: '
    string_timestamp ='Timestamp: '

    if d_rec is None or d_rec == [] or  d_param is None  or d_param == [] or d_timestamp is  None: 
      return 'Please select all the required inputs'
    else:
      string_rec = string_rec + "%s"%d_rec + ' | '
      string_param = string_param +"%s"%d_param + ' | '
      string_timestamp = string_timestamp + "%s"%d_timestamp + ' | '
      string_prefix = string_prefix + string_rec + string_param + string_timestamp  
      return string_prefix


@app.callback(
    dash.dependencies.Output('output-container-download', 'children'),
    [dash.dependencies.Input('down_rec', 'value'),
     dash.dependencies.Input('down_param', 'value'),
     dash.dependencies.Input('down_timestamp', 'value'),
     dash.dependencies.Input('my-date-picker-range', 'start_date'),
     dash.dependencies.Input('my-date-picker-range', 'end_date'),
     dash.dependencies.Input("btn", "n_clicks")])
def generate_csv(d_rec, d_param, d_timestamp, d_start_date, d_end_date, n_clicks):
    start_index_pos=""
    end_index_pos=""
    start_index_tropo=""
    end_index_tropo=""
    path_to_download_folder = str(os.path.join(Path.home(), "Downloads"))
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn' in changed_id:
      if d_rec != None and d_param != None and d_timestamp != None :
        print('the n_clicks has value {}'.format(n_clicks))  
        if d_timestamp=='5min':
          arg_t='300'
        elif d_timestamp=='1min':
          arg_t='60'
        elif d_timestamp=='30sec':
          arg_t='1'
        print('the d_rec has value {}'.format(d_rec))  
        for i in d_rec:
          arg_r="%s"%i
          print('the i has value {}'.format(i))  
          for a in d_param:
            arg_p=a
            if a=='pos':
              get_pos(arg_t)
              arg =  arg_r+'_'+arg_p+'_'+arg_t
              start_index_pos,end_index_pos=extract_index(df_pos,arg_r,arg_p,arg_t,'date',d_start_date,d_end_date)
              filename=arg_r+'_'+arg_p+'_'+d_timestamp
              df_pos[arg][start_index_pos:end_index_pos].to_csv("%s"%path_to_download_folder+'/'+filename+'.csv', index=False,encoding='utf-8', columns=['date','pos_x','pos_y','pos_z'])
            else: 
              if arg_p!='ZHD':
                get_db(d_rec,arg_t)
                arg =  arg_r+'_'+arg_p+'_'+arg_t
                start_index_tropo,end_index_tropo=extract_index(df_tropo,arg_r,arg_p,arg_t,'date',d_start_date,d_end_date)
                filename=arg_r+'_'+arg_p+'_'+d_timestamp
                df_tropo[arg][start_index_tropo:end_index_tropo].to_csv("%s"%path_to_download_folder+'/'+filename+'.csv', index=False,encoding='utf-8', columns=['date','data_val'])
              else:
                get_db(d_rec,arg_t)
                arg =  arg_r+'_'+arg_p+'_'+arg_t
                start_index_tropo,end_index_tropo=extract_index(df_tropo,arg_r,'ZTD',arg_t,'date',d_start_date,d_end_date)
                filename=arg_r+'_'+arg_p+'_'+d_timestamp
                zhd_df = pd.DataFrame(columns=['date','data_val'])
                zhd_df['date']=df_tropo[arg_r+'_ZTD_'+arg_t]['date'][start_index_tropo:end_index_tropo]
                zhd_df['data_val']=df_tropo[arg_r+'_ZTD_'+arg_t]['data_val'][start_index_tropo:end_index_tropo]-df_tropo[arg_r+'_ZWD_'+arg_t]['data_val'][start_index_tropo:end_index_tropo]
                zhd_df.to_csv("%s"%path_to_download_folder+'/'+filename+'.csv', index=False,encoding='utf-8', columns=['date','data_val'])
          

        return 'Download ended check in download folder'
 

              

if __name__ == '__main__':
    app.run_server(debug=True)

