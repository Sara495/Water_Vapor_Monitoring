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

# DATA VISUALIZATION EVERY 5 MINUTES

mapbox_access_token = open("mapbox_token.txt").read()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#Access to database
myFile = open('dbConfig.txt')
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
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
enu = pyproj.Proj(proj='utm',zone='32N',ellps='WGS84', datum='WGS84')
lon, lat, alt = pyproj.transform(ecef, lla, ggm_table['a_priori_x'].values, ggm_table['a_priori_y'].values,  ggm_table['a_priori_z'].values, radians=False)
ggm_table['lon'] = lon
ggm_table['lat'] = lat
ggm_table['alt'] = alt
receivers=ggm_table['short_name_4ch']

# Tropospheric params
types = ['ZTD','ZWD','PWV','PRESSURE','TEMPERATURE','HUMIDITY','N_SAT']
# Load all the tropo values of the first receiver from the database and create the first dataframe in the dictionary df_tropo
start=time.time()
df_tropo= {"%s"%receivers[0]+"_"+tf: pd.read_sql_query(""" select epoch_time, data_val , flag_tropo from gnsstropo where flag_tropo=0 and epoch_time-floor(epoch_time/%s)*%s=0  and id_tropo_table in 
  (select id_tropo_table from troposet 
  where type= %s 
  and troposet.id_result=(select id_result from ggm where short_name_4ch=%s ) ) order by epoch_time
  """,engine,params=(300,300,tf,receivers[0]))for tf in types }
end=time.time()
print('total extract index time is {}'.format( end-start))
for key in df_tropo.items():
    df_tropo[key[0]]['date']=pd.to_datetime(df_tropo[key[0]]['epoch_time'],unit='s')

# Function to load from the db the receiver selected from the user and add its dataframe the to the dictionary df_tropo
list_key=[]
def get_db (menu_list):
  for rec in list(df_tropo.keys()):
    list_key.append( rec.split('_')[0])
    imported_rec=list(set(list_key))
  print(imported_rec)
  for selected_dv in menu_list:
    if selected_dv in imported_rec:
      print('already imported')
    else:
      df_update= {"%s"%selected_dv+"_"+tf: pd.read_sql_query(""" select epoch_time, data_val ,flag_tropo from gnsstropo where flag_tropo=0  and epoch_time-floor(epoch_time/%s)*%s=0  and id_tropo_table in 
        (select id_tropo_table from troposet 
        where type= %s
        and troposet.id_result=(select id_result from ggm where short_name_4ch=%s ) ) order by epoch_time
        """,engine,params=(300,300,tf,selected_dv)) for tf in types}
      for key in df_update.items():
        df_update[key[0]]['date'] = pd.to_datetime(df_update[key[0]]['epoch_time'], unit='s')
      df_tropo.update(df_update)
  return(df_tropo)

# Same function to load from the db the receiver selected from the user from the points on the map 
def get_db_points (selectData):
  for rec in list(df_tropo.keys()):
    list_key.append( rec.split('_')[0])
    imported_rec=list(set(list_key))
  print(imported_rec)

  for sd in range(int(len(selectData['points']))):
    if selectData['points'][sd]['text'] in imported_rec:
      print('already imported')
    else:
      df_update= {"%s"%selectData['points'][sd]['text']+"_"+tf: pd.read_sql_query(""" select epoch_time, data_val, flag_tropo from gnsstropo where flag_tropo=0 and epoch_time-floor(epoch_time/%s)*%s=0  and id_tropo_table in 
        (select id_tropo_table from troposet 
        where type= %s
        and troposet.id_result=(select id_result from ggm where short_name_4ch=%s ) ) order by epoch_time
        """,engine,params=(300,300,tf,selectData['points'][sd]['text'])) for tf in types}
      for key in df_update.items():
        df_update[key[0]]['date'] = pd.to_datetime(df_update[key[0]]['epoch_time'], unit='s')
      df_tropo.update(df_update)
  return(df_tropo)

# For each receiver extract position values
# In this case all the coord values together cause data are few
start=time.time()
df_pos = {"%s"%sn+"_pos": pd.read_sql_query(""" select * from gnssposition where flag_pos=0 and pos_time-floor(pos_time/%s)*%s=0  and id_pos_table in 
  (select id_pos_table from positionset 
  where positionset.id_result=(select id_result from ggm where short_name_4ch=%s ) ) order by pos_time 
  """,engine,params=(300,300,sn,)) for sn in ggm_table['short_name_4ch'] }
end=time.time()
print('pos extract  time is {}'.format( end-start))


# Set layout colors
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# Convert from unix time into datetime and create a new column 'date' for each dataframe with converted time
for key in df_tropo.items():
    df_tropo[key[0]]['date']=pd.to_datetime(df_tropo[key[0]]['epoch_time'],unit='s')
for key in df_pos.items():
    df_pos[key[0]]['date']=pd.to_datetime(df_pos[key[0]]['pos_time'],unit='s')

# Function that extracts the indeces of the dataframe of interest to be used as condition
#for the extraction of the axis values in the extract values funcion
#parameters are df=name of the dataframe, param=tropo param,time_param=time column of the dataframe
#start_date,end_date= boundaries of the rangeslider
start=time.time()
def extract_index(df,name_rec,param,time_param,start_date,end_date):

  tmp_time = df["%s"%name_rec+"_"+param][time_param]
  start_index= df["%s"%name_rec+"_"+param].loc[(tmp_time>=start_date),[time_param]].index[0]
  end_index=df["%s"%name_rec+"_"+param].loc[(tmp_time<=end_date),[time_param]].index[-1]

  return start_index,end_index
end=time.time()
print('total extract index time is {}'.format(end-start))

# Function that extracts the axis values, respectively time and value
#parameters are df=name of the dataframe, param=tropo param,time_param=time column of the dataframe
#value_param= y axis variable, start_index,end_index= boundaries of the rangeslider
def extract_x_axis_values(df,name_rec,param,time_param,start_index,end_index):
    # return epoch time for time axis
    x= df["%s"%name_rec+"_"+param].loc[start_index:end_index][time_param]
    return x
def extract_y_axis_values(df,name_rec,param,value_param,start_index,end_index):
    # return values for y axes
    y= df["%s"%name_rec+"_"+param].loc[start_index:end_index][value_param]
    return y


logo_gred = '/Users/saramaffioli/Desktop/dash/GReD_logo.png' # replace with your own image
encoded_image_gred = base64.b64encode(open(logo_gred, 'rb').read())
logo_polimi = '/Users/saramaffioli/Desktop/dash/poli_logo.png' # replace with your own image
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
      # START MAP + TABLE
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

        # table with ggm information name active constellations and observer
        html.Details([
          html.Summary('Table with receivers description'),
          html.Div([
            dcc.Loading(id='loading_table', children=[
            dcc.Graph(
              id = 'table_receivers')],type="circle") ],style={'backgroundColor': '#111111', 'display':'inline-block','font-weight': 'bold'})
          ],style={'backgroundColor': colors['background'],'textAlign': 'left','color': colors['text'],'padding':'0 0 20px 50px' } ,open=True)

      ],style={'textAlign': 'center','color': colors['text'],'display':'inline-block','padding':'0px 20px 0 0'}),
      # END MAP+ TABLE

      # START TEMPORAL SERIES
      # Plot wv values with type selector and list receivers
      html.Div([
        html.Details([
          html.Summary('Tropospheric delays time series'),
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
          html.Summary('Coordinates time series'),
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
        )],style={'width':600,'backgroundColor': colors['background'],'padding':'20px 0 10px 90px'}),
        # Temporal series pwv
        html.Details([
          html.Summary('PWV time series'),
          dcc.Loading(id='loading_pwv_series', children=[
          dcc.Graph(
            id='pwv_temp_series',
            config={'displaylogo':False,
              'displayModeBar':'hover',
              'modeBarButtonsToRemove': ['hoverCompareCartesian','hoverClosestCartesian','toggleSpikelines']}

        )],type="circle")],style={'backgroundColor': colors['background'],'textAlign': 'left','color': colors['text'],'padding':'20px 0 20px 50px' }),

        # Temporal series for others tropo types values
        html.Details([
          html.Summary('Weather stations PTH series'),
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
@app.callback(Output("loading_table", "children"))
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
  dash.dependencies.Input('map','selectedData')])
def update_rangeslider(selected_dropdown_value,selectData):
  if selectData != None and selectData['points'] !=[]:
    get_db_points(selectData)
    print(selectData)
    print('the points are {}'.format(selectData['points']))
    print(selectData['points'][0]['text'])
    for sd in range(int(len(selectData['points']))):

      min_val=df_tropo["%s"%selectData['points'][sd]['text']+"_ZTD"]['date'].min().month
      max_val=df_tropo["%s"%selectData['points'][sd]['text']+"_ZTD"]['date'].max().month
      marks_val = {date.month:{'label':"%s"%date.strftime("%B")+" "+str(date.year),'style':{'color':'white','font-size':'120%','font-weight': 'bold'}} for date in df_tropo["%s"%selectData['points'][sd]['text']+"_ZTD"]['date'].dt.date.unique()}

      value = [min_val,max_val]
      marks = marks_val

      return min_val,max_val,value,marks

  # Check if any receiver is selected
  if selected_dropdown_value != []:
    get_db(selected_dropdown_value)
    # For each selected receiver
    for rec in selected_dropdown_value:
      min_val=df_tropo["%s"%rec+"_ZTD"]['date'].min().month
      max_val=df_tropo["%s"%rec+"_ZTD"]['date'].max().month
      marks_val = {date.month:{'label':"%s"%date.strftime("%B")+" "+str(date.year),'style':{'color':'white','font-size':'120%','font-weight': 'bold'}} for date in df_tropo["%s"%rec+"_ZTD"]['date'].dt.date.unique()}

    value = [min_val,max_val]
    marks = marks_val

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
    dash.dependencies.Input('map','selectedData')])
def update_time_series(selected_dropdown_value,selected_type,selected_range,selectData):
  print(selected_range)
  trace=[]
  start_date=""
  end_date=""
  start_index=""
  end_index=""

   # Check if any point is selected
  if selectData != None and selectData['points'] !=[]:
    get_db_points(selectData)
    min_y=df_tropo["%s"%selectData['points'][0]['text']+"_ZTD"]['date'].min().year
    min_d=df_tropo["%s"%selectData['points'][0]['text']+"_ZTD"]['date'].min().day
    max_y=df_tropo["%s"%selectData['points'][0]['text']+"_ZTD"]['date'].max().year
    max_d=df_tropo["%s"%selectData['points'][0]['text']+"_ZTD"]['date'].max().day
    if selected_range is not None:
      if selected_range != []:
        start_date=dt(min_y,selected_range[0],min_d)
        end_date=dt(max_y,selected_range[1],max_d+1)
        print(min_y,min_d,max_y,max_d,start_date,end_date)
        start_index,end_index=extract_index(df_tropo,selectData['points'][0]['text'],'ZTD','date',start_date,end_date)

    # Check if a delay is selected
    if selected_type != []:
      # For each selected receiver
      for sd in range(int(len(selectData['points']))):
        # For each selected delay
        for tropo_type in selected_type:
          if tropo_type == 'ZHD':
            y_ztd = extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],'ZTD','data_val',start_index,end_index)
            y_zwd = extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],'ZWD','data_val',start_index,end_index)
            trace.append(go.Scatter(
              x = extract_x_axis_values(df_tropo,selectData['points'][sd]['text'],'ZTD','date',start_index,end_index),
              y = (y_ztd.astype(float) - y_zwd.astype(float))*100,
              mode= 'lines',
              name= "%s"%selectData['points'][sd]['text']+"_ZHD"
            ))
          else:

            trace.append(go.Scatter(
              x = extract_x_axis_values(df_tropo,selectData['points'][sd]['text'],tropo_type,'date',start_index,end_index),
              y = (extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],tropo_type,'data_val',start_index,end_index).astype(float))*100,
              mode= 'lines',
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
              xaxis={'title': 'Date', 'automargin':True},
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
            xaxis={'title': 'Date', 'automargin':True},
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
    get_db(selected_dropdown_value)
    start_plot_fig=time.time()
    min_y=df_tropo["%s"%selected_dropdown_value[0]+"_ZTD"]['date'].min().year
    min_d=df_tropo["%s"%selected_dropdown_value[0]+"_ZTD"]['date'].min().day
    max_y=df_tropo["%s"%selected_dropdown_value[0]+"_ZTD"]['date'].max().year
    max_d=df_tropo["%s"%selected_dropdown_value[0]+"_ZTD"]['date'].max().day
    if selected_range is not None:
      if selected_range != []:
        start_date=dt(min_y,selected_range[0],min_d)
        end_date=dt(max_y,selected_range[1],max_d+1)
        print(min_y,min_d,max_y,max_d,start_date,end_date)
        start_time=time.time()
        start_index,end_index=extract_index(df_tropo,selected_dropdown_value[0],'ZTD','date',start_date,end_date)
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
            y_ztd = extract_y_axis_values(df_tropo,rec,'ZTD','data_val',start_index,end_index)
            end_zhd=time.time()
            print('the extract y values is {}'.format(end_zhd-start_zhd))
            y_zwd = extract_y_axis_values(df_tropo,rec,'ZWD','data_val',start_index,end_index)
            trace.append(go.Scatter(
              x = extract_x_axis_values(df_tropo,rec,'ZTD','date',start_index,end_index),
              y = (y_ztd.astype(float) - y_zwd.astype(float))*100,
              mode= 'lines',
              name= "%s"%rec+"_ZHD"
            ))
          else:
            start_append=time.time()
            trace.append(go.Scatter(
              x = extract_x_axis_values(df_tropo,rec,tropo_type,'date',start_index,end_index),
              y = (extract_y_axis_values(df_tropo,rec,tropo_type,'data_val',start_index,end_index).astype(float))*100,
              mode= 'lines',
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
              xaxis={'title': 'Date', 'automargin':True},
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
            xaxis={'title': 'Date', 'automargin':True},
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
            xaxis={'title': 'Date', 'automargin':True},
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
    dash.dependencies.Input('map','selectedData')])
def update_pwv_series(selected_dropdown_value,selected_range,selectData):
  trace_pwv=[]
  start_date=""
  end_date=""
  start_index=""
  end_index=""
  # Check if any point is selected
  if selectData != None and selectData['points'] !=[]:
    get_db_points(selectData)
    if df_tropo["%s"%selectData['points'][0]['text']+"_PWV"].empty != True:
      min_y=df_tropo["%s"%selectData['points'][0]['text']+"_PWV"]['date'].min().year
      min_d=df_tropo["%s"%selectData['points'][0]['text']+"_PWV"]['date'].min().day
      max_y=df_tropo["%s"%selectData['points'][0]['text']+"_PWV"]['date'].max().year
      max_d=df_tropo["%s"%selectData['points'][0]['text']+"_PWV"]['date'].max().day
      if selected_range is not None:
        if selected_range != []:
          start_date=dt(min_y,selected_range[0],min_d)
          end_date=dt(max_y,selected_range[1],max_d+1)
          print(min_y,min_d,max_y,max_d,start_date,end_date)
          start_index,end_index=extract_index(df_tropo,selectData['points'][0]['text'],'PWV','date',start_date,end_date)
      # For each selected receiver
    for sd in range(int(len(selectData['points']))):
      if df_tropo["%s"%selectData['points'][sd]['text']+"_PWV"].empty == True:
        trace_pwv.append(go.Scatter(
          x = [2],
          y = [2],
          mode = 'lines+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name= "%s"%selectData['points'][sd]['text']+"_PWV"
        ))
      else:
        trace_pwv.append(go.Scatter(
          x = extract_x_axis_values(df_tropo,selectData['points'][sd]['text'],'PWV','date',start_index,end_index),
          y = (extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],'PWV','data_val',start_index,end_index).astype(float))*1000,
          mode= 'lines',
          name= "%s"%selectData['points'][sd]['text']+"_PWV"
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
            xaxis={'title': 'Date', 'automargin':True},
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
    get_db(selected_dropdown_value)
    if df_tropo["%s"%selected_dropdown_value[0]+"_PWV"].empty != True:
      min_y=df_tropo["%s"%selected_dropdown_value[0]+"_PWV"]['date'].min().year
      min_d=df_tropo["%s"%selected_dropdown_value[0]+"_PWV"]['date'].min().day
      max_y=df_tropo["%s"%selected_dropdown_value[0]+"_PWV"]['date'].max().year
      max_d=df_tropo["%s"%selected_dropdown_value[0]+"_PWV"]['date'].max().day
      if selected_range is not None:
        if selected_range != []:
          start_date=dt(min_y,selected_range[0],min_d)
          end_date=dt(max_y,selected_range[1],max_d+1)
          print(min_y,min_d,max_y,max_d,start_date,end_date)
          start_index,end_index=extract_index(df_tropo,selected_dropdown_value[0],'ZTD','date',start_date,end_date)
    # For each selected receiver
    for rec in selected_dropdown_value:
      if df_tropo["%s"%rec+"_PWV"].empty == True:
        trace_pwv.append(go.Scatter(
          x = [2],
          y = [2],
          mode = 'lines+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name= "%s"%rec+"_PWV"
        ))
      else:
        trace_pwv.append(go.Scatter(
          x = extract_x_axis_values(df_tropo,rec,'PWV','date',start_index,end_index),
          y = (extract_y_axis_values(df_tropo,rec,'PWV','data_val',start_index,end_index).astype(float))*1000,
          mode= 'lines',
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
            xaxis={'title': 'Date', 'automargin':True},
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
            xaxis={'title': 'Date', 'automargin':True},
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
        'zoom': 6.9,
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
        'zoom': 6.9,
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
        'zoom': 6.9,
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
  dash.dependencies.Input('map','selectedData')])
def update_tropo_series(selected_dropdown_value,selected_range,selectData):
  fig_tropo = make_subplots(rows=3, cols=1, specs=[[{}], [{}],[{}]],shared_xaxes=True,vertical_spacing=0.08,row_width=[15, 15, 15],
    subplot_titles=("PRESSURE","TEMPERATURE","HUMIDITY"))
  start_date=""
  end_date=""
  start_index=""
  end_index=""
  # Check if any point is selected
  if selectData != None and selectData['points'] !=[]:
    get_db_points(selectData)
    if df_tropo["%s"%selectData['points'][0]['text']+"_PRESSURE"].empty != True:
      min_y=df_tropo["%s"%selectData['points'][0]['text']+"_PRESSURE"]['date'].min().year
      min_d=df_tropo["%s"%selectData['points'][0]['text']+"_PRESSURE"]['date'].min().day
      max_y=df_tropo["%s"%selectData['points'][0]['text']+"_PRESSURE"]['date'].max().year
      max_d=df_tropo["%s"%selectData['points'][0]['text']+"_PRESSURE"]['date'].max().day
      print('TROPO DATE{}'.format(min_y,min_d,max_y,max_d))
      if selected_range is not None:
        if selected_range != []:
          start_date=dt(min_y,selected_range[0],min_d)
          end_date=dt(max_y,selected_range[1],max_d+1)
          print(min_y,min_d,max_y,max_d,start_date,end_date)
          start_index,end_index=extract_index(df_tropo,selectData['points'][0]['text'],'PRESSURE','date',start_date,end_date)

    for sd in range(int(len(selectData['points']))):
            # Pressure
      if df_tropo["%s"%selectData['points'][sd]['text']+"_PRESSURE"].empty == True:
        fig_tropo.append_trace(go.Scatter(
          x = [2],
          y = [2],
          mode = 'lines+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name='Pressure {}'.format(selectData['points'][sd]['text'])
        ), 1, 1)

      else:
        fig_tropo.append_trace(go.Scatter(
          x = extract_x_axis_values(df_tropo,selectData['points'][sd]['text'],'PRESSURE','date',start_index,end_index),
          y = extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],'PRESSURE','data_val',start_index,end_index),
          mode='lines',
          name='Pressure {}'.format(selectData['points'][sd]['text'])
        ), 1, 1)

      # Temperature
      if df_tropo["%s"%selectData['points'][sd]['text']+"_TEMPERATURE"].empty == True:
        fig_tropo.append_trace(go.Scatter(
          x = [2],
          y = [2],
          mode = 'lines+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name='Temperature {}'.format(selectData['points'][sd]['text'])
        ), 2, 1)

      else:
        fig_tropo.append_trace(go.Scatter(
          x = extract_x_axis_values(df_tropo,selectData['points'][sd]['text'],'TEMPERATURE','date',start_index,end_index),
          y = extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],'TEMPERATURE','data_val',start_index,end_index),
          mode='lines',
          name='Temperature {}'.format(selectData['points'][sd]['text'])
        ), 2, 1)

      # Humidity
      if df_tropo["%s"%selectData['points'][sd]['text']+"_HUMIDITY"].empty == True:
        fig_tropo.append_trace(go.Scatter(
          x = [2],
          y = [2],
          mode = 'lines+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name='Humidity {}'.format(selectData['points'][sd]['text'])
        ), 3, 1)

      else:
        fig_tropo.append_trace(go.Scatter(
          x = extract_x_axis_values(df_tropo,selectData['points'][sd]['text'],'HUMIDITY','date',start_index,end_index),
          y = extract_y_axis_values(df_tropo,selectData['points'][sd]['text'],'HUMIDITY','data_val',start_index,end_index),
          mode='lines',
          name='Humidity {}'.format(selectData['points'][sd]['text'])
        ), 3, 1)

      # Update axis properties and layout
    fig_tropo.update_xaxes(title_text="Date",gridcolor='#2D2D2D', gridwidth=0.2,row=3, col=1)
    fig_tropo.update_xaxes(gridcolor='#2D2D2D', gridwidth=0.2,row=1, col=1)
    fig_tropo.update_xaxes(gridcolor='#2D2D2D', gridwidth=0.2,row=2, col=1)
    fig_tropo.update_yaxes(title_text="[mbar]",gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False, row=1, col=1)
    fig_tropo.update_yaxes(title_text="[Â°C]",gridcolor='#2D2D2D',gridwidth=0.2,zeroline=False, row=2, col=1)
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
          dict(count=6,
               label="6m",
               step="month",
               stepmode="backward"),
          dict(count=1,
               label="YTD",
               step="year",
               stepmode="todate"),
          dict(count=1,
               label="1y",
               step="year",
               stepmode="backward"),
          dict(step="all")
      ])),
      type="date",
      row=1,
      col=1
    )

    return fig_tropo

  # Check if any receiver is selected
  if selected_dropdown_value != []:
    get_db(selected_dropdown_value)
    if df_tropo["%s"%selected_dropdown_value[0]+"_PRESSURE"].empty != True:
      min_y=df_tropo["%s"%selected_dropdown_value[0]+"_PRESSURE"]['date'].min().year
      min_d=df_tropo["%s"%selected_dropdown_value[0]+"_PRESSURE"]['date'].min().day
      max_y=df_tropo["%s"%selected_dropdown_value[0]+"_PRESSURE"]['date'].max().year
      max_d=df_tropo["%s"%selected_dropdown_value[0]+"_PRESSURE"]['date'].max().day
      if selected_range is not None:
        if selected_range != []:
          start_date=dt(min_y,selected_range[0],min_d)
          end_date=dt(max_y,selected_range[1],max_d+1)
          print(min_y,min_d,max_y,max_d,start_date,end_date)
          start_index,end_index=extract_index(df_tropo,selected_dropdown_value[0],'PRESSURE','date',start_date,end_date)
    # For each selected receiver
    for rec in selected_dropdown_value:

      # Pressure
      if df_tropo["%s"%rec+"_PRESSURE"].empty == True:
        fig_tropo.append_trace(go.Scatter(
          x = [2],
          y = [2],
          mode = 'lines+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name='Pressure {}'.format(rec)
        ), 1, 1)

      else:
        fig_tropo.append_trace(go.Scatter(
          x = extract_x_axis_values(df_tropo,rec,'PRESSURE','date',start_index,end_index),
          y = extract_y_axis_values(df_tropo,rec,'PRESSURE','data_val',start_index,end_index),
          mode='lines',
          name='Pressure {}'.format(rec)
        ), 1, 1)

      # Temperature
      if df_tropo["%s"%rec+"_TEMPERATURE"].empty == True:
        fig_tropo.append_trace(go.Scatter(
          x = [2],
          y = [2],
          mode = 'lines+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name='Temperature {}'.format(rec)
        ), 2, 1)

      else:
        fig_tropo.append_trace(go.Scatter(
          x = extract_x_axis_values(df_tropo,rec,'TEMPERATURE','date',start_index,end_index),
          y = extract_y_axis_values(df_tropo,rec,'TEMPERATURE','data_val',start_index,end_index),
          mode='lines',
          name='Temperature {}'.format(rec)
        ), 2, 1)

      # Humidity
      if df_tropo["%s"%rec+"_HUMIDITY"].empty == True:
        fig_tropo.append_trace(go.Scatter(
          x = [2],
          y = [2],
          mode = 'lines+text',
          text = 'No data available',
          textposition = 'middle center',
          textfont = {'color': '#ff5050', 'size':35},
          name='Humidity {}'.format(rec)
        ), 3, 1)

      else:
        fig_tropo.append_trace(go.Scatter(
          x = extract_x_axis_values(df_tropo,rec,'HUMIDITY','date',start_index,end_index),
          y = extract_y_axis_values(df_tropo,rec,'HUMIDITY','data_val',start_index,end_index),
          mode='lines',
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
          dict(count=6,
               label="6m",
               step="month",
               stepmode="backward"),
          dict(count=1,
               label="YTD",
               step="year",
               stepmode="todate"),
          dict(count=1,
               label="1y",
               step="year",
               stepmode="backward"),
          dict(step="all")
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
  dash.dependencies.Input('map','selectedData')])
def update_coord_series(selected_dropdown_value,values,ref,selected_range,selectData):
  fig_coord = make_subplots(rows=3, cols=1, specs=[[{}],[{}],[{}]],shared_xaxes=True,vertical_spacing=0.08,subplot_titles=('EAST', 'NORTH', 'UP'),
                          row_width=[15, 15, 15])
  start_date=""
  end_date=""
  start_index=""
  end_index=""
  # Check if any point is selected
  if selectData != None and selectData['points'] !=[]:
    min_y=df_pos["%s"%selectData['points'][0]['text']+"_pos"]['date'].min().year
    min_d=df_pos["%s"%selectData['points'][0]['text']+"_pos"]['date'].min().day
    max_y=df_pos["%s"%selectData['points'][0]['text']+"_pos"]['date'].max().year
    max_d=df_pos["%s"%selectData['points'][0]['text']+"_pos"]['date'].max().day
    if selected_range is not None:
      if selected_range != []:
        start_date=dt(min_y,selected_range[0],min_d)
        end_date=dt(max_y,selected_range[1],max_d+1)
        print(min_y,min_d,max_y,max_d,start_date,end_date)
        start_index,end_index=extract_index(df_pos,selectData['points'][0]['text'],'pos','date',start_date,end_date)


    # For each selected receiver
    for sd in range(int(len(selectData['points']))):
      east, north, up = pyproj.transform(ecef, enu, df_pos["%s"%selectData['points'][sd]['text']+"_pos"]['pos_x'].values, df_pos["%s"%selectData['points'][sd]['text']+"_pos"]['pos_y'].values, \
      df_pos["%s"%selectData['points'][sd]['text']+"_pos"]['pos_z'].values, radians=False)
      df_pos["%s"%selectData['points'][sd]['text']+"_pos"]['east'] = east
      df_pos["%s"%selectData['points'][sd]['text']+"_pos"]['north'] = north
      df_pos["%s"%selectData['points'][sd]['text']+"_pos"]['up'] = up

      # Check reference receiver to be substracted
      if ref is not None:
        east, north, up = pyproj.transform(ecef, enu, df_pos["%s"%ref+"_pos"]['pos_x'].values, df_pos["%s"%ref+"_pos"]['pos_y'].values, df_pos["%s"%ref+"_pos"]['pos_z'].values, radians=False)
        df_pos["%s"%ref+"_pos"]['east'] = east
        df_pos["%s"%ref+"_pos"]['north'] = north
        df_pos["%s"%ref+"_pos"]['up'] = up
        # If remove median equal to True
        if 'True' in values:
        # EAST
          y_east_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos','east',start_index,end_index).astype(float)
          y_east_ref = extract_y_axis_values(df_pos,ref,'pos','east',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos','date',start_index,end_index),
              y = y_east_sd - y_east_sd.median() - y_east_ref + y_east_ref.median(),
              mode = 'lines',
              name = "%s"%selectData['points'][sd]['text']+" "+"EAST"
            ), 1, 1)
          # NORTH
          y_north_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos','north',start_index,end_index).astype(float)
          y_north_ref = extract_y_axis_values(df_pos,ref,'pos','north',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos','date',start_index,end_index),
              y = y_north_sd - y_north_sd.median() - y_north_ref + y_north_ref.median(),
              mode = 'lines',
              name = "%s"%selectData['points'][sd]['text']+" "+"NORTH"
            ), 2, 1)
          # UP
          y_up_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos','up',start_index,end_index).astype(float)
          y_up_ref = extract_y_axis_values(df_pos,ref,'pos','up',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos','date',start_index,end_index),
              y = y_up_sd - y_up_sd.median() - y_up_ref + y_up_ref.median(),
              mode = 'lines',
              name = "%s"%selectData['points'][sd]['text']+" "+"UP"
            ), 3, 1)

        # If remove median is False
        else:
          # EAST
          y_east_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos','east',start_index,end_index).astype(float)
          y_east_ref = extract_y_axis_values(df_pos,ref,'pos','east',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos','date',start_index,end_index),
              y = y_east_sd - y_east_ref ,
              mode = 'lines',
              name = "%s"%selectData['points'][sd]['text']+" "+"EAST"
            ), 1, 1)
          # NORTH
          y_north_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos','north',start_index,end_index).astype(float)
          y_north_ref = extract_y_axis_values(df_pos,ref,'pos','north',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos','date',start_index,end_index),
              y = y_north_sd - y_north_ref,
              mode = 'lines',
              name = "%s"%selectData['points'][sd]['text']+" "+"NORTH"
            ), 2, 1)
          # UP
          y_up_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos','up',start_index,end_index).astype(float)
          y_up_ref = extract_y_axis_values(df_pos,ref,'pos','up',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos','date',start_index,end_index),
              y = y_up_sd - y_up_ref ,
              mode = 'lines',
              name = "%s"%selectData['points'][sd]['text']+" "+"UP"
            ), 3, 1)

      # If no reference receiver is selected
      else:
        # Check value remove median box
        if 'True' in values:
          # EAST
          y_east_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos','east',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos','date',start_index,end_index),
              y = y_east_sd - y_east_sd.median() ,
              mode = 'lines',
              name = "%s"%selectData['points'][sd]['text']+" "+"EAST"
            ), 1, 1)
          # NORTH
          y_north_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos','north',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos','date',start_index,end_index),
              y = y_north_sd - y_north_sd.median(),
              mode = 'lines',
              name = "%s"%selectData['points'][sd]['text']+" "+"NORTH"

            ), 2, 1)
          # UP
          y_up_sd = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos','up',start_index,end_index).astype(float)

          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos','date',start_index,end_index),
              y = y_up_sd - y_up_sd.median() ,
              mode = 'lines',
              name = "%s"%selectData['points'][sd]['text']+" "+"UP"
            ), 3, 1)
        # If remove median not selected
        else:
          # EAST
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos','date',start_index,end_index),
              y = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos','east',start_index,end_index).astype(float),
              mode = 'lines',
              name = "%s"%selectData['points'][sd]['text']+" "+"EAST"
            ), 1, 1)
          # NORTH
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos','date',start_index,end_index),
              y = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos','north',start_index,end_index).astype(float),
              mode = 'lines',
              name = "%s"%selectData['points'][sd]['text']+" "+"NORTH"

            ), 2, 1)
          # UP
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,selectData['points'][sd]['text'],'pos','date',start_index,end_index),
              y = extract_y_axis_values(df_pos,selectData['points'][sd]['text'],'pos','up',start_index,end_index).astype(float),
              mode = 'lines',
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
            dict(count=6,
                 label="6m",
                 step="month",
                 stepmode="backward"),
            dict(count=1,
                 label="YTD",
                 step="year",
                 stepmode="todate"),
            dict(count=1,
                 label="1y",
                 step="year",
                 stepmode="backward"),
            dict(step="all")
        ])),
      type="date",
      row=1,
      col=1
    )

    return fig_coord

  # Check if any receiver is selected
  if selected_dropdown_value != []:
    if df_pos["%s"%selected_dropdown_value[0]+"_pos"].empty != True:
      min_y=df_pos["%s"%selected_dropdown_value[0]+"_pos"]['date'].min().year
      min_d=df_pos["%s"%selected_dropdown_value[0]+"_pos"]['date'].min().day
      max_y=df_pos["%s"%selected_dropdown_value[0]+"_pos"]['date'].max().year
      max_d=df_pos["%s"%selected_dropdown_value[0]+"_pos"]['date'].max().day
      if selected_range is not None:
        if selected_range != []:
          start_date=dt(min_y,selected_range[0],min_d)
          end_date=dt(max_y,selected_range[1],max_d+1)
          print(min_y,min_d,max_y,max_d,start_date,end_date)
          start_index,end_index=extract_index(df_pos,selected_dropdown_value[0],'pos','date',start_date,end_date)
    # For each receiver
    for rec in selected_dropdown_value:
      east, north, up = pyproj.transform(ecef, enu, df_pos["%s"%rec+"_pos"]['pos_x'].values, df_pos["%s"%rec+"_pos"]['pos_y'].values, \
       df_pos["%s"%rec+"_pos"]['pos_z'].values, radians=False)
      df_pos["%s"%rec+"_pos"]['east'] = east
      df_pos["%s"%rec+"_pos"]['north'] = north
      df_pos["%s"%rec+"_pos"]['up'] = up

      # Check reference receiver to be substracted
      if ref is not None:
        east, north, up = pyproj.transform(ecef, enu, df_pos["%s"%ref+"_pos"]['pos_x'].values, df_pos["%s"%ref+"_pos"]['pos_y'].values, df_pos["%s"%ref+"_pos"]['pos_z'].values, radians=False)
        df_pos["%s"%ref+"_pos"]['east'] = east
        df_pos["%s"%ref+"_pos"]['north'] = north
        df_pos["%s"%ref+"_pos"]['up'] = up
        # If remove median equal to True
        if 'True' in values:
        # EAST

          y_east_sd = extract_y_axis_values(df_pos,rec,'pos','east',start_index,end_index).astype(float)
          y_east_ref = extract_y_axis_values(df_pos,ref,'pos','east',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,rec,'pos','date',start_index,end_index),
              y = y_east_sd - y_east_sd.median() - y_east_ref + y_east_ref.median(),
             mode = 'lines',
              name = "%s"%rec+" "+"EAST"
            ), 1, 1)
          # NORTH
          y_north_sd = extract_y_axis_values(df_pos,rec,'pos','north',start_index,end_index).astype(float)
          y_north_ref = extract_y_axis_values(df_pos,ref,'pos','north',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,rec,'pos','date',start_index,end_index),
              y = y_north_sd - y_north_sd.median() - y_north_ref + y_north_ref.median(),
              mode = 'lines',
              name = "%s"%rec+" "+"NORTH"
            ), 2, 1)
          # UP
          y_up_sd = extract_y_axis_values(df_pos,rec,'pos','up',start_index,end_index).astype(float)
          y_up_ref = extract_y_axis_values(df_pos,ref,'pos','up',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,rec,'pos','date',start_index,end_index),
              y = y_up_sd - y_up_sd.median() - y_up_ref + y_up_ref.median(),
              mode = 'lines',
              name = "%s"%rec+" "+"UP"
            ), 3, 1)

        # If remove median is False
        else:
          # EAST
          y_east_sd = extract_y_axis_values(df_pos,rec,'pos','east',start_index,end_index).astype(float)
          y_east_ref = extract_y_axis_values(df_pos,ref,'pos','east',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,rec,'pos','date',start_index,end_index),
              y = y_east_sd - y_east_ref ,
              mode = 'lines',
              name = "%s"%rec+" "+"EAST"
            ), 1, 1)
          # NORTH
          y_north_sd = extract_y_axis_values(df_pos,rec,'pos','north',start_index,end_index).astype(float)
          y_north_ref = extract_y_axis_values(df_pos,ref,'pos','north',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,rec,'pos','date',start_index,end_index),
              y = y_north_sd - y_north_ref ,
              mode = 'lines',
              name = "%s"%rec+" "+"NORTH"
            ), 2, 1)
          # UP
          y_up_sd = extract_y_axis_values(df_pos,rec,'pos','up',start_index,end_index).astype(float)
          y_up_ref = extract_y_axis_values(df_pos,ref,'pos','up',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,rec,'pos','date',start_index,end_index),
              y = y_up_sd  - y_up_ref ,
              mode = 'lines',
              name = "%s"%rec+" "+"UP"
            ), 3, 1)

      # If no reference receiver is selected
      else:
        # Check value remove median box
        if 'True' in values:
          # EAST
          y_east_sd = extract_y_axis_values(df_pos,rec,'pos','east',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,rec,'pos','date',start_index,end_index),
              y = y_east_sd - y_east_sd.median() ,
              mode = 'lines',
              name = "%s"%rec+" "+"EAST"
            ), 1, 1)
          # NORTH
          y_north_sd = extract_y_axis_values(df_pos,rec,'pos','north',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,rec,'pos','date',start_index,end_index),
              y = y_north_sd - y_north_sd.median() ,
              mode = 'lines',
              name = "%s"%rec+" "+"NORTH"

            ), 2, 1)
          # UP
          y_up_sd = extract_y_axis_values(df_pos,rec,'pos','up',start_index,end_index).astype(float)
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,rec,'pos','date',start_index,end_index),
              y = y_up_sd - y_up_sd.median() ,
              mode = 'lines',
              name = "%s"%rec+" "+"UP"
            ), 3, 1)
        # If remove median not selected
        else:
          # EAST
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,rec,'pos','date',start_index,end_index),
              y = extract_y_axis_values(df_pos,rec,'pos','east',start_index,end_index).astype(float),
              mode = 'lines',
              name = "%s"%rec+" "+"EAST"
            ), 1, 1)
          # NORTH
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,rec,'pos','date',start_index,end_index),
              y = extract_y_axis_values(df_pos,rec,'pos','north',start_index,end_index).astype(float),
              mode = 'lines',
              name = "%s"%rec+" "+"NORTH"

            ), 2, 1)
          # UP
          fig_coord.append_trace(go.Scatter(
              x = extract_x_axis_values(df_pos,rec,'pos','date',start_index,end_index),
              y = extract_y_axis_values(df_pos,rec,'pos','up',start_index,end_index).astype(float),
              mode = 'lines',
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
            dict(count=6,
                 label="6m",
                 step="month",
                 stepmode="backward"),
            dict(count=1,
                 label="YTD",
                 step="year",
                 stepmode="todate"),
            dict(count=1,
                 label="1y",
                 step="year",
                 stepmode="backward"),
            dict(step="all")
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

# Callback table receivers
@app.callback(
  dash.dependencies.Output('table_receivers','figure'),
  [dash.dependencies.Input('select_receiver_menu','value'),
  dash.dependencies.Input('map','selectedData')])
def update_table(selected_dropdown_value,selectData):
  value_header = ['Description','Active constellations','Observer']
  col_d = []
  col_a = []
  col_o = []

  if selectData != None and selectData['points'] !=[]:
    for sd in range(int(len(selectData['points']))):
      col_d.append(ggm_table[ggm_table['short_name_4ch']==selectData['points'][sd]['text']]['description_short'])
      col_a.append(ggm_table[ggm_table['short_name_4ch']==selectData['points'][sd]['text']]['active_constellations'])
      col_o.append(ggm_table[ggm_table['short_name_4ch']==selectData['points'][sd]['text']]['observer'])
      print(sd)
    trace_table = go.Table(
    header={"values": value_header, "height": 35,
              "line": {"width": 2, "color": "#7FDBFF"}, "fill":{'color':'#111111'},"font": {"size": 15}},
    cells={"values": [col_d,col_a,col_o],"fill":{'color':'#111111'},  "line": {"color": "#7FDBFF"}})
    figure_table = {'data': [trace_table], 'layout':go.Layout(plot_bgcolor='#111111',paper_bgcolor  ='#111111',font={'color':'#7FDBFF'},
              margin= {
                'l': 30,
                'r': 10,
                'b': 0,
                't': 25
              })}
    return figure_table

  if selected_dropdown_value != []:
    for rec in selected_dropdown_value:
      col_d.append(ggm_table[ggm_table['short_name_4ch']==rec]['description_short'])
      col_a.append(ggm_table[ggm_table['short_name_4ch']==rec]['active_constellations'])
      col_o.append(ggm_table[ggm_table['short_name_4ch']==rec]['observer'])
    trace_table = go.Table(
      header={"values": value_header,"height": 35,
                "line": {"width": 2, "color": "#7FDBFF"},"fill":{'color':'#111111'}, "font": {"size": 15}},
      cells={"values": [col_d,col_a,col_o],"fill":{'color':'#111111'}, "line": {"color": "#7FDBFF"}})
    figure_table = {'data': [trace_table], 'layout':go.Layout( plot_bgcolor='#111111',paper_bgcolor  ='#111111',font={'color':'#7FDBFF'},
              margin= {
                'l': 30,
                'r': 10,
                'b': 0,
                't': 25
              })}
    return figure_table
  else:
    trace_table = go.Table(
      header={"values": value_header,"height": 35,
                "line": {"width": 2, "color": "#7FDBFF"},"fill":{'color':'#111111'}, "font": {"size": 15}})
    figure_table = {'data': [trace_table],'layout':go.Layout(title='Please select a receiver', plot_bgcolor='#111111',paper_bgcolor  ='#111111',font={'color':'#7FDBFF'},
              margin= {
                'l': 30,
                'r': 10,
                'b': 0,
                't': 25
              })}
    return figure_table


if __name__ == '__main__':
    app.run_server(debug=True)

