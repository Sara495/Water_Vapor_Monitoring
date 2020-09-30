import pandas as pd
from psycopg2 import connect
from sqlalchemy import create_engine
import scipy.io as sio
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
import psycopg2
import math
import os
import sys
import sqlalchemy as sq
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.utils import evaluate_print
import time

psycopg2.extensions.register_adapter(np.ndarray, psycopg2._psycopg.AsIs)

# remove trend for each receiver to prepare dataset for outlier remover
def remove_median_coord():
  # create an array with lenght equal to the # of receivers to store the values of the 
  # coordinates in the three directions, and a list for the rec name
  for i in range(len(rec_array[0,:])):
  # create a zero array of arrays with same length of epoch values, and
  # dimension equal to the # of receivers
    med_rec_x=np.zeros((len(out['pos_time']),len(rec_array[0])))
    med_rec_y=np.zeros((len(out['pos_time']),len(rec_array[0])))
    med_rec_z=np.zeros((len(out['pos_time']),len(rec_array[0])))
    name_rec=[]
  # for each receiver remove the median by column    
  for i in range(len(rec_array[0,:])):
    name=rec_array[0,i]['short_name_4ch'][0]
    med_rec_x[:,i]=out["%s"%'pos_x_'+name]- np.nanmedian(out["%s"%'pos_x_'+name])    
    med_rec_y[:,i]=out["%s"%'pos_y_'+name]- np.nanmedian(out["%s"%'pos_y_'+name])
    med_rec_z[:,i]=out["%s"%'pos_z_'+name]- np.nanmedian(out["%s"%'pos_z_'+name])  
    # store the name of the receivers at issue in an array to be used in knn
    name_rec.append(name)
  # compute the median by row and remove it by each column
  med_x= np.nanmedian(med_rec_x,axis=1)
  med_y= np.nanmedian(med_rec_y,axis=1)
  med_z= np.nanmedian(med_rec_z,axis=1)
  for j in range(len(med_rec_x[0])):
    med_rec_x[:,j]=med_rec_x[:,j]-med_x
    med_rec_y[:,j]=med_rec_y[:,j]-med_y
    med_rec_z[:,j]=med_rec_z[:,j]-med_z
  return(med_rec_x,med_rec_y,med_rec_z,name_rec)
  
# remove trend for each receiver to prepare dataset for outlier remover
def remove_median_tropo():
  # create an array with lenght equal to the # of receivers to store the values of the 
  # tropospheric delays ztd, zwd, and a list for the rec name
  for i in range(len(rec_array[0,:])):
  # create a zero array of arrays with same length of epoch values, and
  # dimension equal to the # of receivers
    med_rec_ztd=np.zeros((len(out_tropo['epoch_time']),len(rec_array[0])))
    med_rec_zwd=np.zeros((len(out_tropo['epoch_time']),len(rec_array[0])))
    name_rec=[]
  # for each receiver remove the median by column    
  for i in range(len(rec_array[0,:])):
    name=rec_array[0,i]['short_name_4ch'][0]
    med_rec_ztd[:,i]=out_tropo["%s"%'ztd_'+name]- np.nanmedian(out_tropo["%s"%'ztd_'+name])    
    med_rec_zwd[:,i]=out_tropo["%s"%'zwd_'+name]- np.nanmedian(out_tropo["%s"%'zwd_'+name])
    # store the name of the receivers at issue in an array to be used in knn
    name_rec.append(name)
  # compute the median by row and remove it by each column
  med_ztd= np.nanmedian(med_rec_ztd,axis=1)
  med_zwd= np.nanmedian(med_rec_zwd,axis=1)
  for j in range(len(med_rec_ztd[0])):
    med_rec_ztd[:,j]=med_rec_ztd[:,j]-med_ztd
    med_rec_zwd[:,j]=med_rec_zwd[:,j]-med_zwd
  return(med_rec_ztd,med_rec_zwd,name_rec)


  
# use K Nearest Neighbors stat to detect outliers
def knn_stat(df,time_param,med_rec):
  knn=pd.DataFrame()
  name=[x for x in globals() if globals()[x] is df][0]
  print('dataframe: {}'.format(name))
  knn['date']=df[time_param]
  knn['data_val']=med_rec[:,name_rec.index(r['short_name_4ch'][0])]  
  knn=knn.dropna()
  x_knn = knn['data_val'].values.reshape(-1,1)
  # Train kNN detector
  if name =='out_tropo':
      clf = KNN(contamination=0.01, n_neighbors=20, method='median')
  else:
      clf = KNN(contamination=0.1, n_neighbors=20, method='median')
  clf.fit(x_knn)
  # predict raw anomaly score
  #scores_pred = clf.decision_function(x_knn)*-1
  # prediction of a datapoint category outlier or inlier
  start=time.time()
  an=clf.predict(x_knn) # to be optimized
  end=time.time()
  #knn['anomaly'] = pd.Series(clf.predict(x_knn))
  print('predict comp time {}'.format(end-start))
  knn['anomaly'] = pd.Series(an)
  #fig, ax = plt.subplots(figsize=(10,6))
  a = knn.loc[knn['anomaly'] == 1, ['date', 'data_val']] #anomaly
#  ax.plot(knn['date'], knn['data_val'], color='blue', label = 'Normal')
#  ax.scatter(a['date'],a['data_val'], color='red', label = 'Anomaly')
#  plt.legend()
#  plt.title('KNN')
#  plt.xlabel('Date')
#  plt.show();
#  y_train_scores = clf.decision_scores_
#  evaluate_print('KNN', clf.predict(x_knn),y_train_scores)
  return(a)

# Define dataframe 
GGM = pd.DataFrame(columns=['id_result','short_name_4ch','description_short','active_constellations','a_priori_x','a_priori_y','a_priori_z','observer','agency','ant_type','sol_id'])
Position_Set = pd.DataFrame(columns=['id_pos_table','rate','description','id_result','sol_id'])
GNSS_Position = pd.DataFrame(columns=['id_pos','pos_time','pos_x','pos_y','pos_z','var_x','var_y','var_z','id_pos_table','sol_id','flag_pos'])
Tropo_Set = pd.DataFrame(columns=['id_tropo_table','rate','type','description','id_result','sol_id'])
GNSS_Tropo = pd.DataFrame(columns=['id_tropo','epoch_time','data_val','id_tropo_table','sol_id','flag_tropo'])
Import_Parameters = pd.DataFrame(columns=['sol_id','file_name'])

# GGM list var
id_result = []
short_name_4ch = []
description_short = []
active_constellations = []
a_priori_x = []
a_priori_y = []
a_priori_z = []
observer = []
agency = []
ant_type = []
sol_id_ggm = []
# Position_Set list var
id_pos_table = []
rate_ps = []
id_result_ps = []
sol_id_positionset = [] 
# GNSS_Position list var
id_pos = []
pos_time = []
pos_x = []
pos_y = []
pos_z = []
id_pos_table_pos = []
sol_id_pos = []
flag_pos = []
#var_x = [] var_y = [] var_z = []
# Tropo_Set
id_tropo_table = []
rate =[]
type_field = []
types = ['ZTD','ZWD','PWV','PRESSURE','TEMPERATURE','HUMIDITY','N_SAT']
id_result_ts = []
sol_id_troposet =[]
# GNSS_Tropo list var
id_tropo = []
epoch_time = []
data_val = []
id_tropo_table_trp = []
sol_id_trp = []
flag_tropo = []
# Import_Parameters 
sol_id = []
file_name = []

# ID Counters
s = 0
n = 0 
n_ts = 0
n_trp = 0
n_ps = 0
n_pos = 0

myFile = open('dbConfig.txt')
connStr = myFile.readline()
data_conn = connStr.split(" ",2)
dbname = data_conn[0].split("=",1)[1]
username = data_conn[1].split("=",1)[1]
password = data_conn[2].split("=",1)[1]
print(connStr,dbname,username,password)


input_folder = r'/Users/saramaffioli/Desktop/dash/input/'

for f in os.listdir(input_folder):
  if f != '.DS_Store':
    # Check if the file has been already imported
    conn = connect(connStr)
    cur = conn.cursor()
    cur.execute(
   'SELECT sol_id,file_name FROM importparam')
    check_sol =[]
    check_file=[]
    for row in cur:
      check_sol.append(row[0])
      check_file.append(row[1])

    #print(check_sol,check_file)

    if cur is not None and f in check_file:
      error = 'File {} is already imported'.format(f)
      print(error)
    else:
      file=input_folder+f
      file = sio.loadmat(file)
      l = list(file)
      for v in l:
        if v !='__header__' and v != '__version__' and v != '__globals__':
          name_c = v
            
      rec_array = file[name_c]
      if check_sol != []:
        s=int(check_sol[-1])
      s = s+1
      sol_id.append(s)
      file_name.append("%s"%f)
      initial_advice='Loading file {} '.format("%s"%f)
      print(initial_advice)

      # For each receiver len(rec_array[0,:])
      for i in range(len(rec_array[0,:])):
        r=rec_array[0,i]
        # function to synchornize coord data on timestamp
        # round epoch time values in unix       
        for it in range(len(r['pos_time'])):
          r['pos_time'][it]= round(r['pos_time'][it][0]/30)*30
        name_r=r['short_name_4ch'][0] #this value will be the receiver considered in each loop
        out=pd.DataFrame({'pos_time':r['pos_time'][:,0],"%s"%'pos_x_'+name_r:r['pos_xyz'][:,0],\
                          "%s"%'pos_y_'+name_r:r['pos_xyz'][:,1],"%s"%'pos_z_'+name_r:r['pos_xyz'][:,2] })
        for i in range(len(rec_array[0])):
          if rec_array[0,i]['short_name_4ch'][0] != name_r:
            name=rec_array[0,i]['short_name_4ch'][0]
            # round epoch time values in unix       
            for it in range(len(rec_array[0,i]['pos_time'])):
              rec_array[0,i]['pos_time'][it]= round(rec_array[0,i]['pos_time'][it][0]/30)*30
            out=out.merge(pd.DataFrame({'pos_time':rec_array[0,i]['pos_time'][:,0],\
                                            "%s"%'pos_x_'+name:rec_array[0,i]['pos_xyz'][:,0],\
                                            "%s"%'pos_y_'+name:rec_array[0,i]['pos_xyz'][:,1],\
                                            "%s"%'pos_z_'+name:rec_array[0,i]['pos_xyz'][:,2] }),\
                                            on='pos_time', how='outer')
        # function to synchornize tropispheric data on timestamp 
        for it in range(len(r['epoch_time'])):
          r['epoch_time'][it]= round(r['epoch_time'][it][0]/30)*30    
        name_r=r['short_name_4ch'][0]  #this value will be the receiver considered in each loop
        out_tropo=pd.DataFrame({'epoch_time':r['epoch_time'][:,0],"%s"%'ztd_'+name_r:r['ztd'][:,0],\
                                "%s"%'zwd_'+name_r:r['zwd'][:,0]})
        for i in range(len(rec_array[0])):
          if rec_array[0,i]['short_name_4ch'][0] != name_r:
            name=rec_array[0,i]['short_name_4ch'][0]
            for it in range(len(rec_array[0,i]['epoch_time'])):
              rec_array[0,i]['epoch_time'][it]= round(rec_array[0,i]['epoch_time'][it][0]/30)*30   
            out_tropo=out_tropo.merge(pd.DataFrame({'epoch_time':rec_array[0,i]['epoch_time'][:,0],\
                                            "%s"%'ztd_'+name:rec_array[0,i]['ztd'][:,0],\
                                            "%s"%'zwd_'+name:rec_array[0,i]['zwd'][:,0]}),\
                                            on='epoch_time', how='outer')  
        med_rec_x,med_rec_y,med_rec_z,name_rec = remove_median_coord()
        # store anomalies for coordinates 
        ax = knn_stat(out,'pos_time',med_rec_x)
        ay = knn_stat(out,'pos_time',med_rec_y)
        az = knn_stat(out,'pos_time',med_rec_z)
        a_x=dict.fromkeys(list(ax.date), True)
        a_y=dict.fromkeys(list(ay.date), True)
        a_z=dict.fromkeys(list(az.date), True)
        med_rec_ztd,med_rec_zwd,name_rec = remove_median_tropo()
        # store anomalies for tropospheric delays 
        a_ztd = knn_stat(out_tropo,'epoch_time',med_rec_ztd)
        a_zwd = knn_stat(out_tropo,'epoch_time',med_rec_zwd)
        a_ztd_date = dict.fromkeys(list(a_ztd.date), True)
        a_zwd_date = dict.fromkeys(list(a_zwd.date), True)
        
        conn = connect(connStr)
        cur = conn.cursor()
        # Check if the receiver is already present
        cur.execute(
        'SELECT short_name_4ch FROM ggm')
        short_name = []
        for row in cur:
          short_name.append(row[0])
        print(short_name)
        

        # If the receiver is present check if there are duplicated values
        
        if (str(r['short_name_4ch'][0])).upper() in short_name:
          advice = 'Receiver {} is already imported '.format((str(r['short_name_4ch'][0])).upper())
          print(advice)
          cur.execute('''
          select id_result from ggm where short_name_4ch=%s''',((str(r['short_name_4ch'][0])).upper(),))
          sel_res_id = cur.fetchone()
          id_result_same_rec=sel_res_id[0]
          print(id_result_same_rec)
          # For each combination rate - type
          for rt in range(len(r['rate'])):
            # For each position
            n_ps = n_ps+1
            id_pos_table.append(n_ps)
            rate_ps.append(r['rate'][rt,0])
            id_result_ps.append(id_result_same_rec)
            sol_id_positionset.append(s)
            for p in range(len(r['pos_time'])):
              if math.isnan(r['pos_time'][p,0]) is not True and math.isnan(r['pos_xyz'][p,0]) is not True and math.isnan(r['pos_xyz'][p,1]) is not True and math.isnan(r['pos_xyz'][p,2]) is not True:
                n_pos = n_pos+1 
                id_pos.append(n_pos)
                pos_time.append(r['pos_time'][p,0])
                pos_x.append(r['pos_xyz'][p,0])
                pos_y.append(r['pos_xyz'][p,1])
                pos_z.append(r['pos_xyz'][p,2])
                id_pos_table_pos.append(n_ps)
                sol_id_pos.append(s)
                # check if in any position the coord has been identified as an outlier
                #if yes set flag equal to one else zero
                if r['pos_time'][p,0]  in a_x or \
                  r['pos_time'][p,0]  in a_y or \
                  r['pos_time'][p,0]  in a_z:
                    flag_pos.append(1)
                else:
                    flag_pos.append(0)
                

                #var_x.append(r['var_xyz'][p,0])
                #var_y.append(r['var_xyz'][p,0])
                #var_z.append(r['var_xyz'][p,0])
           
            for tf in types:
              n_ts = n_ts+1
              id_tropo_table.append(n_ts)
              rate.append(r['rate'][rt,0])
              type_field.append(tf)
              id_result_ts.append(id_result_same_rec)
              sol_id_troposet.append(s)

              # For each measurement
              for d in range(len(r['epoch_time'])):
                tmp_time = r['epoch_time'][d,0]
                tmp_data = r[tf.lower()][d,0]
                if math.isnan(tmp_time) is not True and math.isnan(tmp_data) is not True:
                  n_trp = n_trp+1 
                  id_tropo.append(n_trp)
                  epoch_time.append(tmp_time)
                  data_val.append(tmp_data)
                  id_tropo_table_trp.append(n_ts)
                  sol_id_trp.append(s)
                  if tf == 'ZTD':
                    if r['epoch_time'][d,0] in a_ztd_date:
                      if r['epoch_time'][d,0]  in a_x or \
                        r['epoch_time'][d,0]  in a_y or \
                        r['epoch_time'][d,0]  in a_z:
                          flag_tropo.append(2)
                      else:
                          flag_tropo.append(1)
                    else:
                        flag_tropo.append(0)
                  elif tf == 'ZWD':
                    if r['epoch_time'][d,0] in a_zwd_date:
                      if r['epoch_time'][d,0]  in a_x or \
                        r['epoch_time'][d,0]  in a_y or \
                        r['epoch_time'][d,0]  in a_z:
                          flag_tropo.append(2)
                      else:
                          flag_tropo.append(1)
                    else:
                        flag_tropo.append(0)

            

        # Insert created lists in dataframe colummns  

          Tropo_Set['id_tropo_table'] = id_tropo_table
          Tropo_Set['rate'] = rate
          Tropo_Set['type'] = type_field
          Tropo_Set['id_result'] = id_result_ts
          Tropo_Set['sol_id'] = sol_id_troposet

          GNSS_Tropo['id_tropo'] = id_tropo
          GNSS_Tropo['epoch_time'] = epoch_time
          GNSS_Tropo['data_val'] = data_val
          GNSS_Tropo['id_tropo_table'] = id_tropo_table_trp
          GNSS_Tropo['sol_id'] = sol_id_trp
          GNSS_Tropo['flag_tropo'] = flag_tropo

          Position_Set['id_pos_table'] = id_pos_table
          Position_Set['rate'] = rate_ps
          Position_Set['id_result'] = id_result_ps
          Position_Set['sol_id'] = sol_id_positionset

          GNSS_Position['id_pos'] = id_pos
          GNSS_Position['pos_time'] = pos_time
          GNSS_Position['pos_x'] = pos_x
          GNSS_Position['pos_y'] = pos_y
          GNSS_Position['pos_z'] = pos_z
          GNSS_Position['id_pos_table'] = id_pos_table_pos
          GNSS_Position['sol_id'] = sol_id_pos
          #GNSS_Position['var_x'] = var_x
          #GNSS_Position['var_y'] = var_y
          #GNSS_Position['var_z'] = var_z
          GNSS_Position['flag_pos'] = flag_pos

          Import_Parameters['sol_id'] = sol_id
          Import_Parameters['sol_id'] = Import_Parameters['sol_id'].astype(int)

          Import_Parameters['file_name'] = file_name
          # Access to database
          myFile = open('dbConfig.txt')
          connStr = myFile.readline()
          data_conn = connStr.split(" ",2)
          dbname = data_conn[0].split("=",1)[1]
          username = data_conn[1].split("=",1)[1]
          password = data_conn[2].split("=",1)[1]
          print(connStr,dbname,username,password)
          conn = connect(connStr)
          cur = conn.cursor()

          # Create engine for import dataframe
          db_url = 'postgresql://'+username+':'+password+'@localhost:5432/'+dbname
          engine = create_engine(db_url)
          # Import dataframe
      
          Position_Set.to_sql('positionset', engine, if_exists='append',index=False,
            dtype={'id_pos_table': sq.INT(),
             'rate': sq.INT(),
             #'description': sq.CHAR(),
             'id_result':sq.INT(),
             'sol_id': sq.INT()
          })
          GNSS_Position.to_sql('gnssposition', engine ,if_exists='append',index=False,
            dtype={'id_pos': sq.INT(),
              'pos_time': sq.INT(),
              'pos_x': sq.FLOAT(),
              'pos_y': sq.FLOAT(),
              'pos_z': sq.FLOAT(),
              'var_x': sq.FLOAT(),
              'var_y': sq.FLOAT(),
              'var_z': sq.FLOAT(),
              'id_pos_table':sq.INT(),
              'sol_id': sq.INT(),
              'flag_pos': sq.INT()
          })
          Tropo_Set.to_sql('troposet', engine, if_exists='append',index=False,
            dtype={'id_tropo_table':sq.INT(),
              'rate': sq.INT(),
              #'type': sq.CHAR(),
              #'description': sq.CHAR(),
              'id_result': sq.INT(),      
              'sol_id': sq.INT()
          })
          GNSS_Tropo.to_sql('gnsstropo', engine, if_exists='append',index=False,
            dtype={'id_tropo': sq.INT(),
              'epoch_time': sq.INT(),
              'data_val': sq.FLOAT(),
              'id_tropo_table':sq.INT(),
              'sol_id': sq.INT(),
              'flag_tropo' : sq.INT()
          })
          Import_Parameters.to_sql('importparam', engine,if_exists='append',index=False,  
            dtype={'sol_id': sq.INT()
              #'file_name': sq.CHAR()
              })

          final_advice='File {} added to database'.format("%s"%f)
          print(final_advice)
          # Close connection 
          cur.close()
          conn.commit()
          conn.close()

        # Else add new receiver  
        else:   
          n = n+1
          id_result.append(n)
          short_name_4ch.append((str(r['short_name_4ch'][0])).upper())
          description_short.append(str(r['description_short'][0]))
          active_constellations.append(str(r['active_constellations'][0]))
          a_priori_x.append(r['a_priori_xyz'][0,0])
          a_priori_y.append(r['a_priori_xyz'][0,1])
          a_priori_z.append(r['a_priori_xyz'][0,2])
          sol_id_ggm.append(s)
          # Check empy values and replace with NaN
          if r['observer'].shape == (0,):
          	observer_v='NaN'
          	observer.append(observer_v)
          else: observer.append(str(r['observer'][0]))
          if r['agency'].shape == (0,):
          	agency_v = 'NaN'
          	agency.append(agency_v)
          else: agency.append(str(r['agency'][0]))
          if r['ant_type'].shape == (0,):
          	ant_type_v = 'NONE'
          	ant_type.append(ant_type_v)
          else: ant_type.append(str(r['ant_type'][0]))
          # For each combination rate - type
          for rt in range(len(r['rate'])):
            # For each position
            n_ps = n_ps+1
            id_pos_table.append(n_ps)
            rate_ps.append(r['rate'][rt,0])
            id_result_ps.append(n)
            sol_id_positionset.append(s)
            for p in range(len(r['pos_time'])):
              if math.isnan(r['pos_time'][p,0]) is not True and math.isnan(r['pos_xyz'][p,0]) is not True and math.isnan(r['pos_xyz'][p,1]) is not True and math.isnan(r['pos_xyz'][p,2]) is not True:
                n_pos = n_pos+1 
                id_pos.append(n_pos)
                pos_time.append(r['pos_time'][p,0])
                pos_x.append(r['pos_xyz'][p,0])
                pos_y.append(r['pos_xyz'][p,1])
                pos_z.append(r['pos_xyz'][p,2])
                id_pos_table_pos.append(n_ps)
                sol_id_pos.append(s)
                # check if in any position the coord has been identified as an outlier
                #if yes set flag equal to one else zero
                if r['pos_time'][p,0]  in a_x or \
                  r['pos_time'][p,0]  in a_y or \
                  r['pos_time'][p,0]  in a_z:
                    flag_pos.append(1)
                else:
                    flag_pos.append(0)

                #var_x.append(r['var_xyz'][p,0])
                #var_y.append(r['var_xyz'][p,0])
                #var_z.append(r['var_xyz'][p,0])
        
            for tf in types:
              n_ts = n_ts+1
              id_tropo_table.append(n_ts)
              rate.append(r['rate'][rt,0])
              type_field.append(tf)
              id_result_ts.append(n)
              sol_id_troposet.append(s)
              # For each measurement
              for d in range(len(r['epoch_time'])):
                tmp_time = r['epoch_time'][d,0]
                tmp_data = r[tf.lower()][d,0]
                if math.isnan(tmp_time) is not True and math.isnan(tmp_data) is not True:
                  n_trp = n_trp+1 
                  id_tropo.append(n_trp)
                  epoch_time.append(tmp_time)
                  data_val.append(tmp_data)
                  id_tropo_table_trp.append(n_ts)
                  sol_id_trp.append(s)
                  if tf == 'ZTD':
                    if r['epoch_time'][d,0] in a_ztd_date:
                      if r['epoch_time'][d,0]  in a_x or \
                        r['epoch_time'][d,0]  in a_y or \
                        r['epoch_time'][d,0]  in a_z:
                          flag_tropo.append(2)
                      else:
                          flag_tropo.append(1)
                    else:
                        flag_tropo.append(0)
                  elif tf == 'ZWD':
                    if r['epoch_time'][d,0] in a_zwd_date:
                      if r['epoch_time'][d,0]  in a_x or \
                        r['epoch_time'][d,0]  in a_y or \
                        r['epoch_time'][d,0]  in a_z:
                          flag_tropo.append(2)
                      else:
                          flag_tropo.append(1)
                    else:
                        flag_tropo.append(0)


          cur.close()
          conn.close()

      # Insert created lists in dataframe colummns

          GGM['id_result'] = id_result
          GGM['short_name_4ch'] = short_name_4ch
          GGM['description_short'] = description_short
          GGM['active_constellations'] = active_constellations
          GGM['a_priori_x'] = a_priori_x
          GGM['a_priori_y'] = a_priori_y
          GGM['a_priori_z'] = a_priori_z
          GGM['observer'] = observer
          GGM['agency'] = agency
          GGM['ant_type'] = ant_type 
          GGM['sol_id'] = sol_id_ggm  

          Tropo_Set['id_tropo_table'] = id_tropo_table
          Tropo_Set['rate'] = rate
          Tropo_Set['type'] = type_field
          Tropo_Set['id_result'] = id_result_ts
          Tropo_Set['sol_id'] = sol_id_troposet

          GNSS_Tropo['id_tropo'] = id_tropo
          GNSS_Tropo['epoch_time'] = epoch_time
          GNSS_Tropo['data_val'] = data_val
          GNSS_Tropo['id_tropo_table'] = id_tropo_table_trp
          GNSS_Tropo['sol_id'] = sol_id_trp
          GNSS_Tropo['flag_tropo'] = flag_tropo

          Position_Set['id_pos_table'] = id_pos_table
          Position_Set['rate'] = rate_ps
          Position_Set['id_result'] = id_result_ps
          Position_Set['sol_id'] = sol_id_positionset

          GNSS_Position['id_pos'] = id_pos
          GNSS_Position['pos_time'] = pos_time
          GNSS_Position['pos_x'] = pos_x
          GNSS_Position['pos_y'] = pos_y
          GNSS_Position['pos_z'] = pos_z
          GNSS_Position['id_pos_table'] = id_pos_table_pos
          GNSS_Position['sol_id'] = sol_id_pos
          #GNSS_Position['var_x'] = var_x
          #GNSS_Position['var_y'] = var_y
          #GNSS_Position['var_z'] = var_z
          GNSS_Position['flag_pos'] = flag_pos

          Import_Parameters['sol_id'] = sol_id
          Import_Parameters['sol_id'] = Import_Parameters['sol_id'].astype(int)
          Import_Parameters['file_name'] = file_name
          # Access to database
          myFile = open('dbConfig.txt')
          connStr = myFile.readline()
          data_conn = connStr.split(" ",2)
          dbname = data_conn[0].split("=",1)[1]
          username = data_conn[1].split("=",1)[1]
          password = data_conn[2].split("=",1)[1]
          print(connStr,dbname,username,password)
          conn = connect(connStr)
          cur = conn.cursor()

          # Create engine for import dataframe
          db_url = 'postgresql://'+username+':'+password+'@localhost:5432/'+dbname
          engine = create_engine(db_url)
          # Import dataframe
          GGM.to_sql('ggm', engine, if_exists='append', index=False,
            dtype={'id_result': sq.INT(),
              #'short_name_4ch':sq.CHAR(),
              #'description_short':sq.CHAR(),
              #'active_constellations': sq.CHAR(),
              'a_priori_x': sq.FLOAT(),
              'a_priori_y': sq.FLOAT(),
              'a_priori_z': sq.FLOAT(),
              #'observer':sq.CHAR(),
              #'agency':sq.CHAR(),
              #'ant_type':sq.CHAR(),
              'sol_id': sq.INT()
          })
          Position_Set.to_sql('positionset', engine, if_exists='append',index=False,
            dtype={'id_pos_table': sq.INT(),
              'rate': sq.INT(),
              #'description': sq.CHAR(),
              'id_result':sq.INT(),
              'sol_id': sq.INT()
          })
          GNSS_Position.to_sql('gnssposition', engine ,if_exists='append',index=False,
            dtype={'id_pos': sq.INT(),
              'pos_time': sq.INT(),
              'pos_x': sq.FLOAT(),
              'pos_y': sq.FLOAT(),
              'pos_z': sq.FLOAT(),
              'var_x': sq.FLOAT(),
              'var_y': sq.FLOAT(),
              'var_z': sq.FLOAT(),
              'id_pos_table':sq.INT(),
              'sol_id': sq.INT(),
              'flag_pos' : sq.INT()
          })
          Tropo_Set.to_sql('troposet', engine, if_exists='append',index=False,
            dtype={'id_tropo_table':sq.INT(),
              'rate': sq.INT(),
              #'type': sq.CHAR(),
              #'description': sq.CHAR(),
              'id_result': sq.INT(),      
              'sol_id': sq.INT()
          })
          GNSS_Tropo.to_sql('gnsstropo', engine, if_exists='append',index=False,
            dtype={'id_tropo': sq.INT(),
              'epoch_time': sq.INT(),
              'data_val': sq.FLOAT(),
              'id_tropo_table':sq.INT(),
              'sol_id': sq.INT(),
              'flag_tropo': sq.INT()
          })
          Import_Parameters.to_sql('importparam', engine,if_exists='append',index=False,
            dtype={'sol_id': sq.INT()
              #'file_name': sq.CHAR()
              })

          final_advice='File {} added to database'.format("%s"%f)
          print(final_advice)
          # Close connection 
          cur.close()
          conn.commit()
          conn.close()
        

      # Reset values
        GGM = pd.DataFrame(columns=['id_result','short_name_4ch','description_short','active_constellations','a_priori_x','a_priori_y','a_priori_z','observer','agency','ant_type','sol_id'])
        Position_Set = pd.DataFrame(columns=['id_pos_table','rate','description','id_result','sol_id'])
        GNSS_Position = pd.DataFrame(columns=['id_pos','pos_time','pos_x','pos_y','pos_z','var_x','var_y','var_z','id_pos_table','sol_id','flag_pos'])
        Tropo_Set = pd.DataFrame(columns=['id_tropo_table','rate','type','description','id_result','sol_id'])
        GNSS_Tropo = pd.DataFrame(columns=['id_tropo','epoch_time','data_val','id_tropo_table','sol_id','flag_tropo'])
        Import_Parameters = pd.DataFrame(columns=['sol_id','file_name'])
        # GGM list var
        id_result = []
        short_name_4ch = []
        description_short = []
        active_constellations = []
        a_priori_x = []
        a_priori_y = []
        a_priori_z = []
        observer = []
        agency = []
        ant_type = []
        sol_id_ggm = []
        # Position_Set list var
        id_pos_table = []
        rate_ps = []
        id_result_ps = [] 
        sol_id_positionset = [] 
        # GNSS_Position list var
        id_pos = []
        pos_time = []
        pos_x = []
        pos_y = []
        pos_z = []
        id_pos_table_pos = []
        sol_id_pos = []
        flag_pos = []
        #var_x = [] var_y = [] var_z = []
        # Tropo_Set
        id_tropo_table = []
        rate =[]
        type_field = []
        id_result_ts = []
        sol_id_troposet =[]
        # GNSS_Tropo list var
        id_tropo = []
        epoch_time = []
        data_val = []
        id_tropo_table_trp = []
        sol_id_trp = []
        flag_tropo = []
        # Import_Parameters 
        sol_id = []
        file_name = []






print('exit')


