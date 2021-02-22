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
import math
import pysftp

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


  

def knn_stat_tropo(df,time_param):
  knn=pd.DataFrame()
  name=[x for x in globals() if globals()[x] is df][0]
  print('dataframe: {}'.format(name))
  param=name.split('_')[-1]
  knn['date']=df[time_param]
  knn['data_val']=df[("%s"%param+'_'+file['marker_name'][0])]
  print(knn)
  knn=knn.dropna()
  x_knn = knn['data_val'].values.reshape(-1,1)
  # Train kNN detector
  clf = KNN(contamination=0.01, n_neighbors=21, method='median')
  if len(x_knn) <= clf.n_neighbors:
    clf.n_neighbors=math.floor(len(x_knn)/2)
    clf.fit(x_knn)
  else:
    clf.fit(x_knn)
  #predict raw anomaly score
  #scores_pred = clf.decision_function(x_knn)*-1
  #rediction of a datapoint category outlier or inlier
  start=time.time()
  an=clf.predict(x_knn) # to be optimized
  end=time.time()
  #knn['anomaly'] = pd.Series(clf.predict(x_knn))
  print('predict comp time {}'.format(end-start))
  knn['anomaly'] = pd.Series(an)
  #fig, ax = plt.subplots(figsize=(10,6))
  a = knn.loc[knn['anomaly'] == 1, ['date', 'data_val']] #anomaly
  # ax.scatter(knn['date'], knn['data_val'], color='blue', label = 'Normal')
  # ax.scatter(a['date'],a['data_val'], color='red', label = 'Anomaly')
  # plt.legend()
  # plt.title('KNN tropo {} {} {}'.format("%s"%param,"%s"%file['marker_name'][0], 21))
  # plt.xlabel('Date')
  # plt.show()
  # fig.savefig('KNN_tropo_{}_{}_{}.png'.format("%s"%param,"%s"%file['marker_name'][0], 21))
  # y_train_scores = clf.decision_scores_
  return(a)


# Define dataframe 
GGM = pd.DataFrame(columns=['id_result','short_name_4ch','active_constellations','a_priori_x','a_priori_y','sol_id'])
Position_Set = pd.DataFrame(columns=['id_pos_table','id_result','sol_id'])
GNSS_Position = pd.DataFrame(columns=['id_pos','pos_time','pos_x','pos_y','pos_z','id_pos_table','sol_id','flag_pos'])
Tropo_Set = pd.DataFrame(columns=['id_tropo_table','type','id_result','sol_id'])
GNSS_Tropo = pd.DataFrame(columns=['id_tropo','epoch_time','data_val','id_tropo_table','sol_id','flag_tropo'])
Import_Parameters = pd.DataFrame(columns=['sol_id','file_name'])

# GGM list var
id_result = []
short_name_4ch = []
active_constellations = []
a_priori_x = []
a_priori_y = []
sol_id_ggm = []
# Position_Set list var
id_pos_table = []
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
# Tropo_Set
id_tropo_table = []
type_field = []
types = ['ZTD','ZWD','PWV','PRESSURE','TEMPERATURE','HUMIDITY']
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

myHostname = "access834337968.webspace-data.io"
myUsername = "u101458685-tropo-nrt"
myPassword = "sZb.WHeXb2h3j"
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None
#Connect to SFTP page



with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword, port=22, cnopts=cnopts) as sftp:
    print ('Connection succesfully stablished')
    # Switch to a remote directory
    sftp.cwd('/')
    
    # Print data
    for attr in sftp.listdir():
      fold_1=attr
      for attr_2 in sftp.listdir(fold_1+'/'):
        fold_2=attr_2
        for attr_3 in sftp.listdir(fold_1+'/'+fold_2+'/'):
          filename=attr_3
          print(filename)
          sftp.get(fold_1+'/'+fold_2+'/'+filename, "%s"%'./input_real/'+filename)
    sftp.close()


myFile = open('dbConfig_real.txt')
connStr = myFile.readline()
data_conn = connStr.split(" ",2)
dbname = data_conn[0].split("=",1)[1]
username = data_conn[1].split("=",1)[1]
password = data_conn[2].split("=",1)[1]
print(connStr,dbname,username,password)


input_folder = r'input_real/'

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
    if cur is not None and f in check_file:
      error = 'File {} is already imported'.format(f)
      print(error)
    else:
      file=input_folder+f
      file = sio.loadmat(file)
      l = list(file)
      if check_sol != []:
        s=int(check_sol[-1])
        print(check_sol)
      s = s+1
      sol_id.append(s)
      print(s)
      print(sol_id)
      file_name.append("%s"%f)
      initial_advice='Loading file {} '.format("%s"%f)
      print(initial_advice)


      # function to synchornize coord data on timestamp
       # round epoch time values in unix       
      
      name_r=file['marker_name'][0] #this value will be the receiver considered in each loop
      out=pd.DataFrame({'pos_time':file['utc_time'][0],"%s"%'pos_x_'+name_r:file['xyz'][:,0],\
                          "%s"%'pos_y_'+name_r:file['xyz'][:,1],"%s"%'pos_z_'+name_r:file['xyz'][:,2] })
           
      # function to synchornize tropispheric data on timestamp 
      name_r=file['marker_name'][0]  #this value will be the receiver considered in each loop
      out_tropo_ztd=pd.DataFrame({'epoch_time':file['utc_time'][:,0],"%s"%'ztd_'+name_r:file['ztd'][:,0]})
      out_tropo_zwd=pd.DataFrame({'epoch_time':file['utc_time'][:,0],"%s"%'zwd_'+name_r:file['zwd'][:,0]})
      

      # # store anomalies for coordinates 
      # ax = knn_stat(out,'pos_time',med_rec_x)
      # ay = knn_stat(out,'pos_time',med_rec_y)
      # az = knn_stat(out,'pos_time',med_rec_z)
      # a_x=dict.fromkeys(list(ax.date), True)
      # a_y=dict.fromkeys(list(ay.date), True)
      # a_z=dict.fromkeys(list(az.date), True)
      a_x=[]
      a_y=[]
      a_z=[]
        # store anomalies for tropospheric delays 
      a_ztd = knn_stat_tropo(out_tropo_ztd,'epoch_time')
      a_zwd = knn_stat_tropo(out_tropo_zwd,'epoch_time')
      a_ztd_date = dict.fromkeys(list(a_ztd.date), True)
      a_zwd_date = dict.fromkeys(list(a_zwd.date), True)
      
      a_pwv_date=[]
      a_pressure_date=[]
      a_temperature_date=[]
      a_humidity_date=[]

        
        #med_rec_x,med_rec_y,med_rec_z,name_rec = remove_median_coord()
        #med_rec_ztd,med_rec_zwd,name_rec = remove_median_tropo()

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
      
      if (str(file['marker_name'][0])).upper() in short_name:
        advice = 'Receiver {} is already imported '.format((str(file['marker_name'][0])).upper())
        print(advice)
        cur.execute('''
        select id_result from ggm where short_name_4ch=%s''',((str(file['marker_name'][0])).upper(),))
        sel_res_id = cur.fetchone()
        id_result_same_rec=sel_res_id[0]
        print(id_result_same_rec)
      
        # For each position
        n_ps = n_ps+1
        id_pos_table.append(n_ps)
        id_result_ps.append(id_result_same_rec)
        sol_id_positionset.append(s)
        for p in range(len(file['utc_time'][0])):
          if math.isnan(file['utc_time'][p,0]) is not True and math.isnan(file['xyz'][p,0]) is not True and math.isnan(file['xyz'][p,1]) is not True and math.isnan(file['xyz'][p,2]) is not True:
            n_pos = n_pos+1 
            id_pos.append(n_pos)
            pos_time.append(file['utc_time'][p,0])
            pos_x.append(file['xyz'][p,0])
            pos_y.append(file['xyz'][p,1])
            pos_z.append(file['xyz'][p,2])
            id_pos_table_pos.append(n_ps)
            sol_id_pos.append(s)
            # check if in any position the coord has been identified as an outlier
            #if yes set flag equal to one else zero
            if file['utc_time']  in a_x or \
              file['utc_time']  in a_y or \
              file['utc_time']  in a_z:
                flag_pos.append(1)
            else:
                flag_pos.append(0)
              

        for tf in types:
          n_ts = n_ts+1
          id_tropo_table.append(n_ts)
          type_field.append(tf)
          id_result_ts.append(id_result_same_rec)
          sol_id_troposet.append(s)

            # For each measurement
          for d in range(len(file['utc_time'])):
            tmp_time = file['utc_time'][d,0]
            tmp_data = file[tf.lower()][d,0]
            if math.isnan(tmp_time) is not True and math.isnan(tmp_data) is not True:
              n_trp = n_trp+1 
              id_tropo.append(n_trp)
              epoch_time.append(tmp_time)
              data_val.append(tmp_data)
              id_tropo_table_trp.append(n_ts)
              sol_id_trp.append(s)
              if tf == 'ZTD':
                if file['utc_time'][d,0] in a_ztd_date:
                  if file['utc_time'][d,0]  in a_x or \
                    file['utc_time'][d,0]  in a_y or \
                    file['utc_time'] in a_z:
                      flag_tropo.append(2)
                  else:
                      flag_tropo.append(1)
                else:
                    flag_tropo.append(0)
              elif tf == 'ZWD':
                if file['utc_time'] in a_zwd_date:
                  if file['utc_time']  in a_x or \
                    file['utc_time']  in a_y or \
                    file['utc_time']  in a_z:
                      flag_tropo.append(2)
                  else:
                      flag_tropo.append(1)
                else:
                    flag_tropo.append(0)

          

        # Insert created lists in dataframe colummns  

        Tropo_Set['id_tropo_table'] = id_tropo_table
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
        Position_Set['id_result'] = id_result_ps
        Position_Set['sol_id'] = sol_id_positionset

        GNSS_Position['id_pos'] = id_pos
        GNSS_Position['pos_time'] = pos_time
        GNSS_Position['pos_x'] = pos_x
        GNSS_Position['pos_y'] = pos_y
        GNSS_Position['pos_z'] = pos_z
        GNSS_Position['id_pos_table'] = id_pos_table_pos
        GNSS_Position['sol_id'] = sol_id_pos
        GNSS_Position['flag_pos'] = flag_pos

        Import_Parameters['sol_id'] = sol_id
        Import_Parameters['sol_id'] = Import_Parameters['sol_id'].astype(int)

        Import_Parameters['file_name'] = file_name
        # Access to database
        myFile = open('dbConfig_real.txt')
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
            'id_pos_table':sq.INT(),
            'sol_id': sq.INT(),
            'flag_pos': sq.INT()
        })
        Tropo_Set.to_sql('troposet', engine, if_exists='append',index=False,
          dtype={'id_tropo_table':sq.INT(),
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
        short_name_4ch.append((str(file['marker_name'][0])).upper())
        active_constellations.append(str(file['sys'][0]))
        a_priori_x.append(file['lat'][0,0])
        a_priori_y.append(file['lon'][0,0])
        sol_id_ggm.append(s)

        # For each position
        n_ps = n_ps+1
        id_pos_table.append(n_ps)
        id_result_ps.append(n)
        sol_id_positionset.append(s)
        for p in range(len(file['utc_time'][0])):
          if math.isnan(file['utc_time'][p,0]) is not True and math.isnan(file['xyz'][p,0]) is not True and math.isnan(file['xyz'][p,1]) is not True and math.isnan(file['xyz'][p,2]) is not True:
            n_pos = n_pos+1 
            id_pos.append(n_pos)
            pos_time.append(file['utc_time'][p,0])
            pos_x.append(file['xyz'][p,0])
            pos_y.append(file['xyz'][p,1])
            pos_z.append(file['xyz'][p,2])
            id_pos_table_pos.append(n_ps)
            sol_id_pos.append(s)
            # check if in any position the coord has been identified as an outlier
            #if yes set flag equal to one else zero
            if file['utc_time'][p,0]  in a_x or \
              file['utc_time'][p,0]  in a_y or \
              file['utc_time'][p,0]  in a_z:
                flag_pos.append(1)
            else:
              flag_pos.append(0)

        
        for tf in types:
          n_ts = n_ts+1
          id_tropo_table.append(n_ts)
          type_field.append(tf)
          id_result_ts.append(n)
          sol_id_troposet.append(s)
         # For each measurement
          for d in range(len(file['utc_time'])):
            tmp_time = file['utc_time'][d,0]
            tmp_data = file[tf.lower()][d,0]
            if math.isnan(tmp_time) is not True and math.isnan(tmp_data) is not True:
                n_trp = n_trp+1 
                id_tropo.append(n_trp)
                epoch_time.append(tmp_time)
                data_val.append(tmp_data)
                id_tropo_table_trp.append(n_ts)
                sol_id_trp.append(s)
                if tf == 'ZTD':
                  if file['utc_time'][d,0] in a_ztd_date:
                    if file['utc_time'][d,0]  in a_x or \
                      file['utc_time'][d,0]  in a_y or \
                      file['utc_time'][d,0]  in a_z:
                        flag_tropo.append(2)
                    else:
                        flag_tropo.append(1)
                  else:
                      flag_tropo.append(0)
                elif tf == 'ZWD':
                  if file['utc_time'][d,0] in a_zwd_date:
                    if file['utc_time'][d,0]  in a_x or \
                      file['utc_time'][d,0]  in a_y or \
                      file['utc_time'][d,0]  in a_z:
                        flag_tropo.append(2)
                    else:
                        flag_tropo.append(1)
                  else:
                      flag_tropo.append(0)
                elif tf == 'PWV':
                  if file['utc_time'][d,0] in a_pwv_date:
                    if file['utc_time'][d,0]  in a_x or \
                      file['utc_time'][d,0]  in a_y or \
                      file['utc_time'][d,0]  in a_z:
                        flag_tropo.append(2)
                    else:
                        flag_tropo.append(1)
                  else:
                      flag_tropo.append(0)
                elif tf == 'PRESSURE':
                  if file['utc_time'][d,0] in a_pressure_date:
                    if file['utc_time'][d,0]  in a_x or \
                      file['utc_time'][d,0]  in a_y or \
                      file['utc_time'][d,0]  in a_z:
                        flag_tropo.append(2)
                    else:
                        flag_tropo.append(1)
                  else:
                      flag_tropo.append(0)
                elif tf == 'TEMPERATURE':
                  if file['utc_time'][d,0] in a_temperature_date:
                    if file['utc_time'][d,0]  in a_x or \
                      file['utc_time'][d,0]  in a_y or \
                      file['utc_time'][d,0]  in a_z:
                        flag_tropo.append(2)
                    else:
                        flag_tropo.append(1)
                  else:
                      flag_tropo.append(0)
                elif tf == 'HUMIDITY':
                  if file['utc_time'][d,0] in a_humidity_date:
                    if file['utc_time'][d,0]  in a_x or \
                      file['utc_time'][d,0]  in a_y or \
                      file['utc_time'][d,0]  in a_z:
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
        GGM['active_constellations'] = active_constellations
        GGM['a_priori_x'] = a_priori_x
        GGM['a_priori_y'] = a_priori_y
        GGM['sol_id'] = sol_id_ggm  

        Tropo_Set['id_tropo_table'] = id_tropo_table
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
        Position_Set['id_result'] = id_result_ps
        Position_Set['sol_id'] = sol_id_positionset

        GNSS_Position['id_pos'] = id_pos
        GNSS_Position['pos_time'] = pos_time
        GNSS_Position['pos_x'] = pos_x
        GNSS_Position['pos_y'] = pos_y
        GNSS_Position['pos_z'] = pos_z
        GNSS_Position['id_pos_table'] = id_pos_table_pos
        GNSS_Position['sol_id'] = sol_id_pos
        GNSS_Position['flag_pos'] = flag_pos

        Import_Parameters['sol_id'] = sol_id
        Import_Parameters['sol_id'] = Import_Parameters['sol_id'].astype(int)
        Import_Parameters['file_name'] = file_name
        # Access to database
        myFile = open('dbConfig_real.txt')
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
            #'active_constellations': sq.CHAR(),
            'a_priori_x': sq.FLOAT(),
            'a_priori_y': sq.FLOAT(),
            #'a_priori_z': sq.FLOAT(),
             'sol_id': sq.INT()
        })
        Position_Set.to_sql('positionset', engine, if_exists='append',index=False,
          dtype={'id_pos_table': sq.INT(),
            'id_result':sq.INT(),
            'sol_id': sq.INT()
        })
        GNSS_Position.to_sql('gnssposition', engine ,if_exists='append',index=False,
          dtype={'id_pos': sq.INT(),
            'pos_time': sq.INT(),
            'pos_x': sq.FLOAT(),
            'pos_y': sq.FLOAT(),
            'pos_z': sq.FLOAT(),
            'id_pos_table':sq.INT(),
            'sol_id': sq.INT(),
            'flag_pos' : sq.INT()
        })
        Tropo_Set.to_sql('troposet', engine, if_exists='append',index=False,
          dtype={'id_tropo_table':sq.INT(),
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
      GGM = pd.DataFrame(columns=['id_result','short_name_4ch','active_constellations','a_priori_x','a_priori_y','sol_id'])
      Position_Set = pd.DataFrame(columns=['id_pos_table','id_result','sol_id'])
      GNSS_Position = pd.DataFrame(columns=['id_pos','pos_time','pos_x','pos_y','pos_z','id_pos_table','sol_id','flag_pos'])
      Tropo_Set = pd.DataFrame(columns=['id_tropo_table','type','id_result','sol_id'])
      GNSS_Tropo = pd.DataFrame(columns=['id_tropo','epoch_time','data_val','id_tropo_table','sol_id','flag_tropo'])
      Import_Parameters = pd.DataFrame(columns=['sol_id','file_name'])


      # GGM list var
      id_result = []
      short_name_4ch = []
      active_constellations = []
      a_priori_x = []
      a_priori_y = []
      sol_id_ggm = []
      # Position_Set list var
      id_pos_table = []
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
      # Tropo_Set
      id_tropo_table = []
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


