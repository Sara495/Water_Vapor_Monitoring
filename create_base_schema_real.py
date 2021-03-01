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



# Define dataframe 
GGM = pd.DataFrame(columns=['id_result','short_name_4ch','active_constellations','a_priori_x','a_priori_y','sol_id'])
Position_Set = pd.DataFrame(columns=['id_pos_table','id_result','sol_id'])
GNSS_Position = pd.DataFrame(columns=['id_pos','pos_time','pos_x','pos_y','pos_z','id_pos_table','sol_id','flag_pos'])
Tropo_Set = pd.DataFrame(columns=['id_tropo_table','type','id_result','sol_id'])
GNSS_Tropo = pd.DataFrame(columns=['id_tropo','epoch_time','data_val','id_tropo_table','sol_id','flag_tropo'])
Import_Parameters = pd.DataFrame(columns=['sol_id','file_name'])

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
db_url = "%s"%'postgresql://'+username+':'+password+'@localhost:5432/'+dbname
print(db_url)
engine = create_engine(db_url)
# Import dataframe
GGM.to_sql('ggm', engine, if_exists='replace', index=False, 
	dtype={'id_result': sq.INT(),
	      # 'short_name_4ch':sq.CHAR(),
	       'a_priori_x': sq.FLOAT(),
	       'a_priori_y': sq.FLOAT(),
	       #'a_priori_z': sq.FLOAT(),
	       'sol_id': sq.INT()
	})
Position_Set.to_sql('positionset', engine, if_exists='replace', index=False, 
	dtype={'id_pos_table': sq.INT(),
	       'id_result':sq.INT(),
	       'sol_id': sq.INT()
	})
GNSS_Position.to_sql('gnssposition', engine, if_exists='replace', index=False,
	dtype={'id_pos': sq.INT(),
	       'pos_time': sq.FLOAT(),
	       'pos_x': sq.FLOAT(),
	       'pos_y': sq.FLOAT(),
	       'pos_z': sq.FLOAT(),
	       'id_pos_table':sq.INT(),
	       'sol_id': sq.INT(),
	       'flag_pos': sq.INT()
	})
Tropo_Set.to_sql('troposet', engine, if_exists='replace', index=False, 
	dtype={'id_tropo_table':sq.INT(),
	       #'type': sq.CHAR(),
	       'id_result': sq.INT(),      
	       'sol_id': sq.INT()
	})
GNSS_Tropo.to_sql('gnsstropo', engine, if_exists='replace', index=False,
	dtype={'id_tropo': sq.INT(),
	       'epoch_time': sq.FLOAT(),
	       'data_val': sq.FLOAT(),
	       'id_tropo_table':sq.INT(),
	       'sol_id': sq.INT(),
	       'flag_tropo': sq.INT()
	})
Import_Parameters.to_sql('importparam', engine, if_exists='replace', index=False, 
	dtype={'sol_id': sq.INT()
	        #'file_name': sq.CHAR()
	 })

print('Tables added to GNSS REAL TIME database')
# Close connection 
cur.close()
conn.commit()
conn.close()