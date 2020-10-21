# Water_Vapor_Monitoring
Water Vapor Monitoring is an analytic dashboard aimed to the visualization and analysis of GNSS time series for the improvement of weather forecast. 
It is currently developed for Linux and OS.

## Getting Started

### Prerequisites

For running the *Water Vapor Monitoring Dashboard* is needed to have installed the following packages, tested with that recommended versions:
- Dash 1.3.0 (```pip3 install dash==1.7.0```)
- Plotly  ( ```pip3 install plotly==4.4.1```)
- Pandas 0.24.2, Scipy 1.3.0, Numpy 1.16.4 ( ```pip3 install --user scipy matplotlib```,  ```sudo apt-get install python3-panda```)
- Psycopg2 2.7.6.1 ( ```pip3 install psycopg2-binary```)
- Sqlalchemy 1.3.2 ( ```pip3 install SQLAlchemy```)
- PostgreSQL 12, Postgis (``` apt-get install postgresql-12```, ``` apt-get install pgadmin4```) 
- Pyod 0.7.5.1 ( ```pip3 install pyod```)
- Pyproj 2.4.2 ( ```pip3 install pyproj==2.4.2```)
- Dask (``` python -m pip3 install "dask[complete]" ```)

On Debian VM install also llvm 10 before Pyod following the instructions:

 ``` 
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" 
```
```
wget https://apt.llvm.org/llvm.sh
```
```
chmod +x llvm.sh
```
```
sudo ./llvm.sh 10
```

Then set the path for the installation of Pyod package :
```
LLVM_CONFIG=/usr/lib/llvm-10/bin/llvm-config pip3 install pyod
```

In case of problems related to the Colorama package version install as:
```
pip3 install colorama==0.4.3
```
### Installing
### Creation of the database

Run pgAdmin4 application.
To create the database is needed to have installed a server with PostgreSQL with user name postgres and password user, and create a database called db. Check the file dbConfig.txt and set the database name, the user and the password in case is preferable to change those parameters.  
Open the pg_hba.conf file with administrator permission and change "peer" to "md5" on the line concerning "all" other users:

> local      all     all  <s>peer</s> md5 

In Linux Debian in case of password problem set also only the first line DB admin login as:

> local      all     postgres     trust 

The file can be opened with the following commands:
```
sudo vim /etc/postgresql/12/main/pg_hba.conf``` 
```
Or:
```
sudo nano /etc/postgresql/12/main/pg_hba.conf
```
On Mac:
```
sudo su , vim /Library/PostgreSQL/12/data/pg_hba.conf
```

Using vim: 
Edit like pressing ``` i```, then Exit saving``` :wq!```or without saving```:qa!```. 
 
Using nano commands edit the file then press ``` Ctrl+o ,  Enter ,  Ctrl+x ```.

Then restart: 
```
sudo /etc/init.d/postgresql reload
```
On Mac:
``` 
pg_ctl -D /usr/local/var/postgres -l /usr/local/var/postgres/server.log restart
```
where */usr/local/var/postgres is the location of the database storage area*, and */usr/local/var/postgres/server.log* is my log file for postgres.

Then from the terminal:
```
sudo -u postgres psql postgres
```
```
\password  postgres
```
``` 
user
```

### Running

If running the sripts for the first time or any edit to the script createSchema.py, before running it launches the script create_base_schema.py with the following command:
```
python3 create_base_schema.py 
```
To run the scripts download the .ZIP file from the repository and edit the path to the input folder in the createSchema.py file at line 181  ```input_folder = r'/path to… /input/' ``` . 

Run the file createSchema.py to create the database with PostgreSQL using the following instruction in the terminal:
```
python3 createSchema.py
```
To run the webpage:
Start the webpage application by typing in the terminal:
```
python3 app_3.py
```

To open as HTML page: ```page.html```.





## Authors
| Name and Surname  | Email                                  |
|-------------------|----------------------------------------|
| Sara Maffioli   | saramaffioli@outlook.it |


