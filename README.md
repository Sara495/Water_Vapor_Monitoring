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



## Authors
| Name and Surname  | Email                                  |
|-------------------|----------------------------------------|
| Sara Maffioli   | saramaffioli@outlook.it |


