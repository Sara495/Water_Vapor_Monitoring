import pysftp

myHostname = "access834337968.webspace-data.io"
myUsername = "u101458685-tropo-nrt"
myPassword = "sZb.WHeXb2h3j"
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None
#Connect to SFTP page
with pysftp.Connection(host=myHostname, username=myUsername, password=myPassword, port=22, cnopts=cnopts) as sftp:
    print ('Connection succesfully stablished')

    # # Switch to a remote directory
    # sftp.cwd('/2021/')
    
    # # Print data
    # for attr in sftp.listdir():
    #     for attr_2 in sftp.listdir(attr):
    #       print (attr_2)

    # Switch to a remote directory
    sftp.cwd('/')

    # Print data
    for attr in sftp.listdir():
      print(attr)
      for attr_1 in sftp.listdir(attr):
        print(attr_1)
        lis=sftp.listdir(attr1)
        print(lis)
        #for attr_2 in sftp.listdir(attr_1):
         # print(attr_2)
# Define the file that you want to download from the remote directory
    #remoteFilePath = '/var/integraweb-db-backups/TUTORIAL.txt'

# Define the local path where the file will be saved
    # or absolute "C:\Users\sdkca\Desktop\TUTORIAL.txt"
    #localFilePath = './TUTORIAL.txt'

   # sftp.get(remoteFilePath, localFilePath)
# Define the file that you want to upload from your local directorty
    #sftp.remove('/var/custom-folder/TUTORIAL2.txt')
  
    
# connection closed automatically at the end of the with-block

# host: access834337968.webspace-data.io
# port: 22
# user: u101458685-tropo-nrt
# pw: 
# sZb.WHeXb2h3j


