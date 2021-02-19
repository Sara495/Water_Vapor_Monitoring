import schedule
import time
import os

def job():
    print("I'm working...")
    os.system('python3 test_connection.py')
schedule.every().minute.at(":17").do(job)


while True:
    schedule.run_pending()
    time.sleep(1)