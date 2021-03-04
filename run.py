import schedule
import time
import os

def job():
    print("I'm working...")
    os.system('python3 createSchema_real.py')
schedule.every().hour.at(":00").do(job)


while True:
    schedule.run_pending()
    time.sleep(1)