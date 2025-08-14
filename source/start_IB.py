import subprocess
import time
import pyautogui
import pygetwindow as gw
from dotenv import load_dotenv
import os
load_dotenv()

IB_USERNAME = os.getenv('IB_USERNAME')
IB_PASSSWORD = os.getenv('IB_PASSSWORD')
PATH_TO_TWS = os.getenv('PATH_TO_TWS')

def check_if_app_open(app_name):
    windows = gw.getAllTitles()
    for title in windows:
        if title.strip():
            if app_name in title:
                print(f"Found window with title: {title}")
                time.sleep(1)  # Wait a bit for the window to be ready
                return True


def check_if_app_exist(app_name):
    windows = gw.getAllTitles()
    for title in windows:
        if title.strip():
            if app_name in title:
                print(f"Found window with title: {title}")
                return True



def open_ib_con():

    subprocess.Popen(PATH_TO_TWS, shell=True)
    login_open = False
    while not login_open:
        login_open = check_if_app_open("Login")
        time.sleep(1)    

    for i in range(10):
        windows = gw.getWindowsWithTitle('Login')
        if windows:
            windows[0].activate()
            break
        time.sleep(1)

    time.sleep(2)
    pyautogui.write("prosolukasz")
    pyautogui.press('tab')
    pyautogui.write("Ferrari0234!INTERACTIVE")
    pyautogui.press('enter')

    ib_open = False
    while not ib_open:
        ib_open = check_if_app_open("Interactive Brokers")
    
    time.sleep(15)  # Wait for the main window to be ready