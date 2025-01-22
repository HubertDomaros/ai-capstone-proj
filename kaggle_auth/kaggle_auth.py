import os

def auth():
    os.environ['KAGGLE_USERNAME'] = 'arct22'
    os.environ['KAGGLE_KEY'] = "1479f44eb40e4c43e5cafa8da3e314b0"

if __name__ == "__main__":
    auth()