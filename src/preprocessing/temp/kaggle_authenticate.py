import os
from kaggle.api.kaggle_api_extended import KaggleApi

def verify_kaggle_auth():
    try:
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle authentication successful!")
        return True
    except Exception as e:
        print(f"✗ Authentication failed: {str(e)}")
        return False

if __name__ == "__main__":
    verify_kaggle_auth()