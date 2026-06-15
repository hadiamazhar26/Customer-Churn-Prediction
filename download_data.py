import pandas as pd

def download_dataset():
    url = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    print(f"Downloading Telco Customer Churn dataset from: {url}...")
    
    try:
        df = pd.read_csv(url)
        df.to_csv("customer_churn.csv", index=False)
        print(f"Successfully downloaded and saved dataset to 'customer_churn.csv'.")
        print(f"Shape of the dataset: {df.shape}")
        return True
    except Exception as e:
        print(f"Error downloading the dataset: {e}")
        return False

if __name__ == "__main__":
    download_dataset()
