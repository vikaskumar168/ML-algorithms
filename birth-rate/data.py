import requests
import os
import time

def download_kaggle_dataset(url, output_filename="focusing-on-mobile-app-or-website.zip"):
    """
    Downloads a file from a specified URL and saves it to a local file.

    Note: This specific Kaggle URL requires the -L (follow redirects) flag
    when using curl. The Python requests library handles redirects by default.
    You may need to be logged into Kaggle or have a Kaggle API token set up
    for this link to successfully serve the file, as it often requires authentication.

    Args:
        url (str): The URL of the file to download.
        output_filename (str): The local path and filename to save the file as.
    """
    print(f"Starting download from: {url}")
    print(f"Saving to: {output_filename}")

    # Use a small delay and retry mechanism for robustness
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use stream=True to download large files without loading them into memory all at once
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 # 1 Kibibyte
            downloaded_size = 0

            with open(output_filename, 'wb') as file:
                for data in response.iter_content(block_size):
                    downloaded_size += len(data)
                    file.write(data)

                    # Simple progress indicator (optional)
                    if total_size > 0:
                        progress = downloaded_size / total_size * 100
                        print(f"Progress: {progress:.2f}%", end='\r')
            
            print(f"\nSuccessfully downloaded dataset to {output_filename}")
            return

        except requests.exceptions.RequestException as e:
            print(f"\nAttempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Maximum retries reached. Download failed.")
                break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            break


if __name__ == "__main__":
    # The URL provided in the prompt
    DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/kolawale/focusing-on-mobile-app-or-website"
    
    # Define the output path in the current directory
    OUTPUT_FILE = "./focusing-on-mobile-app-or-website.zip"
    
    # Check if the file already exists and prompt the user (optional check)
    if os.path.exists(OUTPUT_FILE):
        print(f"Warning: File '{OUTPUT_FILE}' already exists and will be overwritten.")

    download_kaggle_dataset(DATASET_URL, OUTPUT_FILE)