import requests
from tqdm import tqdm
import pandas as pd
import os

def download_file_from_google_drive(id, destination):
    '''
    Downloads a file from Google Drive.

    This function initiates a session and sends a GET request to Google Drive's export URL with the provided file id. 
    It then retrieves a confirmation token and sends another GET request with the token to download the file. 
    The downloaded file is saved to the specified destination.

    Args:
        id (str): The id of the file to be downloaded.
        destination (str): The path where the downloaded file will be saved.

    Returns:
        None
    '''

    # Check if the file already exists
    if os.path.exists(destination):
        print("Dataset already exists at the destination.")
        return
    
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)



def get_confirm_token(response):
    '''
    Retrieves the confirmation token from the cookies of the response.

    This function iterates over the cookies in the response. If a cookie key starts with 'download_warning', 
    it returns the value of that cookie as the confirmation token.

    Args:
        response (requests.models.Response): The response from the GET request.

    Returns:
        str: The confirmation token, if found. Otherwise, None.
    '''
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None



def save_response_content(response, destination):
    '''
    Saves the content of the response to the destination path and displays a progress bar.

    This function retrieves the total file size from the response headers and initializes a progress bar. 
    It then opens the destination file in write mode and iterates over the content of the response. 
    If a chunk of content is not a keep-alive new chunk, it writes the chunk to the file and updates the progress bar.

    Args:
        response (requests.models.Response): The response from the GET request.
        destination (str): The path where the downloaded file will be saved.

    Returns:
        None
    '''
    CHUNK_SIZE = 32768

    # Get the total file size
    file_size = int(response.headers.get("Content-Length", 0))

    # Initialize a progress bar
    progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            # Filter out keep-alive new chunks
            if chunk:
                # Write the chunk to the file
                f.write(chunk)
                
                # Update the progress bar
                progress_bar.update(len(chunk))

    progress_bar.close()

    if file_size != 0 and progress_bar.n != file_size:
        print("ERROR, something went wrong")

#------------------------------------

def transform_csv_for_huggingface(dataset_path, destination='reviews_hf.csv'):
    '''
    This function transforms a CSV file to be compatible with Hugging Face's datasets library.

    The function performs the following operations:
    1. Reads the CSV file located at `dataset_path` into a pandas DataFrame.
    2. Renames the 'Label' column to 'label' to adhere to Hugging Face's naming conventions.
    3. Subtracts 1 from all values in the 'label' column. This is done because Hugging Face's datasets library expects labels to start from 0.
    4. Drops the 'Id' column as it is not needed for training.
    5. Writes the transformed DataFrame back to a CSV file named 'reviews_hf.csv'. The index is not included in the output file.

    Args:
        dataset_path (str): The path to the original CSV file.
        destination: The path to save result file
    Returns:
        None. The function writes the transformed DataFrame to a CSV file.
    '''
    # Check if the file already exists
    if os.path.exists(destination):
        print("Proccessed file already exists.")
        return
    review_df = pd.read_csv(dataset_path)
    # it is necessary to convert the Label to label
    review_df = review_df.rename(columns={'Label':'label'})
    review_df['label'] = review_df['label'] - 1
    review_df = review_df.drop('Id',axis=1)
    review_df.to_csv(destination,index=None)
