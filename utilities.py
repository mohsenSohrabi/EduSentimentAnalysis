import requests
from tqdm import tqdm
import pandas as pd
import os
import numpy as np

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

def transform_csv_for_huggingface(dataset_path, p = 0.25):
    """
    Transforms a CSV file for use with the Hugging Face library.

    This function reads a CSV file, renames the 'Label' column to 'label', 
    subtracts 1 from all the labels, drops the 'Id' column, and splits the 
    DataFrame into a training set and a test set. The split is determined by 
    the parameter p, which represents the proportion of the DataFrame to be 
    used for the training set. The function then saves the training and test 
    sets as separate CSV files.

    Args:
        dataset_path (str): The path to the CSV file.
        p (float, optional): The proportion of the DataFrame to be used for 
            the training set. Defaults to 0.25.

    Returns:
        None. The function saves the training and test sets as 'review_df_train.csv' 
        and 'review_df_test.csv', respectively.
    """
    review_df = pd.read_csv(dataset_path)
    # it is necessary to convert the Label to label
    review_df = review_df.rename(columns={'Label':'label'})
    review_df['label'] = review_df['label'] - 1
    review_df = review_df.drop('Id',axis=1)
    n = int(np.floor(p * review_df.shape[0]))
    review_df_train = review_df.iloc[:n]
    review_df_test = review_df.iloc[n:]

    review_df_train.to_csv('review_train.csv',index=None)
    review_df_test.to_csv('review_test.csv',index=None)
