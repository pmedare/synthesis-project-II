import pandas as pd
import zipfile


def unzip_folder(file_path):
    # Unzip the uploaded file
    with zipfile.ZipFile(file_path + '/files.zip', 'r') as zip:
        zip.extractall(file_path)


def perform_data_analysis(file_folder):
    # Perform data analysis on the uploaded CSV file
    unzip_folder(file_folder + 'files.zip')
    print(file_folder + '/files.zip')
    classesDF = pd.read_csv(file_folder + '/node_classes.csv')
    edgesDF = pd.read_csv(file_folder + '/edges.csv')

    return True