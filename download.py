import kagglehub
import dataset_tools as dtools
# aerial images: https://datasetninja.com/cattle-detection-and-counting-in-uav-images#download

def kaggle_dwnld():
    # Download latest version
    path = kagglehub.dataset_download("trainingdatapro/cows-detection-dataset", output_dir="data/kagglehub")

    print("Path to cow dataset files:", path)

def dtools_dwnld():
    dtools.download(dataset='Cattle Detection and Counting in UAV Images', dst_dir='data/dtools')
