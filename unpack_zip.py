from zipfile import ZipFile


with ZipFile('stage1_train.zip','r') as zip_ref:
    zip_ref.extractall('./stage1_train')