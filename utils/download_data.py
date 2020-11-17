# coding: utf-8

import os
import requests

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_raw_dir = os.path.join(root_dir, 'data', 'raw')

def download(dataset_name):
    if dataset_name == 'adult':
        download_dir = os.path.join(data_raw_dir, 'adult')
        os.makedirs(download_dir, exist_ok=True)

        links = [
            {'filename': 'adult.data',
             'link': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'},
            {'filename': 'adult.test',
             'link': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'}]

        for l in links:
            file_name, link = l['filename'], l['link']
            download_path = os.path.join(download_dir, file_name)
            print(f'Download from {link} into {download_path}')

            data = requests.get(link, allow_redirects=True)
            with open(download_path, 'wb') as fd: fd.write(data.content)

    print('Finish to download')
