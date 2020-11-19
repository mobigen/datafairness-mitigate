# coding: utf-8

import os
import requests
import zipfile

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
    elif dataset_name == 'bank':
        download_dir = os.path.join(data_raw_dir, 'bank')
        os.makedirs(download_dir, exist_ok=True)

        links = [{
            'filename': 'bank-additional.zip',
            'link': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'}]

        file_name, link = links[0]['filename'], links[0]['link']
        download_path = os.path.join(download_dir, file_name)
        if not os.path.exists(download_path):
            print(f'Download from {link} into {download_path}')

            data = requests.get(link, allow_redirects=True)
            with open(download_path, 'wb') as fd: fd.write(data.content)

        print(f'압축 해제하는 중: {download_path}')
        zip_f = zipfile.ZipFile(download_path)
        extract_dirname = 'bank-additional'
        extract_filename = 'bank-additional-full.csv'
        inner_path = f'{extract_dirname}/{extract_filename}'
        zip_f.extract(inner_path, download_dir)
        zip_f.close()
        os.replace(os.path.join(download_dir, inner_path), os.path.join(download_dir, extract_filename))
        os.removedirs(os.path.join(download_dir, extract_dirname))
        os.remove(os.path.join(download_dir, file_name))
    else:
        raise Exception(f'Unsupported dataset name: {dataset_name}')

    print('Finish to download')
