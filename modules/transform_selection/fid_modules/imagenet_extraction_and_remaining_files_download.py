import os
import sys

"""
Must be run two times : 1 for download and 2 for extraction.
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)
from modules.utils import check_paths
import tarfile
from modules.utils import check_paths
from os import listdir
from os.path import isfile, join


def heavy_file_download(save_path, url):
  import requests
  file_url = url
  filename = url.split('/')[-1]
  r = requests.get(file_url, stream=True)
  file_path = os.path.join(save_path, filename)
  if os.path.exists(file_path):
    return filename
  else:
    with open(file_path, "wb") as pdf:
      for chunk in r.iter_content(chunk_size=1024):

        # writing one chunk at a time to pdf file
        if chunk:
          pdf.write(chunk)
    return filename

def extract_file(path):
  my_tar = tarfile.open(path)
  file_name = path.split('/')[-1].split('.')[0]
  file_folder_list = path.split('/')[:-1]
  file_folder_path = os.path.join('/'.join(file_folder_list), file_name)
  check_paths(file_folder_path)
  my_tar.extractall(file_folder_path)  # specify which folder to extract to
  my_tar.close()

if __name__ == '__main__':
  save_folder_path = os.path.join(PROJECT_PATH, '..', 'fid_extra_files')
  check_paths(save_folder_path)
  model_down_link = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
  cifar10_precalc_stat_link = 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz'
  imagenet_val_precalc_stat_link = 'http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_imagenet_valid.npz'
  links = [model_down_link, cifar10_precalc_stat_link, imagenet_val_precalc_stat_link]
  for link_i in links:
    if os.path.exists(os.path.join(save_folder_path, link_i.split('/')[-1])):
      continue
    heavy_file_download(save_folder_path, link_i)

  onlyfiles = [f for f in listdir(save_folder_path) if isfile(join(save_folder_path, f))]
  for file_name in onlyfiles:
    if file_name.split('.')[-1] in ['tar', 'tgz']:
      print('extracting %s...' % file_name)
      extract_file(os.path.join(save_folder_path, file_name))