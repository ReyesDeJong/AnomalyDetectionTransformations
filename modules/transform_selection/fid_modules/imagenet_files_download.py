import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)
from modules.utils import check_paths
from selenium import webdriver


def download_data_from_web_link(web_url, text_link, save_folder_path):
  # To prevent download dialog
  profile = webdriver.FirefoxProfile()
  profile.set_preference("browser.preferences.instantApply", True)
  profile.set_preference("browser.download.folderList", 2);
  profile.set_preference("browser.download.dir", save_folder_path);
  profile.set_preference("browser.helperApps.neverAsk.saveToDisk",
                         "text/plain, application/octet-stream, application/binary,"
                         " text/csv, application/csv, application/excel,"
                         " text/comma-separated-values, text/xml, application/xml")
  profile.set_preference("browser.helperApps.alwaysAsk.force", False)
  profile.set_preference("browser.download.manager.showWhenStarting", False)

  # Open a browser and log in

  browser = webdriver.Firefox(firefox_profile=profile)
  browser.get(web_url)

  browser.find_element_by_link_text(text_link).click()



if __name__ == '__main__':
  save_folder_path = os.path.join(PROJECT_PATH, '..', 'fid_extra_files')
  check_paths(save_folder_path)
  imagenet_url = 'http://image-net.org/small/download.php'
  image_net_text_link = 'Validation (64x64)'
  download_data_from_web_link(imagenet_url, image_net_text_link,
                              save_folder_path)
  os.remove("geckodriver.log")
