import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from parameters import param_keys
import numpy as np
import pandas as pd
import gzip
from astropy.io import fits
import io
from modules.data_loaders.ztf_stamps_loader import ZTFLoader
from tqdm import tqdm
import pickle


def get_image_from_bytes_stamp(stamp_byte):
    with gzip.open(io.BytesIO(stamp_byte), 'rb') as f:
        with fits.open(io.BytesIO(f.read())) as hdul:
            img = hdul[0].data            
    return img

class FrameToInput(ZTFLoader):

    def __init__(self, params):
        super().__init__(params)
        self.data_path = params[param_keys.DATA_PATH_TRAIN]
        self.converted_data_path = params[param_keys.CONVERTED_DATA_SAVEPATH]



    def _init_dataframe(self, df):
        self.data_frame = df
        self.n_points = len(self.data_frame)
        self.class_names = np.unique(self.data_frame["class"])
        self.class_dict = dict(zip(self.class_names, list(range(len(self.class_names)))))
        print(self.class_dict)
        self.stamp_keys = ["cutoutScience", "cutoutTemplate", "cutoutDifference"]
        self.labels = []
        self.images = []
        self.metadata = []


    def get_dict(self):
        loaded_data = pd.read_pickle(self.data_path)
        if not isinstance(loaded_data, pd.DataFrame):
            return loaded_data
            print("Recovering converted input")
        else:
            self._init_dataframe(loaded_data)
            for i in tqdm(range(self.n_points)):
                serie = self.data_frame.loc[i]
                label = self.class_dict[serie["class"]]
                image_array = []
                for key in self.stamp_keys:
                    image_array.append(get_image_from_bytes_stamp(serie[key]))
                image_tensor = np.stack(image_array, axis=2)
                self.labels.append(label)
                self.images.append(image_tensor)

            aux_dict = {"labels": self.labels, "images": self.images}
            pickle.dump(aux_dict, open(self.converted_data_path, "wb"), protocol=2)
            return aux_dict

    def _get_data(self, path):
        data_dict = self.get_dict()
        return data_dict

if __name__ == "__main__":

    params = {param_keys.DATA_PATH_TRAIN: "/home/rcarrasco/stamp_classifier/pickles/alerts_for_training.pkl"}
    frame_to_input = FrameToInput(params)
    test_dict = frame_to_input.get_dict()
    print(test_dict["labels"])
    print(test_dict["images"][:5])
    print(test_dict["images"][0].shape)
    print(np.unique(test_dict["labels"], return_counts=True))
    print(frame_to_input.class_dict)
