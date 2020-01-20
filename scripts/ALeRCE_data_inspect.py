import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from parameters import param_keys
from modules.data_loaders.frame_to_input import FrameToInput
import numpy as np


if __name__ == "__main__":
    data_name = 'pancho_septiembre.pkl'
    data_folder = "/home/ereyes/Projects/Thesis/datasets/ALeRCE_data/"
    params = {
      param_keys.TEST_SIZE: 100,
      param_keys.VAL_SIZE: 0,
      param_keys.BATCH_SIZE: 0,
      param_keys.CHANNELS_TO_USE: [0, 1, 2],
      param_keys.NANS_TO: 0,
      param_keys.CROP_SIZE: 21,
      param_keys.DATA_PATH_TRAIN: data_folder+data_name,
      param_keys.CONVERTED_DATA_SAVEPATH: data_folder+'converted_'+data_name
    }
    frame_to_input = FrameToInput(params)
    test_dict = frame_to_input.get_dict()
    print(test_dict["labels"])
    print(test_dict["images"][:5])
    print(test_dict["images"][0].shape)
    print(np.unique(test_dict["labels"], return_counts=True))
    print(frame_to_input.class_dict)

    supernovaes_data_name = 'tns_confirmed_sn.pkl'
    params.update({
      param_keys.DATA_PATH_TRAIN: data_folder + supernovaes_data_name,
      param_keys.CONVERTED_DATA_SAVEPATH: data_folder + 'converted_' + supernovaes_data_name
    })
    frame_to_input = FrameToInput(params)
    sne_dict = frame_to_input.get_dict()
    print(np.unique(sne_dict["labels"], return_counts=True))
    print(frame_to_input.class_dict)

    bogus_data_name = 'bogus_juliano_franz_pancho.pkl'
    params.update({
      param_keys.DATA_PATH_TRAIN: data_folder + bogus_data_name,
      param_keys.CONVERTED_DATA_SAVEPATH: data_folder + 'converted_' + bogus_data_name
    })
    frame_to_input = FrameToInput(params)
    bogus_dict = frame_to_input.get_dict()
    print(np.unique(bogus_dict["labels"], return_counts=True))
    print(frame_to_input.class_dict)
