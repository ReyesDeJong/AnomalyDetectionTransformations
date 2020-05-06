import os
import sys

"""
Must be run two times : 1 for download and 2 for extraction.
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

from pathlib import Path
from PIL import Image
from modules.utils import check_paths

png_path = os.path.join(PROJECT_PATH, '..',
    'fid_extra_files/valid_64x64/valid_64x64')
jpg_path = os.path.join(PROJECT_PATH, '..',
    'fid_extra_files/valid_64x64/valid_64x64_jpg')
check_paths(jpg_path)
inputPath = Path(png_path)
inputFiles = inputPath.glob("**/*.png")
outputPath = Path(jpg_path)
for f in inputFiles:
    print(f)
    outputFile = outputPath / Path(f.stem + ".jpg")
    im = Image.open(f)
    im.save(outputFile)