from PIL import Image
import argparse
import json
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser('prep')

parser.add_argument("--model", type=str, help="Which model to call") 
parser.add_argument("--image_file", type=str, help="Path to image file")

args = parser.parse_args()

img = Image.open(args.image_file)
np_img = np.asarray(img)

payload = {'category' : args.model, 'image' : np_img.tolist()}

json.dump(payload, open(Path(args.image_file).stem + '.json', 'w'))