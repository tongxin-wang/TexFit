import os
import os.path

import numpy as np
import torch
import jsonlines
import torch.utils.data as data
from PIL import Image
from random import choice

CLOTH_TYPES = ['tank top', 'tank shirt', 'T-shirt', 'shirt', 'sweater', 'upper clothing',
               'pants', 'shorts', 'trousers', 'skirt', 'lower clothing', 'outer clothing',
               'dress', 'rompers', 'belt', 'sunglasses', 'glasses', 'bag']

def check_cloth_type(text):
    ret_cloth_type = ''
    for cloth_type in CLOTH_TYPES:
        if cloth_type in text:
            ret_cloth_type = cloth_type
            break
    return ret_cloth_type

class DFMMSpotlight(data.Dataset):

    def __init__(self, mask_dir, img_dir, ann_file, downsample_factor=2):
        self._mask_path = mask_dir
        self._image_path = img_dir
        self._mask_fnames = []
        self._image_fnames = []
        self._cloth_texts = []
        self._cloth_text_groups = {}

        for cloth_type in CLOTH_TYPES:
            self._cloth_text_groups[cloth_type] = set()

        self.downsample_factor = downsample_factor

        # load text-region pair data
        assert os.path.exists(ann_file)
        with jsonlines.open(ann_file, 'r') as reader:
            for row in reader:
                self._mask_fnames.append(row['mask'])
                self._image_fnames.append(row['image'])
                row_cloth_type = check_cloth_type(row['text'])
                if row_cloth_type:
                    self._cloth_text_groups[row_cloth_type].add(row['text'])
                self._cloth_texts.append({'type': row_cloth_type, 'text': row['text']})

    def _open_file(self, path_prefix, fname):
        return open(os.path.join(path_prefix, fname), 'rb')

    def _load_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(self._image_path, fname) as f:
            image = Image.open(f)
            if self.downsample_factor != 1:
                width, height = image.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                image = image.resize(
                    size=(width, height), resample=Image.NEAREST)
            image = np.array(image).transpose(2, 0, 1)
        return image.astype(np.float32)

    def _load_mask(self, raw_idx):
        fname = self._mask_fnames[raw_idx]
        with self._open_file(self._mask_path, fname) as f:
            mask = Image.open(f)
            if self.downsample_factor != 1:
                width, height = mask.size
                width = width // self.downsample_factor
                height = height // self.downsample_factor
                mask = mask.resize(
                    size=(width, height), resample=Image.NEAREST)
            mask = np.array(mask)
        return mask.astype(np.float32)

    def __getitem__(self, index):
        mask = self._load_mask(index)
        image = self._load_image(index)
        text_info = self._cloth_texts[index]
        if text_info['type']:
            text = choice(list(self._cloth_text_groups[text_info['type']]))
        else:
            text = text_info['text']

        mask = mask / 255
        mask = torch.LongTensor(mask)
        image = torch.from_numpy(image)

        return_dict = {
            'mask': mask,
            'image': image,
            'text': text,
            'mask_name': self._mask_fnames[index],
            'img_name': self._image_fnames[index]
        }

        return return_dict

    def __len__(self):
        return len(self._image_fnames)
