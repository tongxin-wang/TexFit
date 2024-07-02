# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import datasets
import os
from PIL import Image
import jsonlines


_CITATION = """\
@inproceedings{wang2024texfit,
  title={TexFit: Text-Driven Fashion Image Editing with Diffusion Models},
  author={Wang, Tongxin and Ye, Mang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={9},
  pages={10198--10206},
  year={2024}
}
"""

_DESCRIPTION = """\
A fashion image-region-text pair dataset called DFMM-Spotlight, highlighting local cloth.
"""

_HOMEPAGE = ""

_LICENSE = ""


class DFMMSpotlightDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def _info(self):

        features = datasets.Features(
            {
                "image": datasets.Image(),
                "mask": datasets.Image(),
                "text": datasets.Value("string")
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = '/path/to/DFMM-Spotlight'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "test",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        img_path = os.path.join(filepath, f'{split}_images')
        mask_path = os.path.join(filepath, 'mask')

        images = []
        masks = []
        texts = []
        with jsonlines.open(os.path.join(filepath, 'mask_ann', f'{split}_ann_file.jsonl'), 'r') as reader:
            for row in reader:
                images.append(row['image'])
                masks.append(row['mask'])
                texts.append(row['text'])

        dataset_len = len(images)
        for i in range(dataset_len):
            yield i, {
                        'image': Image.open(os.path.join(img_path, images[i])),
                        'mask': Image.open(os.path.join(mask_path, masks[i])),
                        "text": texts[i],
                    }
