import os
from PIL import Image

from torch.utils.data import Dataset


def extract_images(root):
    """ Extract all images under root """
    img_ext = ['.png', '.jpg', '.jpeg']
    root = os.path.expanduser(root)
    img_paths = []
    for curdir, subdirs, files in os.walk(root):
        for file in files:
            if os.path.splitext(file)[1].lower() in img_ext:
                img_paths.append(os.path.join(curdir, file))
    img_paths = sorted(img_paths)
    return img_paths


class FFHQ(Dataset):
    """The Flickr-Faces-HQ (FFHQ) Dataset.

    Flickr-Faces-HQ (FFHQ) consists of 70,000 high-quality PNG images at 1024×1024 resolution and contains
    considerable variation in terms of age, ethnicity and image background. It also has good coverage of
    accessories such as eyeglasses, sunglasses, hats, etc. The images were crawled from Flickr, thus inheriting
    all the biases of that website, and automatically aligned and cropped using dlib. Only images under
    permissive licenses were collected. Various automatic filters were used to prune the set, and finally
    Amazon Mechanical Turk was used to remove the occasional statues, paintings, or photos of photos.
    (Copied from PapersWithCode)

    There are several versions of the dataset in the official Google Drive link, among which `images1024x1024`
    is the most widely used one.

    Please organize the dataset in the following file structure:

    root
    ├── ffhq-dataset-v2.json
    ├── LICENSE.txt
    ├── README.txt
    ├── thumbnails128x128
    │   ├── 00000.png
    │   ├── ...
    │   └── 69999.png
    └── images1024x1024
        ├── 00000.png
        ├── ...
        └── 69999.png

    Reference:
      - https://github.com/NVlabs/ffhq-dataset
      - https://paperswithcode.com/dataset/ffhq
      - https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL

    """

    def __init__(self, root, split='train', version='images1024x1024', transform=None):
        if split not in ['train', 'test', 'all']:
            raise ValueError(f'Invalid split: {split}')
        if version not in ['images1024x1024', 'thumbnails128x128']:
            raise ValueError(f'Invalid version: {version}')
        self.root = root
        self.split = split
        self.version = version
        self.transform = transform

        # Extract image paths
        image_root = os.path.join(os.path.expanduser(self.root), version)
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')
        self.img_paths = extract_images(image_root)
        if split == 'train':
            self.img_paths = list(filter(lambda p: 0 <= int(os.path.basename(p).split('.')[0]) < 60000, self.img_paths))
        elif split == 'test':
            self.img_paths = list(filter(lambda p: 60000 <= int(os.path.basename(p).split('.')[0]) < 70000, self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        x = Image.open(self.img_paths[index]).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        return x
