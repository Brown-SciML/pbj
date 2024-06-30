import os
import pandas as pd
import numpy as np
from PIL import Image

'''Code adapated from: https://www.kaggle.com/code/wenewone/transfer-learning-example-on-cub-200-2011-dataset'''
class CUB():
    def __init__(self, root, dataset_type='train', transform=None, target_transform=None, is_proto=False):
        self.isproto = is_proto
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        bb = pd.read_csv(os.path.join(root, 'bounding_boxes.txt'), sep=' ', header=None, names=['ID', 'xmin', 'ymin', 'width', 'height'], index_col=0)
        bb['xmin'] = np.where(bb['width'] < bb['height'], bb['xmin'] - ((bb['height'] - bb['width'])/2), bb['xmin'])
        bb['xmax'] = np.where(bb['width'] < bb['height'], bb['xmin'] + bb['width'] + 2*(((bb['height'] - bb['width'])/2)), bb['xmin'] + bb['width'])
        bb['ymin'] = np.where(bb['height'] < bb['width'], bb['ymin'] - ((bb['width'] - bb['height'])/2), bb['ymin'])
        bb['ymax'] = np.where(bb['height'] < bb['width'], bb['ymin'] + bb['height'] + 2*(((bb['width'] - bb['height'])/2)), bb['ymin'] + bb['height'])
        
        df_img = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ', header=None, names=['ID', 'Image'], index_col=0)
        df_label = pd.read_csv(os.path.join(root, 'image_class_labels.txt'), sep=' ', header=None, names=['ID', 'Label'], index_col=0)
        df_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', header=None, names=['ID', 'Train'], index_col=0)
        df = pd.concat([df_img, df_label, df_split], axis=1)
        df['bb']= bb.values.tolist()
        # relabel
        df['Label'] = df['Label'] - 1

        # split data
        if dataset_type == 'test':
            df = df[df['Train'] == 0]
        elif dataset_type == 'train' or dataset_type == 'valid':
            df = df[df['Train'] == 1]
        else:
            raise ValueError('Unsupported dataset_type!')
        self.img_name_list = df['Image'].tolist()
        self.label_list = df['Label'].tolist()
        self.bb = df['bb'].tolist()
        # Convert greyscale images to RGB mode
        self._convert2rgb()
        self._applycrop()

    def __len__(self):
        return len(self.label_list)
    
    def get_img_path(self, idx):
        img_path = os.path.join(self.root, 'images', self.img_name_list[idx])
        return img_path

    def __getitem__(self, idx):
        img_path = self.get_img_path(idx)
        image = Image.open(img_path)
        target = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target
    
    def _convert2rgb(self):
        """
        Converts greyscale images to RGB mode.
        """
        for i, img_name in enumerate(self.img_name_list):
            img_path = os.path.join(self.root, 'images', img_name)
            image = Image.open(img_path)
            color_mode = image.mode
            if color_mode != 'RGB':
                if not os.path.isfile(img_path.replace('.jpg', '_rgb.jpg')):
                    image = image.convert('RGB')
                    image.save(img_path.replace('.jpg', '_rgb.jpg'))
                self.img_name_list[i] = img_name.replace('.jpg', '_rgb.jpg')

    def _applycrop(self):
        for i, img_name in enumerate(self.img_name_list):
            img_path = os.path.join(self.root, 'images', img_name)
            if not os.path.isfile(img_path.replace('.jpg', '_crop.jpg')):
                image = Image.open(img_path)
                image = image.crop([self.bb[i][0], self.bb[i][1], self.bb[i][4], self.bb[i][5]])
                image.save(img_path.replace('.jpg', '_crop.jpg'))
            self.img_name_list[i] = img_name.replace('.jpg', '_crop.jpg')
