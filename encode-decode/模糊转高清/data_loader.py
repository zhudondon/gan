from glob import glob

import PIL.Image
import numpy as np
from skimage.transform import resize


class DataLoader:
    def __init__(self, dataset_name, img_size=(224, 224)):
        self.dataset_name = dataset_name
        self.img_size = img_size

    def load_data(self, batch_size=1, is_test=False):
        # data_type = "test" if is_test else "train"

        # path = glob('./datasets/%s/train/*' % self.dataset_name)
        path = glob('D:/Users/ai/gan/train_input/*')
        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = self.im_read(img_path)

            h, w = self.img_size
            low_h, low_w = int(h / 4), int(w / 4)

            img_hr = resize(img, self.img_size, anti_aliasing=True)
            img_lr = resize(img, (low_h, low_w), anti_aliasing=True)

            # If training => do random flip 随机镜像图片
            if not is_test and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr

    # 读取图片，用rgb模式读取，类型是float
    @staticmethod
    def im_read(path):
        return np.array(PIL.Image.open(path).convert('RGB')).astype(np.float32)

    @staticmethod
    def im_read_224(path):
        img = np.array(PIL.Image.open(path).convert('RGB')).astype(np.float32)
        img_hr = resize(img, (56, 56), anti_aliasing=True)
        return img_hr


