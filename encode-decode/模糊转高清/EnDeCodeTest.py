"""
先定义类，必须有个初始化函数
"""
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from PIL import Image
from matplotlib import transforms
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, LeakyReLU, BatchNormalization, UpSampling2D, \
    MaxPooling2D
from tensorflow.python.keras.losses import BCE
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.types.core import Tensor

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_ORDERING = 'channels_last'

# A_path = "D:/work/pic/测试图片/110101199603010001.jpg"
# A_path = "D:/work/pic/20240716152854.png"
test_image = "D:/work/pic/1721203779179.png"


# 图片转数据
def image_to_data(image):
    return transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(image)


# 保存图片
def save_image(epoch, prediction, path="image"):
    if not os.path.exists("./images"):
        os.makedirs("./images")
    file_name = path + "/" + epoch + ".png"
    save_image(tensor2im(prediction), file_name)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
        aspect_ratio:
    """
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


# 数据转图片
def tensor2im(input_image, im_type=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        im_type (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(im_type)


# 创建基础图片
def create_pic(path, n_path):
    path_list = [path + "/" + pic for pic in os.listdir(path)]
    for file in path_list:
        im = Image.open(file)
        # 原始大小
        (x, y) = im.size
        # 标准，比例
        n_x = 224
        n_y = int(y * n_x / x)

        # 改变尺寸，保持图片高品质
        out = im.resize((n_x, n_y), Image.ANTIALIAS)

        if out.mode == 'RGBA':
            out = out.convert('RGB')

        if not os.path.exists(n_path):
            os.makedirs(n_path)
        out.save(n_path + "/{}".format(file.split("/")[-1]))


class VagueToHD:
    def __init__(self):
        """ 这里要进行 各种初始化
            前置：
            1.优化器初始化
            2.数据加载 shape初始化，（w,h,c）
            3.存储文件
            4.损失函数

            1.编码器初始化 && 解码器初始化 && 模型初始化
        """

        # 优化器
        self.optimizer = Adam(0.0002, 0.5)

        # 损失函数
        self.loss = BCE

        # 输入结构
        input_shape = Input(shape=(224, 224))
        # 编码器
        self.encoder = self.build_encoder(input_shape)
        # 编译
        self.encoder.compile(optimizer=self.optimizer, loss=self.loss)

    # 构建块
    @staticmethod
    def build_block(x, filters, input_image=None, core=(3, 3), pooling=(2, 2), data_format=IMAGE_ORDERING,
                    alpha=0.2, momentum=0.8):
        if x is None or input_image is None:
            x = Conv2D(filters, core, data_format=data_format, padding='same')(x)
        else:
            x = Conv2D(filters, core, data_format=data_format, padding='same')(input_image)
        # 缩小一半
        x = MaxPooling2D(pooling, padding='same')(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = BatchNormalization(momentum=momentum)(x)
        return x

    @staticmethod
    def build_reverse_block(x, filters, core=(3, 3), up_sampling=(2, 2), data_format=IMAGE_ORDERING, alpha=0.2,
                            momentum=0.8):
        x = Conv2D(filters, core, data_format=data_format, padding='same')(x)
        # 增大2倍
        x = UpSampling2D(up_sampling)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = BatchNormalization(momentum=momentum)(x)
        return x

    def build_encoder(self, input_img):
        # 创建一个模型序列
        # model = Sequential()
        # 卷积提前特征
        """
        图片转换大小，比较暴力
        image = tf.io.read_file("image.jpg")
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])

        过程 224 * 224 * 512 -> 112 * 112 * 256 -> 56 * 56 * 128 -> 28 * 28 * 64 -> 14 * 14 * 32 -> 7 * 7 * 16
            CRB
            IMAGE_ORDERING = channel_last (h,w,c)

            LeakyReLU(alpha=0.2) 函数中的alpha=0.2参数‌表示当输入值小于0时，该函数的斜率为0.2

            BatchNormalization(momentum=0.8) 动能为0.8的衰减
        """
        x = self.build_block(None, 512, input_image=input_img)

        # 112 * 112 * 256
        x = self.build_block(x, 256)

        # 56 * 56 * 128
        x = self.build_block(x, 128)

        # 28 * 28 * 64
        x = self.build_block(x, 64)

        # 14 * 14 * 32
        x = self.build_block(x, 32)

        # 7 * 7 * 16
        x = self.build_block(x, 16)

        # 解码部分
        x = self.build_reverse_block(x, 16)
        x = self.build_reverse_block(x, 32)
        x = self.build_reverse_block(x, 64)
        x = self.build_reverse_block(x, 128)
        x = self.build_reverse_block(x, 256)
        x = self.build_reverse_block(x, 512)

        # sigmoid
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        # 输入，输出
        return Model(input_img, decoded)

    @staticmethod
    def load_data():
        image_hds = None
        images = None

        """
            数据分批
        """
        # 假设我们有 x_train 和 y_train 数据
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        # 打乱数据，批处理大小为 32，进行预取
        batch_size = 32
        dataset = dataset.shuffle(buffer_size=len(x_train)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        # 定义 ImageDataGenerator 实例，并进行预处理（如归一化）
        datagen = ImageDataGenerator(rescale=1. / 255)

        # 从文件夹加载训练数据，设置 target_size 和 batch_size
        train_generator = datagen.flow_from_directory(
            'E:/train/gan/DIV2K_train_HR',  # 训练数据的目录
            target_size=(150, 150),  # 将图片调整到目标尺寸
            batch_size=32,  # 每次加载32张图片
            class_mode='categorical'  # 多类别分类任务，'categorical' 会返回 one-hot 编码的标签
        )

        # 验证数据
        validation_generator = datagen.flow_from_directory(
            'dataset/validation',
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical'
        )





        return image_hds, images

    def train(self, epochs=100, batch_size=128, sample_interval=50):
        image_hds, images = self.load_data()

        for epoch in range(epochs):
            # 训练损失
            encode_loss = self.encoder.train_on_batch(images, image_hds)
            # 输出损失
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, encode_loss[0], 100 * encode_loss[1]))

            if epoch % sample_interval == 0:
                # 保存图片和权重
                predict = self.encoder.predict(test_image)
                save_image(epoch, predict, path="images")
                self.encoder.save("encoder.h5")
                self.encoder.save_weights("encoder-weight.h5")

        pass

    def test(self):
        pass


if __name__ == '__main__':
    create_pic("E:/train/gan/DIV2K_train_HR/DIV2K_train_HR", "E:/train/gan/DIV2K_train_HR/DIV2K_train")
