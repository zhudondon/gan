"""
先定义类，必须有个初始化函数
"""
import os

import PIL.Image
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import transforms
from tensorflow.keras import Input
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Dense
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model

from data_loader import DataLoader

IMAGE_ORDERING = 'channels_last'

# A_path = "D:/work/pic/测试图片/110101199603010001.jpg"
# A_path = "D:/work/pic/20240716152854.png"
test_image = "D:/Users/ai/gan/test_out/test.png"
one_image = "D:/Users/ai/gan/test_out/one.png"

tf.config.threading.set_inter_op_parallelism_threads(16)

# 图片转数据
def image_to_data(image):
    return transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(image)


# 保存图片
def save_one_image(epoch, prediction, save_path="image"):
    if not os.path.exists("./images"):
        os.makedirs("./images")
    file_name = save_path + "/" + str(epoch) + ".png"
    image = PIL.Image.fromarray(np.uint8((prediction)))
    image.save(file_name)
    # save_image(tensor2im(prediction), file_name)


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
        if isinstance(input_image, tf.Tensor):  # get the data from a variable
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
        # 原图大小
        self.dataset_name = ""
        self.hr_height = 224
        self.hr_width = 224
        self.batch_size = 100

        # 4*4 16个点变成1个点 补丁的大小
        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # 数据加载器
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_size=(self.hr_height, self.hr_width))
        # 优化器
        self.optimizer = Adam(0.0001, 0.5)

        # 损失函数
        self.loss = binary_crossentropy

        # 输入结构 (h,w,c)
        input_shape = Input(shape=(56, 56, 3))
        # 编码器
        self.encoder = self.build_encoder(input_shape)
        # 编译
        self.encoder.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        if os.path.exists("encoder-weight.h5"):
            self.encoder.load_weights("encoder-weight.h5")


    # 构建块
    @staticmethod
    def build_block(x, filters, input_image=None, core=(3, 3), pooling=(2, 2), data_format=IMAGE_ORDERING,
                    alpha=0.2, momentum=0.8):
        if x is None:
            x = Conv2D(filters, core, data_format=data_format, padding='same')(input_image)
        else:
            x = Conv2D(filters, core, data_format=data_format, padding='same')(x)
        # 缩小一半
        x = MaxPooling2D(pooling, padding='same', strides=2)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = BatchNormalization(momentum=momentum)(x)
        return x

    @staticmethod
    def build_reverse_block(x, filters, core=(3, 3), up_sampling=(2, 2), data_format=IMAGE_ORDERING, alpha=0.2,
                            momentum=0.8):
        # 增大2倍
        x = UpSampling2D(up_sampling)(x)
        x = Conv2D(filters, core, data_format=data_format, padding='same')(x)
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


        # 解码部分
        x = self.build_reverse_block(x, 128)
        x = self.build_reverse_block(x, 256)
        x = self.build_reverse_block(x, 512)
        # 进行4倍放大，回到原图大小
        x = self.build_reverse_block(x, 512)
        x = self.build_reverse_block(x, 512)

        """ todo 这里可以会出现梯度消失的问题，可能需要改进为 残差块 """

        # sigmoid
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        # n个特征值，连接到一千个神经元，每个神经元对当前分类的贡献，这一千个神经元来决定 预测和真实结果的区别
        # 模型的值，是单个样品的结果，输出 1024个概率值标量 [[0.1,0.22,...]]
        # x = Dense(64 * 16)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        # decoded = Dense(1, activation='sigmoid')(x)
        # print(decoded)

        # 输入，输出
        return Model(input_img, decoded)

    def load_data(self, batch_size=50):
        """
            数据分批
        """
        image_hds, images = self.data_loader.load_data(batch_size)

        # 假设我们有 x_train 和 y_train 数据
        # dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        # 打乱数据，批处理大小为 32，进行预取
        # batch_size = 32
        # dataset = dataset.shuffle(buffer_size=len(x_train)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        # 定义 ImageDataGenerator 实例，并进行预处理（如归一化）
        # datagen = ImageDataGenerator(rescale=1. / 255)
        #
        # # 从文件夹加载训练数据，设置 target_size 和 batch_size
        # train_generator = datagen.flow_from_directory(
        #     'D:/Users/ai/gan/DIV2K_train_HR',  # 训练数据的目录
        #     target_size=(150, 150),  # 将图片调整到目标尺寸
        #     batch_size=32,  # 每次加载32张图片
        #     class_mode='categorical'  # 多类别分类任务，'categorical' 会返回 one-hot 编码的标签
        # )
        #
        # # 验证数据
        # validation_generator = datagen.flow_from_directory(
        #     'dataset/validation',
        #     target_size=(150, 150),
        #     batch_size=32,
        #     class_mode='categorical'
        # )

        return image_hds, images


    # def input_result(self,image_hds):
    #     x = Dense(64 * 16)(x)
    #     x = LeakyReLU(alpha=0.2)(x)
    #     decoded = Dense(1, activation='sigmoid')(x)


    def train(self, epochs=10000, batch_size=1, sample_interval=50):
        self.batch_size = batch_size
        for epoch in range(epochs):

            image_hds, images = self.load_data(batch_size)
            # 训练损失
            encode_loss = self.encoder.train_on_batch(images, image_hds)
            # 输出损失
            print("%d [loss: %f, acc.: %.2f%%]" % (epoch, encode_loss[0], 100 * encode_loss[1]))

            if epoch % sample_interval == 0:
                # 保存图片和权重
                test_input = np.array([DataLoader.im_read_224(test_image)]) / 127.5 - 1.
                predict = self.encoder.predict(test_input)
                # 转换为图片格式rgb 逆运算
                predict_image = (predict[0] + 1) * 127.5
                save_one_image(epoch, predict_image, save_path="images")
                self.encoder.save("encoder.h5")
                self.encoder.save_weights("encoder-weight.h5")

    def test(self):
        self.encoder.load_weights("encoder-weight.h5")
        test_input = np.array([DataLoader.im_read_224(one_image)]) / 127.5 - 1.
        predict = self.encoder.predict(test_input)
        # 转换为图片格式rgb 逆运算
        predict_image = (predict[0] + 1) * 127.5
        Image.fromarray(predict_image).show()


if __name__ == '__main__':
    # create_pic("D:/Users/ai/gan/DIV2K_train_HR", "D:/Users/ai/gan/train_input")
    # create_pic("D:/Users/ai/gan/test_input", "D:/Users/ai/gan/test_out")
    hd = VagueToHD()
    hd.train()

