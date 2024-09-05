'''
先定义类，必须有个初始化函数
'''
import os

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, ZeroPadding2D, LeakyReLU, BatchNormalization, UpSampling2D, \
    MaxPooling2D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.losses import BCE
import tensorflow as tf
import tensorflow.python.keras.layers

IMAGE_ORDERING = 'channels_last'

test_image = ""


def format_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    return image


# 图片转数据
def image_to_data():
    pass


# 数据转图片
def data_to_image():
    pass


def save_image(epoch, prediction, path="image"):
    if not os.path.exists("./images"):
        os.makedirs("./images")


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
    pass
