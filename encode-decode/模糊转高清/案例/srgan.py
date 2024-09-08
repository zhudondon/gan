"""
先定义类，必须有个初始化函数

思路，如果要让图片达到原图效果，
那么当前图必须要学习 原图的特征，慢慢向原图靠拢


先生成一张图
原图，假图，进行对比：意思就是 走同一段 特征过程，得到特征，对比



"""
import datetime
import os

import PIL.Image
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Dense
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Reshape
from tensorflow_core.python.keras.layers import Conv2D, Activation, Add

from data_loader import DataLoader

IMAGE_ORDERING = 'channels_last'

# A_path = "D:/work/pic/测试图片/110101199603010001.jpg"
# A_path = "D:/work/pic/20240716152854.png"
test_image = "D:/Users/ai/gan/test_out/test.png"
one_image = "D:/Users/ai/gan/test_out/one.png"

tf.config.threading.set_inter_op_parallelism_threads(16)

K.set_image_data_format(IMAGE_ORDERING)


# 保存图片
def save_one_image(epoch, prediction, save_path="images"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = save_path + "/" + str(epoch) + ".png"
    image = PIL.Image.fromarray(np.uint8((prediction)))
    image.save(file_name)


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


class VagueToHD:
    def __init__(self):
        """
            定义高分辨率和低分辨率的结构，
        """
        # 低分辨率图的shape
        self.batch_size = None
        self.channels = 3
        self.lr_height = 56
        self.lr_width = 56
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        # 高分辨率图的shape
        self.hr_height = self.lr_height * 4
        self.hr_width = self.lr_width * 4
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        """ 
            残差块数量，
            优化器，
            收集特征的网络vgg，不进行训练
            获取数据集
        """
        # 16个残差卷积块
        self.n_residual_blocks = 16
        # 优化器
        optimizer = Adam(0.0002, 0.5)
        # 创建VGG模型，该模型用于提取特征
        self.vgg = self.build_vgg()
        self.vgg.trainable = False

        # 数据集
        # self.dataset_name = 'DIV'
        # self.data_loader = DataLoader(dataset_name=self.dataset_name,
        #                               img_res=(self.hr_height, self.hr_width))

        self.data_loader = DataLoader(dataset_name="", img_size=(self.hr_height, self.hr_width))
        """
            这里是缩小了2**4 大小，结果 看下 辨别器的大小
        """
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        """
            判别器 2分类交叉熵
            生成器模型 输入类型，低分辨率图片
            vgg用于计算生成结果特征，后续用于比较真实值的特征
        """
        # 建立判别模型
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        self.discriminator.summary()
        # 建立生成模型
        self.generator = self.build_generator()
        self.generator.summary()

        # 将生成模型和判别模型结合。训练生成模型的时候不训练判别模型。
        img_lr = Input(shape=self.lr_shape)
        # 生成假图，假图的特征提取
        fake_hr = self.generator(img_lr)
        fake_features = self.vgg(fake_hr)

        # self.discriminator.trainable = False
        # 判定器尝试判定一下假图
        validity = self.discriminator(fake_hr)
        """
            构造一个模型 用于生产假货的
            输入是低分辨率的图，输出是 (假图判定的结果,假图的特征提取)
            
            输出 对应的损失函数 二分类交叉熵，均方差 Mean Squared Error，均方误差
            误差的权重 0.5,1 后面 误差*0.5 + 误差 * 1
        """
        self.combined = Model(img_lr, [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[5e-1, 1],
                              optimizer=optimizer)

    """ 用于 区分生成和真实 图片的 差值， 用mse来计算损失
        输入图片，得到 特征图
    """

    def build_vgg(self):
        # 建立VGG模型，只使用第9层的特征
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_shape)
        # reshaped_img = Reshape((224, 224, 3))(img)
        # img_features = vgg(reshaped_img)
        img_features = vgg(img)
        return Model(img, img_features)

    def build_generator(self):
        # 创建一个模型序列
        # model = Sequential()
        #
        """
            残差块 padding=same 保证形状不变
            进行一次 CRB,CB 形状不变
            每次操作都提取特征，但是不改变形状（特征越来越集中和突出）
        """

        def residual_block(layer_input, filters):
            # 3*3 卷积，步长1, (n-3) / 3 +1 ,变小填充 （在输入的边缘填充适当数量的像素，使得卷积不会减少尺寸）。
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            # 把无效负值变为0，剩下有效的特征
            d = Activation('relu')(d)
            # 正则化，保证输入数据 均匀分布，但是防止反复横跳，加上动量，缓慢改变分布
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            # 保底，梯度不消失
            d = Add()([d, layer_input])
            return d

        """ 
            反卷积 ，卷积反向操作
        """
        def deconv2d(layer_input):
            # 上采样，2*2 增大一倍，这个操作复制 特征，比较粗暴 这个不产生学习参数的
            u = UpSampling2D(size=2)(layer_input)
            #  进行参数学习，形状不变，维度固定256
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # 生成器过程 低分辨率 形状
        img_lr = Input(shape=self.lr_shape)
        """ 
            这里初始化 低分辨输入，卷积得到网络初步结构 
            卷积核 9，快速 聚集特征 到 64维，激活有效特征
        """
        # 第一部分，低分辨率图像进入后会经过一个卷积+RELU函数
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        """ 
            传入c1 初始化一个残差网络
            循环创建残差块，加深网络深度，提取高级特征，维度和尺寸保持不变，但是 特征越来越集中和突出
        """
        # 第二部分，经过16个残差网络结构，每个残差网络内部包含两个卷积+标准化+RELU，还有一个残差边。
        r = residual_block(c1, 64)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, 64)

        """ 
            对残差结果 进行 卷积，形状不变，进一步 集中提取特征
            调整正则化输入
            前面三步 和残差网络很类似，完全没必要这么写
            
            两次 反卷积，形状*4，从而提高分辨率
        """
        # 第三部分，上采样部分，将长宽进行放大，两次上采样后，变为原来的4倍，实现提高分辨率。
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])
        # 两次 放大
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)
        # 这里最终 得到的图片形状，频道是维持不变的，所以 这里必须是 self.channels 也就是3
        # 另外，这里使用 卷积核9 进一步 集中 特征，
        # activation='tanh' 表示使用双曲正切激活函数 tanh，它会将卷积层的输出值压缩到 (-1, 1) 的区间
        # 输出的值，进行压缩
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        # 最终 模型 输入一张低分辨率的图片（张量），输出一个 抽取一大堆特征的，压缩到（-1,1）的 张量
        return Model(img_lr, gen_hr)

        """
            IMAGE_ORDERING = channel_last (h,w,c)

            LeakyReLU(alpha=0.2) 函数中的alpha=0.2参数‌表示当输入值小于0时，该函数的斜率为0.2

            BatchNormalization(momentum=0.8) 动能为0.8的衰减
        """

    """ 
        构建 辨别器
    """
    def build_discriminator(self):
        """
            为了方便 快速生成 提取特征的块，提供了一个方法
            主要是 C+LR + bn(可选) ，核为3

        """

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # 输入形状，是 高清图的图片，用于 得到高清图的特征，和 低分辨率的图片进行 比较
        # 初始化 输入得到 初始层
        # 由一堆的卷积+LeakyReLU+BatchNor构成
        d0 = Input(shape=self.hr_shape)
        """ 
            卷积，形状不变，特征连续 提取，并维度快速增加
            64，128,256,512
            每个维度，进行两次抽取，步长1,2 各进行一次，更为细致的特征
        """
        d1 = d_block(d0, 64, bn=False)
        d2 = d_block(d1, 64, strides=2)
        d3 = d_block(d2, 128)
        d4 = d_block(d3, 128, strides=2)
        d5 = d_block(d4, 256)
        d6 = d_block(d5, 256, strides=2)
        d7 = d_block(d6, 512)
        d8 = d_block(d7, 512, strides=2)

        # 最终得到 形状不变（高分辨率图片），512层的特征图 第8层 strides = 2,形状减小一半
        # 全连接到 1024 各节点（神经元）
        """
            dense公式 y=f(Wx+b) 
            W：权重矩阵，维度为 (输入维度, 输出维度)。
            x：输入数据，通常是上一层的输出。
            b：偏置项，维度为 (输出维度,)。
            f：激活函数，如 ReLU、Sigmoid、tanh 等。
            
            无激活函数（activation=None）：输出的值就是 Wx+b，为线性变换后的结果。
            ReLU（activation='relu'）：输出是 max(0,Wx+b)，将负值部分变为 0，常用于深度神经网络的隐藏层。
            Sigmoid（activation='sigmoid'）：将输出限制在 (0,1) 之间，常用于二分类任务的输出层。
            tanh（activation='tanh'）：将输出限制在 (−1,1) 之间。
            2**4在这里来的
        """
        d9 = Dense(64 * 16)(d8)
        # 进行 9,10 相当于 Dense(64 * 16,activation='LeakyReLU')(d8) 但是默认好像没这个 LeakyReLU，所以拆分了 d9 = 56,56,1024
        d10 = LeakyReLU(alpha=0.2)(d9)
        # 最终，全连接到一个 单元，得到一个 0,1的值
        # 这是一个全连接层，输出单元为 1，意味着该层只会输出一个标量。
        # 该层会从输入数据中提取线性组合特征，输入的每个元素都会乘以一个权重，所有加权和加上偏置项形成输出。概率值，用于判断 误差 56,56,1
        validity = Dense(1, activation='sigmoid')(d10)

        # 返回模型，输入 d0的层，输出 一个概率值
        return Model(d0, validity)

    """ 
        调度器，控制学习率，遍历所有模型，变小 学习率，动态变化
        每过2w 代进行下降
    """

    def scheduler(self, models, epoch):
        # 学习率下降
        if epoch % 20000 == 0 and epoch != 0:
            for model in models:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))

    def load_data(self, batch_size=50):
        """
            数据分批
        """
        image_hds, images = self.data_loader.load_data(batch_size=batch_size, img_size=(self.hr_height, self.hr_width))
        return image_hds, images

    def train(self, epochs=60000, batch_size=1, sample_interval=60, init_epoch=0):
        start_time = datetime.datetime.now()

        if init_epoch != 0:
            self.generator.load_weights("weights/gen_epoch%d.h5" % init_epoch, skip_mismatch=True)
            self.discriminator.load_weights("weights/dis_epoch%d.h5" % init_epoch, skip_mismatch=True)

        for epoch in range(init_epoch, epochs):
            self.scheduler([self.combined, self.discriminator], epoch)

            self.batch_size = batch_size
            imgs_hr, imgs_lr = self.load_data(batch_size)

            """
                输入低分辨率图片 预测假图片
                构造对比结果，根据图形的结构，例如 图片是 4*4，那么就有16个对比结果 正确
                同理错误 
            """
            fake_hr = self.generator.predict(imgs_lr)
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            """
                把真实的图片批量传入辨别器，告诉他是 1:16 的变换结果，都是对的
                把批量生成的假图片传入辨别器，告诉他 1:16 的变换结果，都是假的
                得到辨别器 认知中的 对错，和实际的 对错的 误差
                误差加起来平均值，可以用于观察辨别器的整体辨别情况
            """
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            """
                vgg提取高分辨率图片的特征
                联合训练，传入 低分辨率图片，期望值 都是valid，而且特征是对应上 
                打印 d,g,c 的误差情况
            """
            # 这里要重新拿一次是 为了保证这个输入 大小 控制为 224*224*3, 这里 用于检测特征，但是vgg硬性要求 224，所以要保证预测，和输入的真实值都要进行 转换
            # 成 224 * 224 * 3
            imgs_hr, imgs_lr, imgs_hr_224 = self.data_loader.load_data_triple(batch_size=batch_size, img_size=(self.hr_height, self.hr_width))
            valid = np.ones((batch_size,) + self.disc_patch)
            image_features = self.vgg.predict(imgs_hr_224)
            g_loss = self.combined.train_on_batch(imgs_lr, [valid, image_features])
            print(d_loss, g_loss)
            elapsed_time = datetime.datetime.now() - start_time

            print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, feature loss: %05f] time: %s " \
                  % (epoch, epochs,
                     d_loss[0], 100 * d_loss[1],
                     g_loss[1],
                     g_loss[2],
                     elapsed_time))


            if epoch > 0 and epoch % sample_interval == 0:
                # 保存图片和权重
                test_input = np.array([DataLoader.im_read_real(test_image, size=(self.lr_height, self.lr_width))]) / 127.5 - 1.
                predict = self.generator.predict(test_input)
                # 转换为图片格式rgb 逆运算
                predict_image = (predict[0] + 1) * 127.5
                save_one_image(epoch, predict_image, save_path="sr-images")
                self.generator.save("sr-generator.h5")
                self.generator.save_weights("sr-generator-weight.h5")

                self.discriminator.save("sr-discriminator.h5")
                self.discriminator.save_weights("sr-discriminator-weight.h5")

                self.combined.save("sr-combine.h5")
                self.combined.save_weights("sr-combine-weight.h5")

                if epoch % 500 == 0:
                    os.makedirs('weights', exist_ok=True)
                    self.generator.save("weights/gen_epoch%d.h5" % init_epoch)
                    self.discriminator.save("weights/gen_epoch%d.h5" % init_epoch)


    def test(self):
        self.generator.load_weights("generator-weight.h5")
        test_input = np.array([DataLoader.im_read_real(one_image, self.lr_height, self.lr_width)]) / 127.5 - 1.
        predict = self.generator.predict(test_input)
        # 转换为图片格式rgb 逆运算
        predict_image = (predict[0] + 1) * 127.5
        Image.fromarray(predict_image).show()


if __name__ == '__main__':
    hd = VagueToHD()
    hd.train()
