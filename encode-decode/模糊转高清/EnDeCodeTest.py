
'''
先定义类，必须有个初始化函数
'''
from tensorflow.python.keras.layers import Conv2D, ZeroPadding2D, LeakyReLU, BatchNormalization, UpSampling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
import tensorflow.python.keras.layers



IMAGE_ORDERING = 'channels_last'
class VagueToHD:
    def __init__(self):
        print("")
        ''' 这里要进行 各种初始化
            前置：
            1.优化器初始化
            2.数据加载 shape初始化，（w,h,c）
            3.存储文件
            4.损失函数
            
            1.编码器初始化
            2.解码器初始化
            3.模型初始化
        '''

        # 优化器
        self.optimizer = Adam(0.0002, 0.5)

        # 编码器
        self.encoder = self.build_encoder()

        # 解码器
        self.decoder = self.build_decoder()

    # 构建块
    def build_PCRB(self, filters, core=(3, 3), padding=(1, 1), data_format=IMAGE_ORDERING, alpha=0.2, momentum=0.8):
        x = ZeroPadding2D(padding, data_format=data_format)
        x = Conv2D(filters, core, data_format=data_format)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = BatchNormalization(momentum=momentum)(x)
        return x

    def build_reverse_PCRB(self, filters, core=(3, 3), padding=(1, 1), data_format=IMAGE_ORDERING, alpha=0.2, momentum=0.8):
        UpSampling2D()
        x = Conv2D(filters, core, data_format=data_format)(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = BatchNormalization(momentum=momentum)(x)
        return x
    def build_encoder(self):
        # 创建一个模型序列
        model = Sequential()
        # 卷积提前特征
        ''' 过程 224 * 224 * 512 -> 112 * 112 * 256 -> 56 * 56 * 128 -> 28 * 28 * 64 -> 14 * 14 * 32 -> 7 * 7 * 16 
            CRB
            IMAGE_ORDERING = channel_last (h,w,c)
            
            LeakyReLU(alpha=0.2) 函数中的alpha=0.2参数‌表示当输入值小于0时，该函数的斜率为0.2
            
            BatchNormalization(momentum=0.8) 动能为0.8的衰减
        '''
        model.add(ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))
        model.add(Conv2D(512, (3, 3), strides=1, data_format=IMAGE_ORDERING))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # 112 * 112 * 256
        model.add(ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))
        model.add(Conv2D(256, (3, 3), strides=1, data_format=IMAGE_ORDERING))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # 56 * 56 * 128
        model.add(ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))
        model.add(Conv2D(128, (3, 3), data_format=IMAGE_ORDERING))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # 28 * 28 * 64
        model.add(ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))
        model.add(Conv2D(28, (3, 3), data_format=IMAGE_ORDERING))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # 14 * 14 * 32
        model.add(ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))
        model.add(Conv2D(14, (3, 3), data_format=IMAGE_ORDERING))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # 7 * 7 * 16
        model.add(self.build_PCRB(7))
        return model


    def build_decoder(self, model):
        pass


    def load_data(self):
        pass


    def test(self):
        pass



if __name__ == '__main__':
    pass



