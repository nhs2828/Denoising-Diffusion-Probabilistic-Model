import tensorflow as tf
from tensorflow.keras.layers import *

class Unet(tf.keras.Model):
    def __init__(self, img_size, in_channels, out_channels):
        super().__init__()

        # block 1
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation='relu', kernel_initializer = 'he_normal', input_shape=(img_size, img_size, in_channels))
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # block 2
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # block 3
        self.conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.conv6 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # block 4
        self.conv7 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.conv8 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.maxpool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # block 5
        self.conv9 = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.conv10 = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.upconv1 = Conv2DTranspose(filters=512, kernel_size=(2, 2), strides=(2, 2), padding="valid", kernel_initializer = 'he_normal', activation='relu')

        # block 6
        self.conv11 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.conv12 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.upconv2 = Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=(2, 2), padding="valid", kernel_initializer = 'he_normal', activation='relu')

        # block 7
        self.conv13 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.conv14 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.upconv3 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding="valid", kernel_initializer = 'he_normal', activation='relu')

        # block 8
        self.conv15 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.conv16 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.upconv4 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding="valid", kernel_initializer = 'he_normal', activation='relu')

        # block 9
        self.conv17 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.conv18 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer = 'he_normal', activation='relu')
        self.out = Conv2D(filters=out_channels, kernel_size=(1, 1), strides=(1, 1), padding="valid", activation='sigmoid')


    def call(self, inputs):
        # block 1
        x1 = self.conv1(inputs)
        x1 = self.conv2(x1)
        x1_maxpool = self.maxpool1(x1)
        # block 2
        x2 = self.conv3(x1_maxpool)
        x2 = self.conv4(x2)
        x2_maxpool = self.maxpool2(x2)
        # block 3
        x3 = self.conv5(x2_maxpool)
        x3 = self.conv6(x3)
        x3_maxpool = self.maxpool3(x3)
        # block 4
        x4 = self.conv7(x3_maxpool)
        x4 = self.conv8(x4)
        x4_maxpool = self.maxpool4(x4)
        # block 5
        x5 = self.conv9(x4_maxpool)
        x5 = self.conv10(x5)
        x5_upconv = self.upconv1(x5)
        # crop 1
        size_x4 = x4.shape[1]
        size_up5 = x5_upconv.shape[1]
        crop_size1 = (size_x4-size_up5)//2 # crop center
        # block 6
        x6 = self.conv11(tf.concat([x5_upconv, x4[:, crop_size1:size_x4-crop_size1, crop_size1:size_x4-crop_size1, :]], axis=-1))
        x6 = self.conv12(x6)
        x6_upconv = self.upconv2(x6)
        # crop 2
        size_x3 = x3.shape[1]
        size_up6 = x6_upconv.shape[1]
        crop_size2 = (size_x3-size_up6)//2 # crop center
        # block 7
        x7 = self.conv13(tf.concat([x6_upconv, x3[:, crop_size2:size_x3-crop_size2, crop_size2:size_x3-crop_size2, :]], axis=-1))
        x7 = self.conv14(x7)
        x7_upconv = self.upconv3(x7)
        # crop 3
        size_x2 = x2.shape[1]
        size_up7 = x7_upconv.shape[1]
        crop_size3 = (size_x2-size_up7)//2 # crop center
        # block 8
        x8 = self.conv15(tf.concat([x7_upconv, x2[:, crop_size3:size_x2-crop_size3, crop_size3:size_x2-crop_size3, :]], axis=-1))
        x8 = self.conv16(x8)
        x8_upconv = self.upconv4(x8)
        # crop 4
        size_x1 = x1.shape[1]
        size_up8 = x8_upconv.shape[1]
        crop_size4 = (size_x1-size_up8)//2 # crop center
        # block 9
        x8 = self.conv17(tf.concat([x8_upconv, x1[:, crop_size4:size_x1-crop_size4, crop_size4:size_x1-crop_size4, :]], axis=-1))
        x8 = self.conv18(x8)
        output = self.out(x8)

        return output
    
import tensorflow as tf
from tensorflow.keras.layers import *

class Unet_2(tf.keras.Model):
    def __init__(self, img_size, in_channels, out_channels):
        super().__init__()

        # block 1
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer = 'he_normal', input_shape=(img_size, img_size, in_channels))
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # block 2
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # block 3
        self.conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.conv6 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # block 4
        self.conv7 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.conv8 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.maxpool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")

        # block 5
        self.conv9 = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.conv10 = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.upconv1 = Conv2DTranspose(filters=512, kernel_size=(2, 2), strides=(2, 2), padding="same", kernel_initializer = 'he_normal', activation='relu')

        # block 6
        self.conv11 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.conv12 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.upconv2 = Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=(2, 2), padding="same", kernel_initializer = 'he_normal', activation='relu')

        # block 7
        self.conv13 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.conv14 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.upconv3 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding="same", kernel_initializer = 'he_normal', activation='relu')

        # block 8
        self.conv15 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.conv16 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.upconv4 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding="same", kernel_initializer = 'he_normal', activation='relu')

        # block 9
        self.conv17 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.conv18 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer = 'he_normal', activation='relu')
        self.out = Conv2D(filters=out_channels, kernel_size=(1, 1), strides=(1, 1), padding="same", activation='sigmoid')


    def call(self, inputs):
        # block 1
        x1 = self.conv1(inputs)
        x1 = self.conv2(x1)
        x1_maxpool = self.maxpool1(x1)

        # block 2
        x2 = self.conv3(x1_maxpool)
        x2 = self.conv4(x2)
        x2_maxpool = self.maxpool2(x2)

        # block 3
        x3 = self.conv5(x2_maxpool)
        x3 = self.conv6(x3)
        x3_maxpool = self.maxpool3(x3)

        # block 4
        x4 = self.conv7(x3_maxpool)
        x4 = self.conv8(x4)
        x4_maxpool = self.maxpool4(x4)

        # block 5
        x5 = self.conv9(x4_maxpool)
        x5 = self.conv10(x5)
        x5_upconv = self.upconv1(x5)

        # block 6
        x6 = self.conv11(tf.concat([x5_upconv, x4], axis=-1))
        x6 = self.conv12(x6)
        x6_upconv = self.upconv2(x6)

        # block 7
        x7 = self.conv13(tf.concat([x6_upconv, x3], axis=-1))
        x7 = self.conv14(x7)
        x7_upconv = self.upconv3(x7)

        # block 8
        x8 = self.conv15(tf.concat([x7_upconv, x2], axis=-1))
        x8 = self.conv16(x8)
        x8_upconv = self.upconv4(x8)
 
        # block 9
        x8 = self.conv17(tf.concat([x8_upconv, x1], axis=-1))
        x8 = self.conv18(x8)
        output = self.out(x8)

        return output
    



