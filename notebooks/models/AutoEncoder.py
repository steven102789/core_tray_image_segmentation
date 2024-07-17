from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.optimizers import Adam

def MLP_AutoEncoder(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):
    # 输入层
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x = Flatten()(inputs)
    
    # 编码器
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # 中间的code神经元
    code = Dense(3, activation='relu')(x)
    
    # 解码器
    x = Dense(128, activation='relu')(code)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # 恢复到图像形状
    x = Dense(IMG_HEIGHT * IMG_WIDTH * n_classes, activation='relu')(x)
    x = Reshape((IMG_HEIGHT, IMG_WIDTH, n_classes))(x)
    
    # 输出层
    outputs = Dense(n_classes, activation='softmax')(x)
    
    # 构建模型
    model = Model(inputs, outputs)

    return model