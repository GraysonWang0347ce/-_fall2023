import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline


# 从文件中读取数据，使用过采样与下采样解决数据不平衡问题
class Data:
    def __init__(self):
        self.dataset = np.loadtxt('yeast-2_vs_8.csv', delimiter=",", dtype=str)

        # 从数据集中分离出数据与标签
        self.data_x = self.dataset[1:, :-1]
        self.data_y = self.dataset[1:, -1:]  # 1:462 -1:20

        # 将data_y 转化为一维数组
        self.data_y = np.reshape(self.data_y, (len(self.data_y),))

        # 标签
        self.feature_names = self.dataset[0:1, :]
        # 将标签展开为一维list
        self.feature_names = np.reshape(self.feature_names, (1, -1)).tolist()[0]

        # 将1类数据下采样，-1类数据上采样，使得二者均为160个，保存在XSampled,YSampled内
        pipeline = make_pipeline(RandomUnderSampler(sampling_strategy=0.25),
                                 RandomOverSampler(sampling_strategy="auto"))
        self.data_XSampled, self.data_YSampled = pipeline.fit_resample(self.data_x, self.data_y)

        # print(np.shape(self.data_XSampled))
