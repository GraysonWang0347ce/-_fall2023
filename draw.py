import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# 使matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']


class Draw:
    def __init__(self):
        pass

    def draw_scatter(self, x, y,
                     title='Figure',
                     x_label='维度0',
                     y_label='维度1',
                     savename='./figure.png'):

        colors = np.copy(y)
        colors = colors.transpose()

        colors[colors == '1'] = "blue"
        colors[colors == '-1'] = 'green'

        # print(colors)
        # print(x)

        # 使用PCA将x[160,8] 将为 x[160,2]
        pca = PCA(n_components=2)

        # X [160, 2] 用于2d 散点图的绘制
        X = pca.fit_transform(x)

        # 将X中数据全部转化至(0,1)间
        X = (X - np.min(X)) / (np.max(X) - np.min(X))

        legend_labels = []

        plt.scatter(X[:, 0], X[:, -1], c=colors,
                    s=20, marker="o", alpha=0.35, label="points")

        red, blue = 0, 0

        for i, color in enumerate(colors):
            sc = plt.scatter(X[i, 0], X[i, 1], c=[color], marker='o', s=30, alpha=0.25)
            if color == 'green' and not red:
                legend_labels.append(sc)
                red += 1
            elif color == 'blue' and not blue:
                legend_labels.append(sc)
                blue += 1

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # 显示图例
        plt.legend(handles=legend_labels,
                   labels=['1类数据' if i == 1 else '-1类数据' for i in range(2)])

        # plt.show()
        plt.savefig(savename, dpi=300, bbox_inches='tight')
