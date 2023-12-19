from sklearn import tree
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']


class model:
    def __init__(self):
        pass

    def train(self, x, y):
        pass


class my_decision_tree(model):
    def __init__(self):
        super(my_decision_tree, self).__init__()
        pass

    def train(self, x, y):
        self.clf = tree.DecisionTreeClassifier()
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)

    def present(self, feature_names):
        plt.figure(figsize=(12, 8))
        plot_tree(self.clf, filled=True, feature_names=feature_names,
                  class_names=['1', '-1'], rounded=True)

        plt.title("所训练决策树图示")
        # plt.show()
        plt.savefig("决策树图示.png")


class my_knn(model):
    def __init__(self):
        super(my_knn, self).__init__()
        pass

    def train(self, x, y):
        self._x = x
        self._y = y

        # 将x，y内所有元素转换为float
        self._x = self._x.astype(float)
        self._y = self._y.astype(float)

        self.clf = KNeighborsClassifier(n_neighbors=3)
        self.clf.fit(self._x, self._y)

    def predict(self, x):
        x = x.astype(float)
        return self.clf.predict(x)

    def present(self):
        plt.figure(figsize=(10, 8))

        # 生成网格数据来绘制决策边界
        x_min, x_max = self._x[:, 0].min() - 1, self._x[:, 0].max() + 1
        y_min, y_max = self._x[:, 1].min() - 1, self._x[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel()),
                                   np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel()),
                                   np.zeros_like(xx.ravel())])
        Z = Z.reshape(xx.shape)

        # 绘制决策边界
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

        # 绘制训练样本
        scatter = plt.scatter(self._x[:, 0], self._x[:, 1], c=self._y, cmap=plt.cm.plasma, edgecolors='k')

        plt.title("所训练KNN图示")
        # plt.show()
        plt.savefig("KNN图示.png")


class my_svm(model):
    def __init__(self, kernel):
        super(my_svm, self).__init__()
        self.kernel = kernel

    def train(self, x, y):
        self._x = x
        self._y = y

        # 将x，y内所有元素转换为float
        self._x = self._x.astype(float)
        self._y = self._y.astype(float)

        self.clf = SVC(kernel=self.kernel)
        self.clf.fit(self._x, self._y)

    def predict(self, x):
        x = x.astype(float)
        return self.clf.predict(x)

    def present(self):
        plt.figure(figsize=(10, 8))

        # 选择前两个维度进行可视化
        feature1_index, feature2_index = 0, 1

        # 生成网格数据来绘制决策边界
        x_min, x_max = self._x[:, feature1_index].min() - 1, self._x[:, feature1_index].max() + 1
        y_min, y_max = self._x[:, feature2_index].min() - 1, self._x[:, feature2_index].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel()),
                                   np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel()),
                                   np.zeros_like(xx.ravel())])
        Z = Z.reshape(xx.shape)

        # 绘制决策边界和间隔
        plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

        # 绘制训练样本
        scatter = plt.scatter(self._x[:, feature1_index], self._x[:, feature2_index],
                              c=self._y, cmap=plt.cm.plasma,
                              edgecolors='k')

        plt.title("所训练SVM图示" + "(kernel=" + self.kernel + ")")
        plt.xlabel(f"Feature {feature1_index + 1}")
        plt.ylabel(f"Feature {feature2_index + 1}")

        plt.savefig("SVM图示" + self.kernel + ".png")
        # plt.show()
