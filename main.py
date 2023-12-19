from data import Data
import sklearn as sk
from draw import Draw
from model import my_decision_tree, my_knn, my_svm

if __name__ == '__main__':
    data = Data()
    draw = Draw()

    # 原始数据的绘制
    draw.draw_scatter(data.data_x, data.data_y, title="原始数据降维表视图", savename="原始数据.png")

    # 采样后的数据的绘制
    draw.draw_scatter(data.data_XSampled, data.data_YSampled, title="数据(采样后)降维表示图", savename="采样后的数据.png")

    # 分割训练集，测试集
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(data.data_XSampled, data.data_YSampled,
                                                                           test_size=0.3, random_state=1)

    # 1.训练决策树
    tree = my_decision_tree()
    tree.train(x_train, y_train)
    y_hat = tree.predict(x_test)

    # 1.1 决策树的准确率召回率,与方差计算
    print("决策树的准确率为：", sk.metrics.accuracy_score(y_test, y_hat))
    print("决策树的召回率为：", sk.metrics.recall_score(y_test, y_hat, pos_label='1'))
    print("决策树的方差为：", sk.metrics.precision_score(y_test, y_hat, pos_label='1'))

    # 1.2 决策树的可视化
    # 画出树的形状
    tree.present(data.feature_names)

    # 重新获取原始数据集
    x_train, x_test, y_train, y_test = \
        sk.model_selection.train_test_split(data.data_XSampled, data.data_YSampled,
                                            test_size=0.3, random_state=1)

    # 2.训练KNN
    knn = my_knn()
    knn.train(x_train, y_train)
    y_hat = knn.predict(x_test)

    # 2.1KNN的准确率，召回率与方差计算
    y_test = y_test.astype(float)
    print("KNN的准确率为：", sk.metrics.accuracy_score(y_test, y_hat))
    print("KNN的召回率为：", sk.metrics.recall_score(y_test, y_hat, pos_label=1.0))
    print("KNN的方差为：", sk.metrics.precision_score(y_test, y_hat, pos_label=1.0))

    # 2.2KNN算法可视化
    knn.present()

    # 重新获取原始数据集
    x_train, x_test, y_train, y_test = \
        sk.model_selection.train_test_split(data.data_XSampled, data.data_YSampled,
                                            test_size=0.3, random_state=1)

    # 3.训练SVM
    # 训练不同核函数下的SVM
    kernels = ['poly', 'rbf', 'sigmoid', 'linear']
    for kernel in kernels:
        svm = my_svm(kernel=kernel)
        svm.train(x_train, y_train)
        y_hat = svm.predict(x_test)

        # 3.1 SVM的准确率，召回率与方差计算
        y_test = y_test.astype(float)
        print("SVM " + kernel + " 的准确率为：", sk.metrics.accuracy_score(y_test, y_hat))
        print("SVM " + kernel + "的召回率为：", sk.metrics.recall_score(y_test, y_hat, pos_label=1.0))
        print("SVM " + kernel + "的方差为：", sk.metrics.precision_score(y_test, y_hat, pos_label=1.0))

        # 3.2 SVM算法可视化
        svm.present()

        # 重新获取原始数据集
        x_train, x_test, y_train, y_test = \
            sk.model_selection.train_test_split(data.data_XSampled, data.data_YSampled,
                                                test_size=0.3, random_state=1)
