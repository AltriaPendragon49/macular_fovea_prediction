yolo用法：
  train中：
    1.先定义转换函数：将标注框、图像信息转化为yolo格式[0, x_center, y_center, width, height] # 0表示Fovea类别
    2.定义准备函数：
        2.1创建dataset,将数据集中的图片保存在dataset\imgaes\train中，将标注框信息保存在dataset\labels\train中
        2.2创建dataset.yaml文件，定义数据集路径、训练集、验证集、类别数、类别名称
    3.定义训练函数：
        3.1设置权重目录、输出目录
        3.2下载或加载预训练模型
        3.3设置训练参数
        3.4训练模型
        3.5保存模型
  test中：
    1.定义预测函数：
        1.1设置模型路径
        1.2加载模型
        1.3设置测试图像目录
        1.4处理测试图像
        1.5保存预测结果

