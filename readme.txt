先打开LG-preprocess：程序会处理文本，并新建data文件夹用于存储
再打开LG-email-filter：第一次打开时，程序会初始化文档矩阵，并存入numpy_save.npz，再学习和测试
		    之后打开，程序会直接进入学习和测试