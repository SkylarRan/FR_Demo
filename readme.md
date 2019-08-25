# face_recognition
- 采用 mtcnn 提取人脸区域
- 采用 facenet-keras 模型
- 采用 SVC 分类模型
- 样本集
    - train: 125 （每人至少3张，部分同事的训练样本在6+）
    - test: 25 
- 准确率
    - 训练样本多的同事，识别率大约在 0.3 ~ 0.65 之间
    - 只要3张训练样本的同事，识别率在 0.1 ~ 0.3 之间，且可能识别错误
    
- 使用
    - 直接进入face_recognition.py运行主程序
    - 若添加新数据则运行train_model()函数
    - 使用已保存的模型，则预测图片时运行recognize_pic()函数