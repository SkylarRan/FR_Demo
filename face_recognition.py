from PIL import Image
from numpy import expand_dims
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from count_time import execute_time
from dataset import extract_face, img_to_encoding, create_model, Dataset


class SVM:
    def __init__(self):
        self.model = None
        self.out_encoder = None

    def build_model(self, dataset):
        self.model = SVC(kernel='linear', probability=True)
        # 需要将每个名人姓名的字符串目标变量转换为整数。
        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(dataset.y_train)

    def train(self, dataset):
        # 首先，对人脸嵌入向量进行归一化是一种很好的做法。这是一个很好的实践，因为向量通常使用距离度量进行比较。
        in_encoder = Normalizer(norm='l2')
        X_train = in_encoder.transform(dataset.X_train)
        X_test = in_encoder.transform(dataset.X_test)

        y_train = self.out_encoder.transform(dataset.y_train)
        y_test = self.out_encoder.transform(dataset.y_test)

        self.model.fit(X_train, y_train)

        # 使用拟合模型对训练和测试数据集中的每个样本进行预测，然后计算分类正确率来实现。
        # predict
        yhat_train = self.model.predict(X_train)
        yhat_test = self.model.predict(X_test)

        # score
        score_train = accuracy_score(y_train, yhat_train)
        score_test = accuracy_score(y_test, yhat_test)

        # summarize
        print('Accuracy: train={:.3f}, test={:.3f}'.format(score_train * 100, score_test * 100))

    def save_model(self, file_path):
        joblib.dump((self.model, self.out_encoder), file_path)

    def load_model(self, file_path):
        self.model, self.out_encoder = joblib.load(file_path)

    def predict(self, image, model):
        image = extract_face(image)
        image = expand_dims(image, axis=0)
        image_embedding = img_to_encoding(image, model)
        pre_class = self.model.predict(image_embedding)
        probability = self.model.predict_proba(image_embedding)
        print("pre_class: {}".format(pre_class))
        print(self.out_encoder.classes_)
        print("probability: {}".format(probability))
        class_index = pre_class[0]
        probability = probability[0, class_index] * 100
        pre_name = self.out_encoder.inverse_transform(pre_class)
        print('Predicted: {} {:.3f}'.format(pre_name[0], probability))
        return pre_name[0], probability


@ execute_time
def train_model():
    facenet = create_model()
    print("加载数据")
    dataset = Dataset("face-dataset/")
    dataset.load(facenet)

    svm = SVM()
    svm.build_model(dataset)
    svm.train(dataset)
    svm.save_model("svm_classifier.model")
    print("save model successfully!")


@ execute_time
def recognize_pic(pic_file):
        # 使用保存的模型
        facenet = create_model()
        svm = SVM()
        svm.load_model("svm_classifier.model")
        print("load model successfully!")

        pre_name, probability = svm.predict(pic_file, facenet)
        plt.imshow(Image.open(pic_file))
        title = '{} ({:.3f})'.format(pre_name, probability)
        plt.title(title)
        plt.show()
        return pre_name, probability


if __name__ == '__main__':
    # train_model()
    pre_name, probability = recognize_pic("test3.jpg")
