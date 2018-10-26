from autokeras.image_supervised import load_image_dataset
from autokeras import ImageClassifier
from keras.datasets import mnist


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# print(x_train.shape, x_train[0].shape, x_train.reshape(x_train.shape + (1,)).shape)

x_train, y_train = load_image_dataset(csv_file_path="train.csv",
                                      images_path="images")
# print('x_train', x_train[0].shape, type(x_train))

# x_train = x_train.reshape(x_train.shape[0], 396, 532, 4)

print('x_train', x_train.shape)
print('y_train', y_train.shape)

x_test, y_test = load_image_dataset(csv_file_path="test.csv",
                                    images_path="images")
print(x_test.shape)
print(y_test.shape)

clf = ImageClassifier(verbose=True, augment=False)
clf.fit(x_train, y_train)
clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
y = clf.evaluate(x_test, y_test)
print(y)