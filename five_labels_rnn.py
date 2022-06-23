import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from all_labels_download import download_data_as_dataframe
from sklearn import preprocessing


# path = ...
df1 = download_data_as_dataframe(path, set_no='both', subject=4)
df2 = download_data_as_dataframe(path, set_no='side', subject=4)
df = pd.concat([df1, df2], axis=0)

values = df.values.astype('int32')
max_row = df.shape[0]
max_row -= max_row % -100
values = values[0:max_row]

X, y = values[:, :-1], values[:, -1]
X = preprocessing.scale(X)

comp = 0.99
pca = PCA(n_components=comp)
X = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print("shapes before reshape")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

max_row_train = X_train.shape[0]
PCA_features = X_train.shape[1]
max_row_test = X_test.shape[0]

X_test1 = X_test[:int(max_row_test/2), ]
X_test2 = X_test[int(max_row_test/2):, ]
y_test1 = y_test[:int(max_row_test/2)]
y_test2 = y_test[int(max_row_test/2):]
print(X_test2.shape, X_test1.shape, y_test2.shape, y_test1.shape)

X_train = X_train.reshape((max_row_train, 1, PCA_features))
X_test1 = X_test1.reshape((int(max_row_test/2), 1, PCA_features))
X_test2 = X_test2.reshape((int(max_row_test/2), 1, PCA_features))
print("shapes after reshape")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(264, activation='softmax', input_shape=(1, PCA_features)))
model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
csv_logger = tf.keras.callbacks.CSVLogger('training.log')
model.fit(X_train, y_train, epochs=30, verbose=1, validation_data=(X_test1, y_test1), callbacks=[csv_logger])
loss, accu = model.evaluate(X_test2, y_test2, verbose=1)
print('Accuracy: %.3f' % accu)

# tf.keras.utils.plot_model(model, to_file='model.pdf', show_layer_activations=True, show_shapes=True, rankdir='LR')
logs_table = pd.read_table("training.log", sep=',', header=0)
plt.plot(logs_table['epoch'], logs_table['accuracy'], label="accuracy")
plt.plot(logs_table['epoch'], logs_table['val_accuracy'], label="validation accuracy")
plt.title('Accuracy, subject 4., raw data')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# plt.plot(logs_table['epoch'], logs_table['loss'], label="loss")
# plt.plot(logs_table['epoch'], logs_table['val_loss'], label="validation loss")
# plt.title('Loss, subject 4., raw data')
# plt.xlabel("Epoch")
# plt.ylabel("Loss value")
# plt.legend()
