import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data_dict = pickle.load(open('data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

num_classes = len(np.unique(encoded_labels))
y_onehot = to_categorical(encoded_labels, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(
    data, y_onehot, test_size=0.2, shuffle=True, stratify=encoded_labels
)

x_train = x_train / np.max(x_train)
x_test = x_test / np.max(x_test)

input_shape = x_train.shape[1]  

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_shape,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')

model.save('deep_model.h5')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

label_mapping = {i: chr(65 + i) for i in range(9)} 

with open('deep_model_info.pkl', 'wb') as f:
    pickle.dump({
        'num_classes': num_classes,
        'label_mapping': label_mapping
    }, f)

print("Deep learning model saved successfully!") 