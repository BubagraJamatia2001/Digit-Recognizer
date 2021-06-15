import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

model = keras.models.load_model("model.h5")

for i in range(1, 6):
    img = cv2.imread(f"D:\\Jupyter Works\\Minor Project Digit Recognizer\\image samples\\{i}.png")[:, :, 0]
    img = np.invert(np.array([img]))
    plt.imshow(img[0], cmap=plt.cm.binary)
    predict = model.predict(img)
    print(f'The Prediction is : {np.argmax(predict)}')
    plt.show()


