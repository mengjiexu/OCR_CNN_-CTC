import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import pickle


idx2word = pickle.load(open('data/idx2word3.pkl', 'rb'))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
inputs, targets, seq_len, cost, optimizer, acc, decoded, global_step, learning_rate, log_prob = build_model()
saver = tf.train.Saver()
saver.restore(session, './models/ocr_fcn3.model-845000')
#%%
# 使用fcn模型进行预测
img = cv2.imread('./images_result/test.jpg')# )'../images/img_1946-1602-1553-1910-285_399816.jpg'
print(img.shape)
h, w, _ = img.shape
bili = 40 * 1.0 / h
aim_h = 40
aim_w = int(bili * w)
img2 = cv2.resize(img, (aim_w, aim_h))
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
img2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
plt.figure(figsize=(30, 10))
plt.imshow(img2)
plt.show()
print(img2.shape)
img2 = np.ones([40, 1275, 3], np.uint8) * 255
img2[3:36,:,:] = img
plt.figure(figsize=(30, 10))
plt.imshow(img2)
plt.show()
print(img2.shape)

img = cv2.imread('./images_result/test.jpg',0)# '../images/img_1946-1602-1553-1910-285_399816.jpg') # )'../images/img_1946-1602-1553-1910-285_399816.jpg'
print(img.shape)
h, w = img.shape
bili = 30 * 1.0 / h
aim_h = 30
aim_w = int(bili * w)
gray = img
ret, gray = cv2.threshold(gray, 126, 255, cv2.THRESH_BINARY)
img2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
img2 = cv2.resize(img, (aim_w, aim_h))
h, w = img2.shape
img3 = np.ones([40, w], np.uint8) * 255
img3[5:35,:] = img2
decoded_ = session.run(decoded, feed_dict={inputs: np.array([img3[:,:,np.newaxis]]),seq_len:np.array([280])})
output = [] #
for v in decoded_[0][1]:
    output.append(idx2word[v])
print(''.join(output))
print(decoded_[0])
plt.figure(figsize=(30, 10))
plt.imshow(img2)
plt.show()