import tensorflow as tf
import cv2
import numpy as np
from dataset import load_train_set
import time
from utils import sparse_tuple_from
import pickle
from model import build_model


def random_cut(image, w=800, h=40):
    ih, iw = image.shape
    x_start = np.random.randint(0, iw - w)
    y_start = np.random.randint(0, ih - h)
    return image[y_start:y_start+h, x_start:x_start+w]


def do_report():
    test_feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
    dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
    print('acc:', 1-accuracy)
#     report_accuracy(dd, test_targets)
    # decoded_list = decode_sparse_tensor(dd)


def do_batch():
    feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
    b_cost, steps, _ = session.run([cost, global_step, optimizer], feed)
    if steps > 0 and steps % 1000 == 0:
        # do_report()
        save_path = saver.save(session, "models/ocr_fcn3.model", global_step=steps)
        print('save to', save_path)
    return b_cost, steps


if __name__ == '__main__':

    word2idx = pickle.load(open('data/word2idx3.pkl', 'rb'))
    idx2word = pickle.load(open('data/idx2word3.pkl', 'rb'))
    temp_arr = []
    for key in word2idx:
        temp_arr.append(word2idx[key])
    label_max = max(temp_arr)
    label_max += 1
    train_iter = load_train_set(192, label_max=label_max)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    inputs, targets, seq_len, cost, optimizer, acc, decoded, global_step, learning_rate, log_prob = \
        build_model(num_classes=max(temp_arr)+2)
    # Initializate the weights and biases
    init = tf.global_variables_initializer()
    session.run(init)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    img = cv2.imread('images_result/test.jpg', 0)
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
    img3[5:35, :] = img2
    print('shape:', img3.shape)

    batch_size = 1
    num_epochs = 10000
    # saver.restore(session, './models/ocr_fcn3.model-845000')
    print('start')
    for curr_epoch in range(num_epochs):
        #     if curr_epoch % 10 == 1 and batch_size < 32:
        #         batch_size = int(batch_size * 2) + 2
        #     else:
        #         batch_size = int(batch_size / 8) + 2
        train_arr, train_labels, train_seq_arr = next(train_iter)
        print("Epoch.......", curr_epoch)
        print(len(train_arr))
        train_cost = train_ler = 0
        for batch in range(int(len(train_arr) / batch_size) - 1):
            start = time.time()
            train_inputs = np.array(list(map(random_cut, train_arr[batch * batch_size:(batch + 1) * batch_size])))[:, :, :,
                           np.newaxis]
            train_targets = sparse_tuple_from(train_labels[batch * batch_size:(batch + 1) * batch_size])
            train_seq_len = train_seq_arr[batch * batch_size:(batch + 1) * batch_size]
            # print("get data time", time.time() - start)
            start = time.time()
            c, steps = do_batch()
            train_cost += c * batch_size
            seconds = time.time() - start
            if batch % 100 == 0:
                print("Step:", steps, ", batch seconds:", seconds, ", loss:", c)

            try:
                if c < 0.001:
                    print(c)
                    break;
            except:
                print(c)
                break;
        decoded_ = session.run(decoded, feed_dict={inputs: np.array([img3[:, :, np.newaxis]]), seq_len: np.array([280])})
        output = []  #
        for v in decoded_[0][1]:
            output.append(idx2word[v])
        print(''.join(output))
