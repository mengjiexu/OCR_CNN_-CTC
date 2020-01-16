import pickle
import gc
import cv2
import numpy as np
import glob


def load_train_set(my_seq_len=89, label_max=1, max_label_width=28,
                   output_length=200000, img_list='data/img_list.csv'):
    # load dataset
    cut_zhuanzhe = lambda m: m.split('\n')[0]
    file_lst = list(map(cut_zhuanzhe, open(img_list).readlines()))
    np.random.shuffle(file_lst)
    start_idx = 0
    while True:
        end_idx = min(len(file_lst), start_idx + output_length)
        image_arr = []
        label_arr = []
        idx = 0
        seq_len_arr = []
        for f in file_lst[start_idx:end_idx]:
            image = cv2.imread(f, 0)
        #     image = np.array(image[:,:500,:])
            label = list(map(int, f.split('/')[-1].split('_')[1].split('-')))
            while len(label) < max_label_width:
                label.append(label_max)
            image_arr.append(image)
            label_arr.append(label)
            seq_len_arr.append(my_seq_len)
            idx += 1
        start_idx +=  output_length
        if start_idx >= len(file_lst) - 10000:
            start_idx = 0
            np.random.shuffle(file_lst)

        gc.collect()
        label_targets = [np.asarray(i) for i in label_arr]
        yield image_arr, label_targets, seq_len_arr


if __name__ == '__main__':
    train_data = load_train_set(192, 3000)
    print(next(train_data))