# pylint: skip-file
""" file iterator for pasval voc 2012"""
import mxnet as mx
import numpy as np
import sys, os
from mxnet.io import DataIter
from PIL import Image
import cv2
import random
import re
import math
import logging

class FileIter(DataIter):
    def __init__(self, traindata_dir,
                 bgr_mean = (117, 117, 117),
                 data_name = "data",
                 label_name = "l2_label"):
        super(FileIter, self).__init__()
        self.traindata_dir = traindata_dir
        self.mean = np.array(bgr_mean) # (B, G, R)
        self.data_name = data_name
        self.label_name = label_name

        # config
        self.batch_size = 32
        self.dst_size = 128
        self.label_dim = 96 * 2

        self.train_imgs = []
        self.train_labels = []
        pattern = re.compile(r'.*jpg$')
        for dirpath, dirnames, filenames in os.walk(self.traindata_dir):
            for filename in filenames:
                match = pattern.match(filename)
                if match:
                    self.train_imgs.append(os.path.join(dirpath, filename))
                    self.train_labels.append(os.path.join(dirpath, filename.replace('.jpg', '.96pt')))

        self.all_train_data = zip(self.train_imgs, self.train_labels)
        logging.info('all train data count %d', len(self.all_train_data))
        self.train_cursor = 0

        self.data, self.label = self._read()

    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
        data = {}
        label = {}
        data[self.data_name], label[self.label_name] = self._read_img()
        return list(data.items()), list(label.items())

    def _read_img(self):
        train_img = np.zeros((self.batch_size, 3, self.dst_size, self.dst_size), dtype=np.float32)
        train_label = np.zeros((self.batch_size, self.label_dim), dtype=np.float32)

        for i in range(self.batch_size):
            train_index = self.train_cursor + i
            img = cv2.imread(self.all_train_data[train_index][0])
            label = np.loadtxt(self.all_train_data[train_index][1], skiprows=1).astype(np.float32)
            box_label = np.expand_dims(label, axis=0)
            [rect_x, rect_y, rect_w, rect_h] = cv2.boundingRect(box_label)

            center_x = rect_x + rect_w / 2
            center_y = rect_y + rect_h / 2
            max_wh = max(rect_w, rect_h)
            expand_ratio = random.random() * 1 + 1.1
            ori_max_wh = max_wh
            max_wh = int(max_wh * expand_ratio)

            max_left = rect_x + rect_w * 1.2 - max_wh
            max_top = rect_y + rect_h * 1.2 - max_wh
            max_right = rect_x - rect_w * 0.2
            max_bottom = rect_y - rect_h * 0.2

            if max_right < max_left:
                new_x = center_x - max_wh / 2
            else:
                new_x = random.randint(int(max_left), int(max_right))
            if max_bottom < max_top:
                new_y = center_y - max_wh / 2
            else:
                new_y = random.randint(int(max_top), int(max_bottom))

            # new_x = center_x - max_wh / 2
            # new_y = center_y - max_wh / 2
            # new_x = random.randint(int(max_left), int(max_right))
            # new_y = random.randint(int(max_top), int(max_bottom))

            if new_x < 0:
                inc_left = -new_x
            else:
                inc_left = 0

            if new_y < 0:
                inc_top = -new_y
            else:
                inc_top = 0

            if new_x + max_wh > img.shape[1]:
                inc_right = new_x + max_wh - img.shape[1]
            else:
                inc_right = 0

            if new_y + max_wh > img.shape[0]:
                inc_bottom = new_y + max_wh - img.shape[0]
            else:
                inc_bottom = 0

            img = cv2.copyMakeBorder(img, inc_top, inc_bottom, inc_left, inc_right, cv2.BORDER_CONSTANT)

            if new_x < 0:
                new_x = 0
            if new_y < 0:
                new_y = 0

            # cv2.rectangle(img, (new_x, new_y), (new_x + max_wh, new_y + max_wh), (0, 255, 0), thickness=2)

            new_label = np.copy(label)
            new_label[:, 0] = new_label[:, 0] + inc_left
            new_label[:, 1] = new_label[:, 1] + inc_top

            # new_label = new_label.astype(np.int32)
            # for j in range(new_label.shape[0]):
            #     cv2.circle(img, (new_label[j, 0], new_label[j, 1]), 1, (255, 0, 0), 2)
            # cv2.imshow('img', img)
            # cv2.waitKey()

            img = img[new_y:new_y + max_wh, new_x:new_x + max_wh, :]
            scale_ratio = float(self.dst_size) / max_wh
            img = cv2.resize(img, (self.dst_size, self.dst_size))
            new_label[:, 0] = new_label[:, 0] - new_x
            new_label[:, 1] = new_label[:, 1] - new_y
            new_label = new_label * scale_ratio

            # new_label = new_label.astype(np.int32)
            # for j in range(new_label.shape[0]):
            #     cv2.circle(img, (new_label[j, 0], new_label[j, 1]), 1, (255, 0, 0), 1)
            # cv2.imshow('img', img)
            # cv2.waitKey()

            img = img.transpose((2, 0, 1))
            new_label = new_label / self.dst_size
            new_label = np.reshape(new_label.transpose(), (-1))
            train_img[i, :, :, :] = img
            train_label[i, :] = new_label

        # show train data
        # for i in range(train_img.shape[0]):
        #     img = train_img[i, :, :, :]
        #     img = img.transpose((1, 2, 0)).astype(np.uint8)
        #     img_clone = np.zeros(img.shape, dtype=np.uint8)
        #     img_clone[...] = img
        #     label = (np.reshape(train_label[i, :], (2, train_label.shape[1] / 2)) * self.dst_size).astype(np.int32)
        #     for j in range(label.shape[1]):
        #         cv2.circle(img_clone, (label[0, j], label[1, j]), 1, (255, 0, 0), 1)
        #     cv2.imshow('img', img_clone)
        #     cv2.waitKey()

        return (train_img, train_label)

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, v.shape) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, v.shape) for k, v in self.label]

    def get_batch_size(self):
        return self.batch_size

    def reset(self):
        self.train_cursor = 0
        random.shuffle(self.all_train_data)

    def iter_next(self):
        if self.train_cursor + self.batch_size > len(self.all_train_data):
            return False
        else:
            return True

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            self.train_cursor = self.train_cursor + self.batch_size
            return {self.data_name: self.data[0][1], self.label_name: self.label[0][1]}
        else:
            raise StopIteration
