import numpy as np
import os
import pickle
from xml.etree import ElementTree


class PascalVocLoader(object):

    def __init__(self, dir_path, labels):
        self._dir_path = dir_path
        self._labels = labels
        self._num_labels = len(labels)
        self._data = dict()

    @property
    def dir_path(self):
        return self._dir_path

    @property
    def labels(self):
        return self._labels

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def data(self):
        return self._data

    def reset_data(self):
        self._data = dict()

    def load(self, dump=False):
        filenames = os.listdir(self.dir_path)
        for filename in filenames:
            if not filename[-4:] == '.xml':
                continue

            bounding_boxes = []
            one_hot_classes = []

            # parse xml
            tree = ElementTree.parse(os.path.join(self.dir_path, filename))
            root = tree.getroot()
            size_tree = root.find('size')

            # get image size
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)

            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)/width
                    ymin = float(bounding_box.find('ymin').text)/height
                    xmax = float(bounding_box.find('xmax').text)/width
                    ymax = float(bounding_box.find('ymax').text)/height
                bounding_box = [xmin, ymin, xmax, ymax]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
                one_hot_class = self._to_one_hot(class_name)
                one_hot_classes.append(one_hot_class)

            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))

            # add image label
            self._data[image_name] = image_data

        if dump:
            with open(dump, 'wb') as f:
                pickle.dump(self.data, f)

    def _to_one_hot(self, name):
        one_hot_vector = [0] * self.num_labels
        if name in self.labels:
            index = self.labels.index(name)
            one_hot_vector[index] = 1
        else:
            print('unknown label: %s' % name)

        return one_hot_vector
