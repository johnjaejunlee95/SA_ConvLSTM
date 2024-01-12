import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset

class MovingMNIST(Dataset):

    def __init__(self, args, is_train, root, n_frames_input, n_frames_output, num_objects):

        super(MovingMNIST, self).__init__()
        
        path = root + '/train-images-idx3-ubyte.gz'
        with gzip.open(path, 'rb') as f:
            self.datas = np.frombuffer(f.read(), np.uint8, offset=16)
            self.datas = self.datas.reshape(-1, 28,28)

        if is_train is True:
            self.datas = self.datas[0 : 10000]
        else:
            self.datas = self.datas[10000 : 13000]

        self.image_size = (28,28)
        self.input_size = (args.img_size, args.img_size)
        self.step_length = 0.1
        self.num_objects = num_objects

        self.num_frames_input =n_frames_input
        self.num_frames_output = n_frames_output
        self.num_frames_total = self.num_frames_input + self.num_frames_output


    def _get_random_trajectory(self, seq_length):

        assert self.input_size[0] == self.input_size[1]
        assert self.image_size[0] == self.image_size[1]

        canvas_size = self.input_size[0] - self.image_size[0]

        x = random.random()
        y = random.random()

        theta = random.random() * 2 * np.pi

        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)

        for i in range(seq_length):

            y += v_y * self.step_length
            x += v_x * self.step_length

            if x <= 0.: x = 0.; v_x = -v_x;
            if x >= 1.: x = 1.; v_x = -v_x
            if y <= 0.: y = 0.; v_y = -v_y;
            if y >= 1.: y = 1.; v_y = -v_y

            start_y[i] = y
            start_x[i] = x

        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)

        return start_y, start_x

    def _generate_moving_mnist(self, num_digits=2):

        data = np.zeros((self.num_frames_total, *self.input_size), dtype=np.float32)

        for n in range(num_digits):

            start_y, start_x = self._get_random_trajectory(self.num_frames_total)
            ind = np.random.randint(0, self.__len__())
            digit_image = self.datas[ind]

            for i in range(self.num_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.image_size[0]
                right = left + self.image_size[1]
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]

        return data

    def __getitem__(self, item):

        num_digits = random.choice(self.num_objects)
        images = self._generate_moving_mnist(num_digits)

        inputs = torch.from_numpy(images[:self.num_frames_input]).permute(0, 3, 1, 2).contiguous()
        targets = torch.from_numpy(images[self.num_frames_output:]).permute(0, 3, 1, 2).contiguous()

        return inputs / 255., targets / 255.

    def __len__(self):
        return self.datas.shape[0]


class MovingMNIST_Test(Dataset):
    def __init__(self,  path, n_frames_input, n_frames_output):
        super(MovingMNIST_Test, self).__init__()

        self.num_frames_input =n_frames_input
        self.num_frames_output = n_frames_output
        self.num_frames_total = self.num_frames_input + self.num_frames_output

        self.dataset =  np.load(path + 'mnist_test_seq.npy')
        self.dataset = self.dataset[..., np.newaxis]
                
    def __getitem__(self, index):
        images =  self.dataset[:, index, ...]

        inputs = torch.from_numpy(images[:self.num_frames_input]).permute(0, 3, 1, 2).contiguous()
        targets = torch.from_numpy(images[self.num_frames_output:]).permute(0, 3, 1, 2).contiguous()

        return inputs / 255., targets / 255.

    def __len__(self):
        return len(self.dataset[1])


# class MovingMNIST(data.Dataset):
#     def __init__(self, root, is_train, n_frames_input, n_frames_output, num_objects,
#                  transform=None):
#         '''
#         param num_objects: a list of number of possible objects.
#         '''
#         super(MovingMNIST, self).__init__()

#         self.dataset = None
#         if is_train:
#             self.mnist = load_mnist(root)
#         else:
#             if num_objects[0] != 2:
#                 self.mnist = load_mnist(root)
#             else:
#                 self.dataset = load_fixed_set(root, False)
#         self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

#         self.is_train = is_train
#         self.num_objects = num_objects
#         self.n_frames_input = n_frames_input
#         self.n_frames_output = n_frames_output
#         self.n_frames_total = self.n_frames_input + self.n_frames_output
#         self.transform = transform
#         # For generating data
#         self.image_size_ = 64
#         self.digit_size_ = 28
#         self.step_length_ = 0.1

#     def get_random_trajectory(self, seq_length):
#         ''' Generate a random sequence of a MNIST digit '''
#         canvas_size = self.image_size_ - self.digit_size_
#         x = random.random()
#         y = random.random()
#         theta = random.random() * 2 * np.pi
#         v_y = np.sin(theta)
#         v_x = np.cos(theta)

#         start_y = np.zeros(seq_length)
#         start_x = np.zeros(seq_length)
#         for i in range(seq_length):
#             # Take a step along velocity.
#             y += v_y * self.step_length_
#             x += v_x * self.step_length_

#             # Bounce off edges.
#             if x <= 0:
#                 x = 0
#                 v_x = -v_x
#             if x >= 1.0:
#                 x = 1.0
#                 v_x = -v_x
#             if y <= 0:
#                 y = 0
#                 v_y = -v_y
#             if y >= 1.0:
#                 y = 1.0
#                 v_y = -v_y
#             start_y[i] = y
#             start_x[i] = x

#         # Scale to the size of the canvas.
#         start_y = (canvas_size * start_y).astype(np.int32)
#         start_x = (canvas_size * start_x).astype(np.int32)
#         return start_y, start_x

#     def generate_moving_mnist(self, num_digits=2):
#         '''
#         Get random trajectories for the digits and generate a video.
#         '''
#         data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
#         for n in range(num_digits):
#             # Trajectory
#             start_y, start_x = self.get_random_trajectory(self.n_frames_total)
#             ind = random.randint(0, self.mnist.shape[0] - 1)
#             digit_image = self.mnist[ind]
#             for i in range(self.n_frames_total):
#                 top = start_y[i]
#                 left = start_x[i]
#                 bottom = top + self.digit_size_
#                 right = left + self.digit_size_
#                 # Draw digit
#                 data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

#         data = data[..., np.newaxis]
#         return data

#     def __getitem__(self, idx):
#         length = self.n_frames_input + self.n_frames_output
#         if self.is_train or self.num_objects[0] != 2:
#             # Sample number of objects
#             num_digits = random.choice(self.num_objects)
#             # Generate data on the fly
#             images = self.generate_moving_mnist(num_digits)
#         else:
#             images = self.dataset[:, idx, ...]


#         r = 1
#         w = int(64 / r)
#         images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))

#         input = images[:self.n_frames_input]
#         if self.n_frames_output > 0:
#             output = images[self.n_frames_input:length]
#         else:
#             output = []

#         frozen = input[-1]

#         output = torch.from_numpy(output / 255.0).contiguous().float()
#         input = torch.from_numpy(input / 255.0).contiguous().float()

#         out = [idx, output, input, frozen, np.zeros(1)]
#         return out

#     def __len__(self):
#         return self.length