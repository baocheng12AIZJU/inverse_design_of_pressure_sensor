# Inverse_Design/Navigation.py
# For recommendation in the inverse_design_sensor loop

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import tqdm
import time
import shap
import random

class Navigation:
    '''Navigation achieving diversity in the input and output space (X) for initialization and incremental recommendation'''

    def __init__(self):
        # define the range of each parameter in the input space
        self.beta_range = [0.058, 0.161]  # mass fraction of the curing agent
        self.h_range = [0.1, 0.91]  # height of microstructures
        self.d_range = [0.1, 0.91]  # side length of microstructures
        self.rou_range = [0.1, 0.51]  # density of microstructures
        print('*' * 25, 'Initializing...', '*' * 25)
        # load the SVM model and parameters
        with open('./params/SVM_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        # filter d and h using the SVM classifier
        self.h_candidate = np.arange(self.h_range[0], self.h_range[1], 0.03)
        self.d_candidate = np.arange(self.d_range[0], self.d_range[1], 0.03)
        self.h_candidate, self.d_candidate = np.meshgrid(self.h_candidate, self.d_candidate)
        conf = self.model.predict_proba(np.c_[self.h_candidate.ravel(), self.d_candidate.ravel()])
        filter = conf[:, 1] > 0.8
        self.h_candidate = self.h_candidate.ravel()[filter]
        self.d_candidate = self.d_candidate.ravel()[filter]
        self.beta_candidate = np.arange(self.beta_range[0], self.beta_range[1], 0.006)
        self.rou_candidate = np.arange(self.rou_range[0], self.rou_range[1], 0.02)

        # define the input space
        self.param_candidate = []
        for i in tqdm.tqdm(self.beta_candidate):
            for j, k in zip(self.h_candidate, self.d_candidate):
                for l in self.rou_candidate:
                    self.param_candidate.append([i, j, k, l])
        self.param_candidate = np.asarray(self.param_candidate)
        # normalization
        self.param_min = np.min(self.param_candidate, axis=0, keepdims=True)
        self.param_max = np.max(self.param_candidate, axis=0, keepdims=True)
        self.param_candidate = (self.param_candidate - self.param_min) / (self.param_max - self.param_min)
        print('Shape of the parameter input space:', self.param_candidate.shape)
        print('*' * 25, 'Initialized.', '*' * 25)

        # define the container for selected parameters
        self.selected_params = []

    def initial_recommend(self, N = 10):
        print('Recommending point 1...')
        # chose the first input point
        central_point = np.mean(self.param_candidate, axis=0, keepdims=True)
        # find the point that is closest to the central point as the first point
        first_point = self.param_candidate[np.argmin(np.sqrt(np.sum((self.param_candidate - central_point)**2, axis=1)))]
        self.selected_params.append(first_point)
        # delete the first point from the input space
        delect_index = np.bincount(np.where(self.param_candidate == first_point)[0]).argmax()
        self.param_candidate = np.delete(self.param_candidate, delect_index, axis=0)
        n = 1
        while n < N:
            print(f'Recommending point {n+1}...')
            # calculate the distance between the selected point and the rest of the input space
            dist_arr = []
            for point in self.selected_params:
                dist = np.sqrt(np.sum((self.param_candidate - point)**2, axis=1))
                dist_arr.append(dist)
            dist_arr = np.asarray(dist_arr)
            dist_arr = np.min(dist_arr, axis=0)
            selected_point = self.param_candidate[np.argmax(dist_arr)]
            # delete the selected point from the input space
            delete_index = np.bincount(np.where(self.param_candidate == selected_point)[0]).argmax()
            self.param_candidate = np.delete(self.param_candidate, delete_index, axis=0)
            self.selected_params.append(selected_point)
            n += 1
        print('Recommendation finished.')
        print('*'*50)
        # inverse normalization
        outputs = np.asarray(self.selected_params) * (self.param_max-self.param_min) + self.param_min
        return outputs

    def incremental_recommend(self, model, N=10):
        for n in tqdm.trange(N):
            dist_arr = []
            for point in self.selected_params:
                # calculate the distance in the input space
                dist_X = np.sqrt(np.sum((self.param_candidate - point)**2, axis=1))
                # calculate the distance in the output space
                input_range = self.param_candidate
                input_point = point[np.newaxis, :]
                dist_Y = torch.sqrt(torch.sum((model.predict(torch.tensor(input_range).float()) -
                                               model.predict(torch.tensor(input_point).float()))**2, dim=1)).detach().numpy()
                dist = dist_Y*dist_X

                dist_arr.append(dist)
            dist_arr = np.asarray(dist_arr)
            dist_arr = np.min(dist_arr, axis=0)
            selected_param = self.param_candidate[np.argmax(dist_arr)]
            self.selected_params.append(selected_param)
            # update the input space --> remove the selected points
            delete_index = np.bincount(np.where(self.param_candidate == selected_param)[0]).argmax()
            self.param_candidate = np.delete(self.param_candidate, delete_index, axis=0)

        outputs = np.asarray(self.selected_params) * (self.param_max-self.param_min) + self.param_min
        return outputs[-N:]
if __name__ == '__main__':
    navigator = Navigation()



