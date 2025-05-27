import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from packages_own.models import Regressor, expotionalp, calculate_linear_properties
from packages_own.ensemble_models import EnsembleRegressor
from packages_own.Navigation import Navigation
import csv
import matplotlib.pyplot as plt
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt

# Initialize system and models

class MultiObjectiveProblem(Problem):
    def __init__(self, pressure_range, ensemble, svm_model, sensitivity_limit, linearity_limit):
        super().__init__(
            n_var=4,  # Four optimization variables
            n_obj=2,  # Two optimization objectives
            n_constr=3,  # One constraint (SVM prediction success rate >= 0.8)
            xl=[0.0, 0.0, 0.0, 0.0],  # Lower bounds
            xu=[1, 1, 1, 1]  # Upper bounds (normalized)
        )
        self.history = []  # Store optimization history
        self.pressure_range = pressure_range
        self.ensemble = ensemble
        self.svm_model = svm_model
        self.sensitivity_limit = sensitivity_limit
        self.linearity_limit = linearity_limit

    def _evaluate(self, X, out, *args, **kwargs):
        F, G = [], []
        for params in X:
            # Convert parameters to Tensor and predict
            input_tensor = torch.tensor(params).unsqueeze(0).float()
            avg_output = self.ensemble.predict(input_tensor)
            output_np = avg_output.squeeze().detach().numpy()

            # Calculate linear properties
            r_squared, slope = calculate_linear_properties(self.pressure_range, output_np, expotionalp)
            parms_denormalized_height = params[1] * (0.9 - 0.1) + 0.1
            parms_denormalized_side_length = params[2] * (0.9 - 0.1) + 0.1

            # Use SVM model to predict success rate
            svm_prediction = self.svm_model.predict_proba([[parms_denormalized_height, parms_denormalized_side_length]])[0][1]  # Assuming second class is success
            svm_success_rate = svm_prediction

            # Record optimization history
            self.history.append((params.copy(), r_squared, slope, svm_success_rate))

            # Optimization objectives (maximize R² and Slope, converted to minimization)
            F.append([-r_squared, -slope])

            # Constraints (Slope >= 0 and SVM prediction success rate >= 0.8)
            G.append([0.8 - svm_success_rate, self.sensitivity_limit-slope, self.linearity_limit-r_squared])  # Converted to g(x) <= 0 form

        out["F"] = np.array(F)
        out["G"] = np.array(G)

def select_uniform_points(F, n_select=6):
    # 1. Normalize
    F_normalized = (F - np.min(F, axis=0)) / (np.max(F, axis=0) - np.min(F, axis=0))

    # 2. Sort by first objective (sensitivity)
    sort_idx = np.argsort(F_normalized[:, 0])
    F_sorted = F_normalized[sort_idx]

    # 3. Calculate cumulative arc length
    diffs = np.diff(F_sorted, axis=0)  # Calculate differences between adjacent points
    segment_distances = np.linalg.norm(diffs, axis=1)  # Calculate Euclidean distance for each segment
    cumulative_distances = np.insert(np.cumsum(segment_distances), 0, 0)  # Calculate cumulative distance
    total_length = cumulative_distances[-1]  # Total arc length

    # 4. Generate target positions
    target_positions = np.linspace(0, total_length, n_select)

    # 5. Find nearest neighbor indices
    selected_indices = []
    for pos in target_positions:
        idx = np.abs(cumulative_distances - pos).argmin()  # Find nearest neighbor index
        selected_indices.append(idx)

    # Remove duplicates and maintain order
    selected_indices = sorted(np.unique(selected_indices))

    # 6. Return original sorted indices
    original_indices = sort_idx[selected_indices]
    return original_indices