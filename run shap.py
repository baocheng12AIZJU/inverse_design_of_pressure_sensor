import shap
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import linregress
from matplotlib.ticker import MaxNLocator
from packages_own.models import Regressor, expotionalp,calculate_linear_properties_array
from packages_own.ensemble_models import EnsembleRegressor
from packages_own.shap import plot_shap_summary,plot_shap_dependence
from packages_own.Navigation import Navigation
def predict_with_models(input_tensor):
    """
    Predict using the ensemble of models and calculate linear properties.
    """
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

    avg_output = ensemble.predict(input_tensor).detach().numpy()
    a, b, c = avg_output.T
    x_linear_plot = np.linspace(0.3, 150, 25)
    r_squared,slopes = calculate_linear_properties_array(x_linear_plot, avg_output, expotionalp)

    return slopes
# Initialize Navigation and Ensemble Models
navigator = Navigation()
ensemble = EnsembleRegressor(num_models=5, model_class=Regressor)
ensemble.load_model_weights(loop=9)

# Load base and target data
index = np.arange(navigator.param_candidate.shape[0])
np.random.seed(42)
np.random.shuffle(index)
base_data = navigator.param_candidate[index]
target_data = np.load('./params/material_params/8_selected_params.npy')

# SHAP Kernel Explainer setup
background_data_kmeans = shap.kmeans(base_data[:5000], 100)  # Use KMeans for background selection
explainer = shap.KernelExplainer(predict_with_models, background_data_kmeans)

# Calculate SHAP values
shap_values = explainer.shap_values(target_data)

mask = (shap_values >= -0.05) & (shap_values <= 0.05)#150KPA 0.05,others,kpa.0.5
valid_samples_mask = mask.all(axis=1)
filtered_shap_values = shap_values[valid_samples_mask]
filtered_target_data = target_data[valid_samples_mask]
# Execute SHAP plotting
feature_labels = np.array(['Crosslinker', 'Height', 'Side length', 'Density'])
normalized_feature_labels = np.array(['Crosslinker (Normalized)', 'Height (Normalized)', 'Side length (Normalized)', 'Density (Normalized)'])
# Summary Plot
plot_shap_summary(
    shap_values=filtered_shap_values,
    target_data=filtered_target_data,
    feature_names=feature_labels,
    output_path='data_cache/plot/summary.png'
)

# Dependence Plots
plot_shap_dependence(
    shap_values=filtered_shap_values,
    target_data=filtered_target_data,
    feature_names=feature_labels,
    normalized_feature_names=normalized_feature_labels,
    output_path_prefix='data_cache/plot'
)
predictions = predict_with_models(target_data)
