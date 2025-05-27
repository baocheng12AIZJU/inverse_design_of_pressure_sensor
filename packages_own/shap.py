import shap
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import linregress
from matplotlib.ticker import MaxNLocator
from packages_own.models import Regressor, expotionalp,calculate_linear_properties_array
from packages_own.ensemble_models import EnsembleRegressor
from packages_own.Navigation import Navigation

# Initialize Navigation and Ensemble Models
navigator = Navigation()
ensemble = EnsembleRegressor(num_models=5, model_class=Regressor)
ensemble.load_model_weights(loop=9)




# Plot SHAP summary
def plot_shap_summary(shap_values, target_data, feature_names, output_path):
    """
    Plot the SHAP summary plot.
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 18
    shap.plots._labels.labels['FEATURE_VALUE_LOW'] = '0'
    shap.plots._labels.labels['FEATURE_VALUE_HIGH'] = '1'

    shap.summary_plot(shap_values, target_data, feature_names=feature_names, show=False, sort=True, color_bar_label='')
    ax = plt.gca()
    ax.set_xlabel('')
    fig = plt.gcf()
    fig.set_size_inches(2.5, 2.5)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    plt.savefig(output_path, bbox_inches='tight', transparent=True, dpi=600)
    plt.show(block=True)
    plt.close()

# Plot SHAP dependence
def plot_shap_dependence(shap_values, target_data, feature_names, normalized_feature_names, output_path_prefix):
    """
    Plot SHAP dependence plots for each feature.
    """
    for i, feature in enumerate(feature_names):
        shap.dependence_plot(i, shap_values, target_data, feature_names=normalized_feature_names,
                             interaction_index="auto", show=False)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylabel('')

        fig = plt.gcf()
        fig.set_size_inches(3.5 / 1.4, 2.5 / 1.4)
        plt.savefig(f'{output_path_prefix}/dependence_{feature}.png', bbox_inches='tight', transparent=True, dpi=600)
        plt.show(block=True)
        plt.close()


