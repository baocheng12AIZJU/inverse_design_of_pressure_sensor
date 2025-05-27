import numpy as np
from packages_own.Navigation import Navigation
from packages_own.models import Regressor
from packages_own.ensemble_models import EnsembleRegressor

# Initialize the navigation system for managing recommendations and selected parameters
navigator = Navigation()

# Initialize an ensemble model with 5 regressors
ensemble = EnsembleRegressor(num_models=5, model_class=Regressor)
# Specify the current loop index
current_loop = 2
if current_loop == 1:
    navigator.selected_params = []
    current_recom = navigator.initial_recommend(N=10)
else:
    navigator.selected_params = list(np.load(f'./params/material_params/{current_loop - 1}_selected_params.npy'))
    ensemble.load_model_weights(loop=current_loop - 1)#load model weights in the previous loop
    current_recom = navigator.incremental_recommend(ensemble, N=10)
# Print the current recommendations
print(current_recom)

# Save the current recommendations to a file
np.save(f'./params/material_params/{current_loop}_cur_recommendations.npy', current_recom)

# Save the updated selected parameters for the current loop to a file
np.save(f'./params/material_params/{current_loop}_selected_params.npy', np.asarray(navigator.selected_params))
