import numpy as np
import csv
import torch
from packages_own.models import evaluate_model_each_loop_run, load_testdata, load_traindata,Regressor,evaluate_mre_points

# Define the loops to evaluate
# `evaluate_loops` specifies the data rounds to evaluate; in this case, we only evaluate round 1
# evaluate_loops = np.arange(1,10,1)
evaluate_loops = [9]

# Define the number of runs per loop
num_runs_per_loop = 5

# Load the test data and corresponding labels
test_data, test_labels = load_testdata()

# Initialize a list to store error metrics for each loop
errors = []

# Iterate over each evaluation loop
for loop_id in evaluate_loops:

    print(f"Starting evaluation for loop {loop_id}...")

    # Initialize lists to store errors for fitting and six-point evaluations for the current loop
    errors_list_fitting = []
    test_pre = []

    # Iterate over the runs in the current loop
    for run_id in range(num_runs_per_loop):
        # Load the training data for the current loop
        _, selected_recommendations, all_labels = load_traindata(loop_id)

        # Initialize the regressor model
        regressor = Regressor()

        # Load the saved model parameters for the specific loop and run
        regressor.load_state_dict(torch.load(f'./params/NN_params/regressor_{loop_id}-{run_id}.pth'))

        # Evaluate the model for the current loop and run
        result = evaluate_model_each_loop_run(
            regressor,  # The trained regressor model
            loop_id,  # Current loop ID
            run_id,  # Current run ID
            selected_recommendations,  # Selected recommendations (training data)
            all_labels,  # Corresponding labels
            test_data,  # Test data
            test_labels  # Test labels
        )

        # Append the MRE (Mean Relative Error) for six-point evaluation to the list
        test_pre.append(result['test_pre'].numpy())

        # Append the MRE for fitting to the list
        errors_list_fitting.append(result['mre_val'])

# Compute the average errors for the current loop
    errors.append({
        'loop': loop_id,
        'mre_fitting': np.mean(errors_list_fitting),
        'mre_sixpoint':evaluate_mre_points(np.mean(test_pre,axis=0),test_labels)
    })

# Save the average errors for all loops to a CSV file
with open('./params/evaluate_params/MRE_errors_for_each_loop.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['loop', 'MRE fitting', 'MRE sixpoint'])
    for error in errors:
        writer.writerow([error['loop'], error['mre_fitting'], error['mre_sixpoint']])

# Print completion message
print("Evaluation completed. Results saved to CSV.")
