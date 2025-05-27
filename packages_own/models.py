import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from packages_own.Navigation import Navigation
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from scipy.stats import linregress
class Regressor(nn.Module):
    def __init__(self, input_size=4, hidden_sizes=[15,17,15], output_size=3, dropout_rate=0):
        super(Regressor, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes):  # Add dropout layer if not the output layer
                layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.reg = nn.Sequential(*layers)

        self.initialize()

    def initialize(self):
        """Initialize the model parameters using Kaiming normalization for weights
                and setting biases to zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def train_model(self, train_loader, n_epochs=4000):
        """
        Train the model using the provided training data loader.

        Args:
            train_loader (DataLoader): Data loader for training data.
            n_epochs (int): Number of training epochs.

        Returns:
            list: A list of average epoch losses.
        """
        train_losses = []
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0031236)
        criterion = nn.L1Loss()

        for epoch in tqdm(range(n_epochs), desc="Training Progress"):
            self.train()
            epoch_loss = 0.0

            for inputs_batch, targets_batch in train_loader:
                # Forward pass
                outputs = self.reg(inputs_batch)
                loss = criterion(outputs, targets_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Calculate and store average loss for the epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)

        return train_losses






    def predict(self,x ):
        self.eval()
        return self.reg(x)

    def forward(self, x):
        return self.reg(x)

    def evaluate(self, inputs_val, targets_val):
        """
        Evaluate the model using validation data.

        Args:
            inputs_val (Tensor): Validation input data.
            targets_val (Tensor): Validation target data.

        Returns:
            tuple: MSE, MAE, MRE losses, and model predictions.
        """
        self.eval()
        with torch.no_grad():
            outputs_val = self.reg(inputs_val)

            # Calculate Mean Squared Error (MSE)
            criterion_mse = nn.MSELoss()
            loss_mse = criterion_mse(outputs_val, targets_val).item()

            # Calculate Mean Absolute Error (MAE)
            criterion_mae = nn.L1Loss()
            loss_mae = criterion_mae(outputs_val, targets_val).item()

            # Calculate Mean Relative Error (MRE) of fitting parameters
            criterion_mre = lambda x, y: torch.mean(torch.abs((x - y) / y))
            loss_mre = criterion_mre(outputs_val, targets_val).item()

        return loss_mse, loss_mae, loss_mre, outputs_val
def plot_data(te_pre_average, te_label, te_data, round_cnt, aug=False):
    """
    Plot predicted vs ground truth data for visualization.
    """
    navigator = Navigation()
    te_data_denorm = te_data * (navigator.param_max - navigator.param_min) + navigator.param_min
    x_extract = np.arange(0.01, 40, 0.1)  # X-axis range

    plt.figure(figsize=(10, 8))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 36

    for i in range(te_pre_average.shape[0]):
        y_pred = expotionalp(x_extract, *te_pre_average[i])
        y_true = expotionalp(x_extract, *te_label.numpy()[i])

        plt.plot(x_extract * 10, y_pred, label='Predicted', color='red')
        plt.plot(x_extract * 10, y_true, label='Ground Truth', color='blue')
        plt.ylabel('Voltage (V)')
        plt.xlabel('Pressure (KPa)')
        plt.title(f'Sensor({te_data_denorm[i][0]:.3f}/{te_data_denorm[i][1]:.2f}/'
                  f'{te_data_denorm[i][2]:.2f}/{te_data_denorm[i][3]:.2f})', fontsize=36)
        plt.legend()
        plt.tight_layout()
        plt.clf()
        plt.cla()

    plt.close('all')
    return




def load_testdata():
    """
    Load test data and labels.
    """
    te_data = np.load('./params/NN_params/features_testset_norm.npy')
    te_label = np.load('./params/NN_params/labels_testset.npy')
    return torch.from_numpy(te_data).float(), torch.from_numpy(te_label).float()


def load_traindata(loop, batch_size=5000):
    """
    Load training data for a specific loop.
    """
    selected_recom = np.load(f'./params/material_params/{loop}_selected_params.npy')
    all_labels = np.load(f'./params/NN_params/all_labels_{loop}.npy')
    selected_recom = torch.from_numpy(selected_recom).float()
    all_labels = torch.from_numpy(all_labels).float()



    train_dataset = TensorDataset(selected_recom, all_labels)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), selected_recom, all_labels


def train_model_each_loop_run(loop, run, train_loader,regressor=None):
    """
    Train the model for a given loop and run.
    """
    regressor.train_model(train_loader)
    # torch.save(regressor.state_dict(), f'./params/NN_params/regressor_{loop}-{run}.pth')
    return regressor


def evaluate_model_each_loop_run(regressor, loop, run, selected_recom, all_labels, te_data, te_label):
    """
    Evaluate the trained model on training and test datasets.
    """
    tra_metrics = regressor.evaluate(selected_recom, all_labels)
    te_metrics = regressor.evaluate(te_data, te_label)

    return {
        'round': loop,
        'run': run,
        'mse_train': tra_metrics[0],
        'mae_train': tra_metrics[1],
        'mre_train': tra_metrics[2],
        'mse_val': te_metrics[0],
        'mae_val': te_metrics[1],
        'mre_val': te_metrics[2],
        'test_pre': te_metrics[3],
    }


def evaluate_mre_points(te_pre_average, te_label):
    if isinstance(te_pre_average, torch.Tensor):
        te_pre_average = te_pre_average.numpy()
    if isinstance(te_label, torch.Tensor):
        te_label = te_label.numpy()
    x_extract = np.array([2, 10, 50, 100,250, 400])  # 最终使用
    mre_sixpoint_all = []
    for i in range(te_label.shape[0]):
        y_pred = expotionalp(x_extract, *te_pre_average[i])
        y_true = expotionalp(x_extract, *te_label[i])
        mre = np.mean(np.abs(y_pred - y_true) / y_true)
        mre_sixpoint_all.append(mre)
    return np.mean(mre_sixpoint_all)
def expotionalp(x, a, b, c):
    """
    Exponential function used for fitting.
    """
    return np.exp(a + b / (x/10 + c))

def calculate_linear_properties(target_pressure_range, params, func):
    """
        Calculate slopes and R-squared values of single params for linear properties.
        """
    y = func(target_pressure_range, *params)
    slope, intercept, r_value, p_value, std_err = linregress(target_pressure_range, y)
    r_squared = r_value ** 2
    return r_squared, slope
def calculate_linear_properties_array(x, params, func):
    """
    Calculate slopes and R-squared values of arrays for linear properties.
    """
    slopes, r_squared = [], []

    for param_set in params:
        y = func(x, *param_set)
        # y=y-func(0, *param_set)#减去Vo,不受影响
        slope, intercept, r_value, _, _ = linregress(x, y)
        slopes.append(slope)
        r_squared.append(r_value ** 2)

    return  np.array(r_squared),np.array(slopes)
def calculate_voltages_array(x, params, func):
    """
    Calculate slopes and R-squared values of arrays for linear properties.
    """
    voltages = []

    for param_set in params:
        voltage = func(x, *param_set)
        # voltage = voltage - func(0, *param_set)
        voltages.append(voltage)


    return  np.array(voltages)

