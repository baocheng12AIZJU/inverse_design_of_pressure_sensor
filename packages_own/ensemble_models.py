
import torch  # PyTorch library for tensor computations

class EnsembleRegressor:
    def __init__(self, num_models=5, model_class=None, model_weights_path="./params/NN_params/"):
        """
        Initialize the ensemble regressor class.

        Parameters:
        - num_models (int): The number of models in the ensemble.
        - model_class (class): The class of the individual models, which must be instantiable.
        - model_weights_path (str): The file path where model weights are stored.
        """
        self.num_models = num_models
        self.model_class = model_class
        self.model_weights_path = model_weights_path
        self.models = self._initialize_models()  # Initialize the ensemble with the specified number of models

    def _initialize_models(self):
        """
        Initialize a list of model instances.

        Returns:
        - A list containing `num_models` instances of the specified model class.
        """
        models = [self.model_class() for _ in range(self.num_models)]
        return models

    def load_model_weights(self, loop):
        """
        Load pretrained weights for each model in the ensemble.

        Parameters:
        - loop (int): The current iteration/loop index used to specify the file names of the weights.
        """
        for i, model in enumerate(self.models):
            # Construct the file path for the model weights
            weight_path = f"{self.model_weights_path}/regressor_{loop}-{i}.pth"

            # Load the state dictionary (weights) into the model
            model.load_state_dict(torch.load(weight_path))

            # Set the model to evaluation mode to disable dropout and batch normalization during inference
            model.eval()

    def predict(self, input_tensor):
        """
        Perform prediction using the ensemble and return the average result.

        Parameters:
        - input_tensor (torch.Tensor): The input tensor to be passed to the models.

        Returns:
        - avg_output (torch.Tensor): The average prediction output from all models in the ensemble.
        """
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        # Get predictions from all models in the ensemble
        outputs = [model(input_tensor) for model in self.models]

        # Compute the mean of the predictions across all models
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output
