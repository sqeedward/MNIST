# Neural Network Implementation in Octave

This repository contains an implementation of a simple feedforward neural network in Octave. The code includes functions for calculating the cost and gradients, initializing weights, loading datasets, making predictions, and splitting the dataset into training and testing sets.

## Features

- **Cost Calculation**: Computes the cost using the cross-entropy loss function and includes L2 regularization.
- **Gradient Calculation**: Calculates gradients for backpropagation.
- **Weight Initialization**: Initializes weights using a uniform distribution based on the layer sizes.
- **Dataset Loading**: Function to load datasets from a specified path.
- **Prediction**: Uses the trained weights to predict classes based on input features.
- **Data Splitting**: Splits the dataset into training and testing sets based on a specified percentage.

## Functions

### `calculate_cost(X, Y, O1, O2, alpha)`

Calculates the cost and regularization term for a neural network.

- **Parameters**:
  - `X`: Input features.
  - `Y`: True labels.
  - `O1`: Weights for the hidden layer.
  - `O2`: Weights for the output layer.
  - `alpha`: Regularization parameter.

- **Returns**:
  - `J`: Total cost.
  - `cost`: Vector of individual costs.

### `cost_function(params, X, y, lambda, input_layer_size, hidden_layer_size, output_layer_size)`

Calculates the cost and gradients for the neural network using backpropagation.

- **Parameters**:
  - `params`: Flattened weights for both layers.
  - `X`: Input features.
  - `y`: True labels.
  - `lambda`: Regularization parameter.
  - `input_layer_size`: Number of input features.
  - `hidden_layer_size`: Number of hidden units.
  - `output_layer_size`: Number of output classes.

- **Returns**:
  - `grad`: Gradient of the cost with respect to weights.
  - `J`: Total cost.

### `initialize_weights(L_prev, L_next)`

Initializes the weights of a layer with a uniform distribution.

- **Parameters**:
  - `L_prev`: Number of neurons in the previous layer.
  - `L_next`: Number of neurons in the next layer.

- **Returns**:
  - `matrix`: Initialized weights.

### `load_dataset(path)`

Loads the dataset from the specified path.

- **Parameters**:
  - `path`: Path to the dataset file.

- **Returns**:
  - `X`: Input features.
  - `y`: True labels.

### `predict_classes(X, weights, input_layer_size, hidden_layer_size, output_layer_size)`

Predicts the output classes for given input features.

- **Parameters**:
  - `X`: Input features.
  - `weights`: Flattened weights for the neural network.
  - `input_layer_size`: Number of input features.
  - `hidden_layer_size`: Number of hidden units.
  - `output_layer_size`: Number of output classes.

- **Returns**:
  - `classes`: Predicted class probabilities.

### `split_dataset(X, y, percent)`

Splits the dataset into training and testing sets based on a specified percentage.

- **Parameters**:
  - `X`: Input features.
  - `y`: True labels.
  - `percent`: Percentage of data to use for training.

- **Returns**:
  - `x_train`: Training features.
  - `y_train`: Training labels.
  - `X_test`: Testing features.
  - `y_test`: Testing labels.

## Usage

1. Load your dataset using the `load_dataset` function.
2. Initialize the weights using `initialize_weights`.
3. Split the dataset into training and testing sets with `split_dataset`.
4. Train your model using the `cost_function` for forward propagation and backpropagation.
5. Make predictions using the `predict_classes` function.

## Requirements

- GNU Octave
- (Optional) Octave packages for matrix operations and data visualization.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit issues or pull requests for any improvements or bug fixes!
