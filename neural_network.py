import numpy as np
import argparse


class NeuralNetwork:
    def __init__(self, layer_sizes, activation="relu", output_activation="softmax", learning_rate=0.01):
        """
        Initialize the neural network.

        Args:
            layer_sizes: List of integers representing neurons in each layer
                        e.g., [784, 128, 64, 10] for input->hidden1->hidden2->output
            activation: Activation function for hidden layers ('relu', 'sigmoid', 'tanh')
            output_activation: Activation for output layer ('softmax', 'sigmoid', 'linear')
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        # He initialization for weights
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    # Activation functions
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def linear(self, z):
        return z

    def activate(self, z, activation_type):
        """Apply activation function"""
        if activation_type == "relu":
            return self.relu(z)
        elif activation_type == "sigmoid":
            return self.sigmoid(z)
        elif activation_type == "tanh":
            return self.tanh(z)
        elif activation_type == "softmax":
            return self.softmax(z)
        elif activation_type == "linear":
            return self.linear(z)

    def activate_derivative(self, z, activation_type):
        """Apply derivative of activation function"""
        if activation_type == "relu":
            return self.relu_derivative(z)
        elif activation_type == "sigmoid":
            return self.sigmoid_derivative(z)
        elif activation_type == "tanh":
            return self.tanh_derivative(z)

    def forward(self, X):
        """
        Forward propagation through the network.

        Args:
            X: Input data of shape (batch_size, input_features)

        Returns:
            Output of the network and cache of intermediate values
        """
        cache = {"A0": X}
        A = X

        # Forward through hidden layers
        for i in range(self.num_layers - 2):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.activate(Z, self.activation)
            cache[f"Z{i+1}"] = Z
            cache[f"A{i+1}"] = A

        # Output layer
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.activate(Z, self.output_activation)
        cache[f"Z{self.num_layers-1}"] = Z
        cache[f"A{self.num_layers-1}"] = A

        return A, cache

    def backward(self, X, y, cache):
        """
        Backward propagation to compute gradients.

        Args:
            X: Input data
            y: True labels
            cache: Dictionary of intermediate values from forward pass

        Returns:
            Gradients for weights and biases
        """
        m = X.shape[0]
        grads = {}

        # Output layer gradient
        A_out = cache[f"A{self.num_layers-1}"]

        if self.output_activation == "softmax":
            dZ = A_out - y
        else:
            dZ = (A_out - y) * self.activate_derivative(cache[f"Z{self.num_layers-1}"], self.output_activation)

        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            A_prev = cache[f"A{i}"]

            grads[f"dW{i}"] = np.dot(A_prev.T, dZ) / m
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self.activate_derivative(cache[f"Z{i}"], self.activation)

        return grads

    def update_parameters(self, grads):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads[f"dW{i}"]
            self.biases[i] -= self.learning_rate * grads[f"db{i}"]

    def compute_loss(self, y_pred, y_true):
        """Compute cross-entropy loss"""
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss

    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """
        Train the neural network.

        Args:
            X: Training data
            y: Training labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            verbose: Whether to print training progress
        """
        n_samples = X.shape[0]
        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            batches = 0

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward pass
                y_pred, cache = self.forward(X_batch)

                # Compute loss
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss
                batches += 1

                # Backward pass
                grads = self.backward(X_batch, y_batch, cache)

                # Update parameters
                self.update_parameters(grads)

            # Calculate average loss and accuracy
            avg_loss = epoch_loss / batches
            y_pred_full, _ = self.forward(X)
            accuracy = self.accuracy(y_pred_full, y)

            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

        return history

    def predict(self, X):
        """Make predictions on new data"""
        y_pred, _ = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def accuracy(self, y_pred, y_true):
        """Calculate accuracy"""
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == labels)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train a Neural Network from scratch using NumPy", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Network architecture
    parser.add_argument(
        "--layer_sizes", type=int, nargs="+", default=[2, 8, 4, 2], help="List of layer sizes (e.g., --layer_sizes 2 8 4 2)"
    )

    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")

    # Activation functions
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "tanh"],
        help="Activation function for hidden layers",
    )
    parser.add_argument(
        "--output_activation",
        type=str,
        default="softmax",
        choices=["softmax", "sigmoid", "linear"],
        help="Activation function for output layer",
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print training progress")
    parser.add_argument("--no_verbose", dest="verbose", action="store_false", help="Do not print training progress")

    return parser.parse_args()


def create_synthetic_dataset():
    """Create a simple synthetic dataset (XOR problem extended)"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0]])
    return X, y


def main():
    """Main function to run the neural network with CLI arguments"""
    args = parse_arguments()

    # Set random seed
    np.random.seed(args.seed)

    # Create dataset
    X, y = create_synthetic_dataset()

    # Validate layer sizes with dataset
    if args.layer_sizes[0] != X.shape[1]:
        print(f"Warning: First layer size ({args.layer_sizes[0]}) doesn't match input features ({X.shape[1]})")
        print(f"Adjusting first layer to {X.shape[1]}")
        args.layer_sizes[0] = X.shape[1]

    if args.layer_sizes[-1] != y.shape[1]:
        print(f"Warning: Last layer size ({args.layer_sizes[-1]}) doesn't match output classes ({y.shape[1]})")
        print(f"Adjusting last layer to {y.shape[1]}")
        args.layer_sizes[-1] = y.shape[1]

    # Create and train network
    nn = NeuralNetwork(
        layer_sizes=args.layer_sizes,
        activation=args.activation,
        output_activation=args.output_activation,
        learning_rate=args.learning_rate,
    )

    print("Training Neural Network...")
    print()
    history = nn.train(X, y, epochs=args.num_epochs, batch_size=args.batch_size, verbose=args.verbose)

    # Make predictions
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Final Loss:     {history['loss'][-1]:.4f}")
    print(f"Final Accuracy: {history['accuracy'][-1]:.4f}")
    print()
    print("Predictions:")
    predictions = nn.predict(X)
    for i, (input_val, pred, true) in enumerate(zip(X, predictions, np.argmax(y, axis=1))):
        status = "✓" if pred == true else "✗"
        print(f"{status} Input: {input_val} -> Predicted: {pred}, True: {true}")
    print("=" * 60)


if __name__ == "__main__":
    main()
