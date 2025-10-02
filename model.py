# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Functions and procedures
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions, save_path='model_results.png', show=True):
    """
    Plot training, testing, and prediction results.

    Args:
        train_data (array-like): Feature values for training data.
        train_labels (array-like): Target labels for training data.
        test_data (array-like): Feature values for test data.
        test_labels (array-like): Target labels for test data.
        predictions (array-like): Model predictions on test data.
        save_path (str, optional): Path to save the plot. Default is 'model_results.png'.
        show (bool, optional): Whether to display the plot. Default is True.
    """
    # Basic validation
    assert len(train_data) == len(train_labels), "train_data and train_labels must have the same length"
    assert len(test_data) == len(test_labels), "test_data and test_labels must have the same length"
    assert len(test_data) == len(predictions), "test_data and predictions must have the same length"

    plt.figure(figsize=(8, 6))
    plt.scatter(train_data, train_labels, c="b", label="Training data", alpha=0.6)
    plt.scatter(test_data, test_labels, c="g", label="Testing data", alpha=0.6)
    plt.scatter(test_data, predictions, c="r", label="Predictions", alpha=0.6)

    plt.legend(shadow=True)
    plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
    plt.title('Model Results', fontsize=14)
    plt.xlabel('X axis values', fontsize=11)
    plt.ylabel('Y axis values', fontsize=11)

    if save_path:
        plt.savefig(save_path, dpi=120)
    if show:
        plt.show()



def mae(y_true, y_pred):
    """
    Calculates mean absolute error between y_true and y_pred.
    """
    metric = tf.keras.metrics.MeanAbsoluteError()
    metric.update_state(y_true, y_pred)
    return metric.result()


def mse(y_true, y_pred):
    """
    Calculates mean squared error between y_true and y_pred.
    """
    metric = tf.keras.metrics.MeanSquaredError()
    metric.update_state(y_true, y_pred)
    return metric.result()


# Check Tensorflow version
print("TensorFlow version:", tf.__version__)

# Create features and labels
X = np.arange(-100, 100, 4)
y = np.arange(-90, 110, 4)

# Split data into train and test sets
N = 25
X_train = X[:N]
y_train = y[:N]
X_test = X[N:]
y_test = y[N:]

# Reshape input arrays for Keras (samples, features)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Set random seed
tf.random.set_seed(1989)

# Create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,)), 
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['mae']
)

# Fit the model
model.fit(X_train, y_train, epochs=100, verbose=0)

# Make predictions
y_preds = model.predict(X_test)

# Plot predictions
plot_predictions(train_data=X_train, train_labels=y_train,
                 test_data=X_test, test_labels=y_test,
                 predictions=y_preds)

# Calculate metrics
mae_1 = np.round(float(mae(y_test, y_preds.squeeze()).numpy()), 2)
mse_1 = np.round(float(mse(y_test, y_preds.squeeze()).numpy()), 2)
print(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')

print('Done')

# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}.')
