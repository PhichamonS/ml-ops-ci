# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Functions and procedures
def plot_predictions(train_data, train_labels,  test_data, test_labels,  predictions):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(6, 5))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  # Show the legend
  plt.legend(shadow='True')
  # Set grids
  plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
  # Some text
  plt.title('Model Results', fontsize=14)
  plt.xlabel('X axis values', fontsize=11)
  plt.ylabel('Y axis values', fontsize=11)
  # Show
  plt.savefig('model_results.png', dpi=120)


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
