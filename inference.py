import numpy as np
import tensorflow as tf
import joblib

# Define the custom loss function (must match the one used during training)
def custom_loss(y_true, y_pred):
    velocity_true = y_true[:, 0]
    velocity_pred = y_pred[:, 0]
    distance_true = y_true[:, 1]
    distance_pred = y_pred[:, 1]
    
    velocity_loss = tf.reduce_mean(tf.square(velocity_true - velocity_pred))
    distance_loss = tf.reduce_mean(tf.square(distance_true - distance_pred))
    
    return velocity_loss + 10.0 * distance_loss

# Load the saved model
model = tf.keras.models.load_model('vehicle_dynamics_model.keras', custom_objects={'custom_loss': custom_loss})
print("Model loaded successfully.")

# Load the scalers
scaler_X = joblib.load('scaler_X.joblib')
scaler_y_v = joblib.load('scaler_y_v.joblib')
scaler_y_d = joblib.load('scaler_y_d.joblib')
print("Scalers loaded successfully.")

# Example input samples: [v_h, d, F_t, F_b, v_p]
# You can replace with your own data or read from a file/CLI args
samples = np.array([
    [10.0, 20.0, 3000.0, 0.0, 15.0],  # Example 1: Moderate speed, closing distance
    [20.0, 10.0, 0.0, 1000.0, 18.0],  # Example 2: Braking scenario
    [5.0, 50.0, 5000.0, 0.0, 25.0]     # Example 3: Rapid acceleration
])

# Scale the inputs
samples_scaled = scaler_X.transform(samples)

# Make predictions (scaled)
preds_scaled = model.predict(samples_scaled)

# Inverse scale the outputs
preds_v = scaler_y_v.inverse_transform(preds_scaled[:, 0].reshape(-1, 1))
preds_d = scaler_y_d.inverse_transform(preds_scaled[:, 1].reshape(-1, 1))

# Combine into [next_v_h, next_d]
preds = np.column_stack((preds_v, preds_d))

# Print results
print("\nPredictions:")
for i, (sample, pred) in enumerate(zip(samples, preds)):
    print(f"Sample {i+1}: Input = {sample}")
    print(f"Predicted next state: [v_h = {pred[0]:.4f}, d = {pred[1]:.4f}]\n")