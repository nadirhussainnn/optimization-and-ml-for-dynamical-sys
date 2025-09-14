import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import joblib

# Setting style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Vehicle parameters
m_eq = 1500
rho = 1.225
A_f = 2.2
c_d = 0.3
c_r = 0.01
g = 9.81

# Nonlinear dynamics function
def vehicle_dynamics(x, u, v_p, theta=0):
    v_h, d = x
    F_t, F_b = u
    
    # Resistance forces
    F_roll = m_eq * g * c_r * np.cos(theta)
    F_g = m_eq * g * np.sin(theta)
    F_air = 0.5 * rho * A_f * c_d * v_h**2
    F_resist = F_roll + F_g + F_air
    
    # Dynamics
    dv_h = (F_t - F_b - F_resist) / m_eq
    dd = v_p - v_h
    
    return np.array([dv_h, dd])

# Euler integration for discretization
def simulate_system(x0, u_sequence, v_p_sequence, dt=0.5, theta=0):
    N = len(u_sequence)
    x_sequence = np.zeros((N+1, len(x0)))
    x_sequence[0] = x0
    
    for k in range(N):
        dx = vehicle_dynamics(x_sequence[k], u_sequence[k], v_p_sequence[k], theta)
        x_sequence[k+1] = x_sequence[k] + dx * dt
        
    return x_sequence

# Generating training data with more distance-focused scenarios
def generate_training_data(num_samples=50000, dt=0.5):
    X = np.zeros((num_samples, 5))  # [v_h, d, F_t, F_b, v_p]
    y = np.zeros((num_samples, 2))  # [v_h_next, d_next]
    
    for i in range(num_samples):
        # Created different scenarios with emphasis on distance changes
        scenario = np.random.choice(['normal', 'close_following', 'rapid_change'])
        
        if scenario == 'normal':
            v_h = np.random.uniform(5, 30)
            d = np.random.uniform(10, 100)
            F_t = np.random.uniform(0, 4000)
            F_b = np.random.uniform(0, 2000)
            v_p = np.random.uniform(5, 30)
        elif scenario == 'close_following':
            v_h = np.random.uniform(5, 30)
            d = np.random.uniform(2, 20)  # Closer distances
            F_t = np.random.uniform(0, 5000)
            F_b = np.random.uniform(0, 3000)
            v_p = v_h + np.random.uniform(-5, 5)  # Similar to host velocity
        else:  # rapid_change
            v_h = np.random.uniform(5, 30)
            d = np.random.uniform(5, 50)
            F_t = np.random.uniform(2000, 6000)
            F_b = np.random.uniform(1000, 4000)
            v_p = v_h + np.random.uniform(-10, 10)  # Large differences
        
        X[i] = [v_h, d, F_t, F_b, v_p]
        
        # Simulating one step
        x_current = np.array([v_h, d])
        u_current = np.array([F_t, F_b])
        dx = vehicle_dynamics(x_current, u_current, v_p)
        x_next = x_current + dx * dt
        
        # Storing next state
        y[i] = x_next
    
    return X, y

# Custom loss function in order to weight distance error more, i.e safety first then speed
def custom_loss(y_true, y_pred):
    # Separating velocity and distance components
    velocity_true = y_true[:, 0]
    velocity_pred = y_pred[:, 0]
    distance_true = y_true[:, 1]
    distance_pred = y_pred[:, 1]
    
    # Calculating losses with different weights
    velocity_loss = tf.reduce_mean(tf.square(velocity_true - velocity_pred))
    distance_loss = tf.reduce_mean(tf.square(distance_true - distance_pred))
    
    # Weight distance more heavily (10:1 ratio)
    return velocity_loss + 10.0 * distance_loss

# Simple NN model
def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(output_dim)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=custom_loss,  # We use our custom loss function here
                  metrics=['mae'])
    return model

# Generating data
print("Generating training data...")
X, y = generate_training_data(50000)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scaling data separately for better distance handling
scaler_X = StandardScaler()
scaler_y_v = StandardScaler()
scaler_y_d = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# Scaling velocity and distance outputs separately
y_train_v_scaled = scaler_y_v.fit_transform(y_train[:, 0].reshape(-1, 1))
y_train_d_scaled = scaler_y_d.fit_transform(y_train[:, 1].reshape(-1, 1))
y_train_scaled = np.column_stack((y_train_v_scaled, y_train_d_scaled))

y_val_v_scaled = scaler_y_v.transform(y_val[:, 0].reshape(-1, 1))
y_val_d_scaled = scaler_y_d.transform(y_val[:, 1].reshape(-1, 1))
y_val_scaled = np.column_stack((y_val_v_scaled, y_val_d_scaled))

y_test_v_scaled = scaler_y_v.transform(y_test[:, 0].reshape(-1, 1))
y_test_d_scaled = scaler_y_d.transform(y_test[:, 1].reshape(-1, 1))
y_test_scaled = np.column_stack((y_test_v_scaled, y_test_d_scaled))

# Creating and train model
print("Building and training model...")
model = build_model(X_train.shape[1], y_train.shape[1])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train_scaled, 
                    epochs=100, batch_size=32, 
                    validation_data=(X_val_scaled, y_val_scaled), verbose=1,
                    callbacks=[early_stopping])

# Evaluating model on all sets
print("Evaluating model...")

# Training set evaluation
y_train_pred_scaled = model.predict(X_train_scaled)
y_train_pred_v = scaler_y_v.inverse_transform(y_train_pred_scaled[:, 0].reshape(-1, 1))
y_train_pred_d = scaler_y_d.inverse_transform(y_train_pred_scaled[:, 1].reshape(-1, 1))
y_train_pred = np.column_stack((y_train_pred_v, y_train_pred_d))

# Validation set evaluation
y_val_pred_scaled = model.predict(X_val_scaled)
y_val_pred_v = scaler_y_v.inverse_transform(y_val_pred_scaled[:, 0].reshape(-1, 1))
y_val_pred_d = scaler_y_d.inverse_transform(y_val_pred_scaled[:, 1].reshape(-1, 1))
y_val_pred = np.column_stack((y_val_pred_v, y_val_pred_d))

# Test set evaluation
y_test_pred_scaled = model.predict(X_test_scaled)
y_test_pred_v = scaler_y_v.inverse_transform(y_test_pred_scaled[:, 0].reshape(-1, 1))
y_test_pred_d = scaler_y_d.inverse_transform(y_test_pred_scaled[:, 1].reshape(-1, 1))
y_test_pred = np.column_stack((y_test_pred_v, y_test_pred_d))

# Calculating metrics for all sets
def calculate_metrics(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    mse_velocity = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_distance = mean_squared_error(y_true[:, 1], y_pred[:, 1])
    mae_velocity = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_distance = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    
    r2_velocity = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_distance = r2_score(y_true[:, 1], y_pred[:, 1])
    r2_overall = r2_score(y_true, y_pred)
    
    print(f"\n{set_name} Set Performance:")
    print(f"Overall MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2_overall:.6f}")
    print(f"Velocity MSE: {mse_velocity:.6f}, MAE: {mae_velocity:.6f}, R²: {r2_velocity:.6f}")
    print(f"Distance MSE: {mse_distance:.6f}, MAE: {mae_distance:.6f}, R²: {r2_distance:.6f}")
    
    return {
        'mse': mse, 'mae': mae, 'r2': r2_overall,
        'mse_velocity': mse_velocity, 'mae_velocity': mae_velocity, 'r2_velocity': r2_velocity,
        'mse_distance': mse_distance, 'mae_distance': mae_distance, 'r2_distance': r2_distance
    }

train_metrics = calculate_metrics(y_train, y_train_pred, "Training")
val_metrics = calculate_metrics(y_val, y_val_pred, "Validation")
test_metrics = calculate_metrics(y_test, y_test_pred, "Test")

# Creating comprehensive visualization
fig = plt.figure(figsize=(12, 8))

# 1. Training history
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True)

# 2. Compare with real system on a test trajectory
def create_test_trajectory():
    """Create a realistic test trajectory"""
    T = 50  # time steps
    dt = 0.5
    
    # Preceding vehicle velocity profile
    v_p = np.ones(T) * 20  # m/s
    v_p[10:20] = 25  # acceleration
    v_p[30:40] = 15  # deceleration
    
    # Control inputs
    F_t = np.zeros(T)
    F_b = np.zeros(T)
    
    # Simple controller for demonstration
    for i in range(T):
        if i < 10:
            F_t[i] = 2000  # Accelerate
        elif i < 20:
            F_t[i] = 1500  # Maintain
        elif i < 30:
            F_t[i] = 1000  # Gentle deceleration
        else:
            F_b[i] = 500   # Brake
    
    return v_p, F_t, F_b, dt

# Create test trajectory
v_p_test, F_t_test, F_b_test, dt = create_test_trajectory()
T = len(v_p_test)

# Initial state
x0 = np.array([5, 15])  # v_h=5 m/s, d=15 m

# Simulate real system
u_sequence = np.column_stack((F_t_test, F_b_test))
x_real = simulate_system(x0, u_sequence, v_p_test, dt)

# Simulate with neural network
x_nn = np.zeros((T+1, 2))
x_nn[0] = x0

for k in range(T):
    # Prepare input for NN
    nn_input = np.array([x_nn[k, 0], x_nn[k, 1], F_t_test[k], F_b_test[k], v_p_test[k]])
    nn_input_scaled = scaler_X.transform(nn_input.reshape(1, -1))
    
    # Predict next state
    nn_output_scaled = model.predict(nn_input_scaled, verbose=0)
    
    nn_output_v = scaler_y_v.inverse_transform(nn_output_scaled[:, 0].reshape(-1, 1))
    nn_output_d = scaler_y_d.inverse_transform(nn_output_scaled[:, 1].reshape(-1, 1))
    
    nn_output = np.column_stack((nn_output_v, nn_output_d))
    
    x_nn[k+1] = nn_output[0]

# 7. Velocity comparison
plt.subplot(2, 2, 2)
plt.plot(np.arange(T+1)*dt, x_real[:, 0], 'b-', label='Real System', linewidth=2)
plt.plot(np.arange(T+1)*dt, x_nn[:, 0], 'r--', label='Neural Network', linewidth=2)
plt.plot(np.arange(T)*dt, v_p_test, 'g-', label='Preceding Vehicle', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Comparison')
plt.legend()
plt.grid(True)

# 8. Distance comparison
plt.subplot(2, 2, 3)
plt.plot(np.arange(T+1)*dt, x_real[:, 1], 'b-', label='Real System', linewidth=2)
plt.plot(np.arange(T+1)*dt, x_nn[:, 1], 'r--', label='Neural Network', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Distance Comparison')
plt.legend()
plt.grid(True)

# 9. Prediction errors
plt.subplot(2, 2, 4)
velocity_error = np.abs(x_real[:, 0] - x_nn[:, 0])
distance_error = np.abs(x_real[:, 1] - x_nn[:, 1])
plt.plot(np.arange(T+1)*dt, velocity_error, 'b-', label='Velocity Error')
plt.plot(np.arange(T+1)*dt, distance_error, 'r-', label='Distance Error')
plt.xlabel('Time (s)')
plt.ylabel('Absolute Error')
plt.title('Prediction Errors on Test Trajectory')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('neural_net_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate performance metrics for the test trajectory
mse_velocity = mean_squared_error(x_real[:, 0], x_nn[:, 0])
mse_distance = mean_squared_error(x_real[:, 1], x_nn[:, 1])
mae_velocity = mean_absolute_error(x_real[:, 0], x_nn[:, 0])
mae_distance = mean_absolute_error(x_real[:, 1], x_nn[:, 1])
r2_velocity = r2_score(x_real[:, 0], x_nn[:, 0])
r2_distance = r2_score(x_real[:, 1], x_nn[:, 1])

print(f"\nTest Trajectory Performance:")
print(f"Velocity MSE: {mse_velocity:.6f}, MAE: {mae_velocity:.6f}, R²: {r2_velocity:.6f}")
print(f"Distance MSE: {mse_distance:.6f}, MAE: {mae_distance:.6f}, R²: {r2_distance:.6f}")

# Save scalers
joblib.dump(scaler_X, 'scaler_X.joblib')
joblib.dump(scaler_y_v, 'scaler_y_v.joblib')
joblib.dump(scaler_y_d, 'scaler_y_d.joblib')
print("Scalers saved.")

# Save model in native Keras format
model.save('vehicle_dynamics_model.keras')
print("Model saved as 'vehicle_dynamics_model.keras'")
