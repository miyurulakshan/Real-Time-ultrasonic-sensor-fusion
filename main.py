import serial
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

# Kalman Filter Parameters
Q = 0.000001  # Process Variance
R = 1000    # Measurement Variance

# Exponential Moving Average Filter Parameters
alpha = 0.2

ser = serial.Serial('/dev/tty.usbmodem141301', 9600)  # Replace '/dev/tty.usbmodem141301' with your identified serial port


# Function to apply the low-pass FIR filter with optimized coefficients
def apply_filter(samples, coefficients):
    return lfilter(coefficients, 1.0, samples)


# Function to evaluate the fitness of a set of filter coefficients using a simple objective function
def fitness_function(coefficients):
    filtered_data = apply_filter(raw_data, coefficients)
    error = np.mean(np.abs(filtered_data - raw_data))
    return 1 / (1 + error)  # Inverse the error to get higher fitness for lower errors


# Kalman Filter function
def kalman_filter(data):
    n = len(data)
    x_est = np.zeros(n)
    P_est = np.zeros(n)
    K_gain = np.zeros(n)
    x_est[0] = data[0]
    P_est[0] = 1.0

    for i in range(1, n):
        x_pred = x_est[i-1]
        P_pred = P_est[i-1] + Q

        K_gain[i] = P_pred / (P_pred + R)
        x_est[i] = x_pred + K_gain[i] * (data[i] - x_pred)
        P_est[i] = (1 - K_gain[i]) * P_pred

    return x_est

# Exponential Moving Average Filter function
def exp_moving_average(data):
    n = len(data)
    ema = np.zeros(n)
    ema[0] = data[0]

    for i in range(1, n):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

    return ema


try:
    raw_data_sensor1 = []
    raw_data_sensor2 = []
    raw_data_sensor3 = []

    while True:
        data = ser.readline().decode().strip()
        if data:
            distance1_cm, distance2_cm, distance3_cm = map(float, data.split(','))
            raw_data_sensor1.append(distance1_cm)
            raw_data_sensor2.append(distance2_cm)
            raw_data_sensor3.append(distance3_cm)
            print(f"Raw Data - Sensor 1: {distance1_cm:.2f} , Sensor 2: {distance2_cm:.2f} , Sensor 3: {distance3_cm:.2f} ")
except KeyboardInterrupt:
    ser.close()
    print("Serial connection closed.")

# Create subplots for raw data from each sensor in centimeters
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(raw_data_sensor1, label='Sensor 1 Raw Data', color='blue', alpha=0.6)
plt.ylabel('Ultrasonic Signal Travelling time (Micro seconds)')
plt.title('Raw Ultrasonic Signal Data - Sensor 1')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(raw_data_sensor2, label='Sensor 2 Raw Data', color='green', alpha=0.6)
plt.ylabel('Ultrasonic Signal Travelling time (Micro seconds)')
plt.title('Raw Ultrasonic Signal Data - Sensor 2')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(raw_data_sensor3, label='Sensor 3 Raw Data', color='red', alpha=0.6)
plt.xlabel('Time')
plt.ylabel('Ultrasonic Signal Travelling time (Micro seconds)')
plt.title('Raw Ultrasonic Signal Data - Sensor 3')
plt.grid(True)

plt.tight_layout()

# Apply filters and techniques to the data from each sensor
filtered_data_sensor1 = kalman_filter(raw_data_sensor1)
filtered_data_sensor2 = kalman_filter(raw_data_sensor2)
filtered_data_sensor3 = kalman_filter(raw_data_sensor3)

# Apply exponential moving average filter to the Kalman-filtered data from each sensor
filtered_data_sensor1 = exp_moving_average(filtered_data_sensor1)
filtered_data_sensor2 = exp_moving_average(filtered_data_sensor2)
filtered_data_sensor3 = exp_moving_average(filtered_data_sensor3)


# Create subplots for filtered data from each sensor in centimeters
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(filtered_data_sensor1, label='Sensor 1 Filtered Data', color='blue')

plt.title('Filtered Ultrasonic Signal Data - Sensor 1')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(filtered_data_sensor2, label='Sensor 2 Filtered Data', color='green')
plt.ylabel('Ultrasonic Signal Travelling time (Micro seconds)')
plt.title('Filtered Ultrasonic Signal  Data - Sensor 2')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(filtered_data_sensor3, label='Sensor 3 Filtered Data', color='red')
plt.xlabel('Time')

plt.title('Filtered Ultrasonic Signal Data - Sensor 3')
plt.grid(True)

plt.tight_layout()

# Check if one sensor has abnormal data, then discard its data and combine the data from the other two sensors
if np.std(filtered_data_sensor1) > 100:
    combined_data = (filtered_data_sensor2 + filtered_data_sensor3) / 2.0
    abnormal_sensor = "Sensor 1"
elif np.std(filtered_data_sensor2) > 100:
    combined_data = (filtered_data_sensor1 + filtered_data_sensor3) / 2.0
    abnormal_sensor = "Sensor 2"
elif np.std(filtered_data_sensor3) > 100:
    combined_data = (filtered_data_sensor1 + filtered_data_sensor2) / 2.0
    abnormal_sensor = "Sensor 3"
else:
    combined_data = (filtered_data_sensor1 + filtered_data_sensor2 + filtered_data_sensor3) / 3.0
    abnormal_sensor = None

# Convert combined data back to centimeters for printing
combined_data_cm = combined_data

# Create a plot for the combined data
final_value_cm = int((combined_data_cm[-1] / 2) * 0.0343)

print(f"Final Filtered and Combined Sensor Value: {final_value_cm}.001 cm")

plt.figure(figsize=(12, 6))
plt.plot(combined_data_cm, label='Combined Filtered Data', color='purple')
plt.xlabel('Time')
plt.ylabel('Ultrasonic Signal Travelling time (Micro seconds)')
plt.title('Combined Filtered Ultrasonic Signal Travelling time Data')
plt.grid(True)
plt.legend()

# Display information about the abnormal sensor, if any
if abnormal_sensor:
    plt.text(0.5, 0.9, f"Abnormal Sensor: {abnormal_sensor}", fontsize=12, transform=plt.gca().transAxes, ha='center')

# Print the final filtered and combined sensor value in centimeters

plt.tight_layout()
plt.show()
