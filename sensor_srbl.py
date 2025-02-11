#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Seunghoon Kang
Email: alaska97@snu.ac.kr
Affiliation: Soft Robotics & Bionics Lab, Seoul National Univeristy
Date: 2024-10-01
Description: ALCHEMIST project
            Reads sensor data from an ESP32 using serial communication
update: version 5.1 (2025-02-11)
"""

import time
import serial
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import joblib

DEFAULT_HZ = 150  # Default frequency (Hz)

class MLPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(4, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 3)
        # )
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # return self.model(x)


class FingerSensor:
    def __init__(self, 
        port: str = '/dev/ttyUSB0', 
        baudrate: int = 115200, 
        timeout: int = 1,
        hz: int = DEFAULT_HZ
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.hz = hz
        self.mcu = None
        self.offsets = np.zeros(8)  # Initialize offsets

        # Open Serial
        self.open_serial()
        self.initialize_offset()

        # Load Sensor Calibration Model
        self.calibration = SensorForce()

    def open_serial(self) -> None:
        """Open the serial connection to the ESP32."""
        try:
            self.mcu = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.mcu.flushInput() # Flush input buffer instead of waiting
            print("Port Opened.")
        except serial.SerialException as e:
            print(f"Error opening serial connection: {e}")
            raise

    def read(self) -> str:
        max_attempts = 5  # Avoid infinite loop in case of hardware issues
        attempts = 0
        while attempts < max_attempts:
            try:
                self.mcu.write(b'A')  # Send 'A' to request data
                time.sleep(0.001)  # Small delay to allow ESP32 to respond
                data = self.mcu.readline().decode('utf-8').strip()
                if data:  # If data is received, return it
                    return data
                else:
                    print(f"Warning: Empty response from sensor. Retrying... ({attempts+1}/{max_attempts})")
            except Exception as e:
                print(f"Error reading serial data: {e}")
            
            attempts += 1
            time.sleep(0.005)  # Short delay before retrying

        raise RuntimeError("Failed to receive valid data from sensor after multiple attempts.")
 
    def initialize_offset(self, max_samples: int = 40) -> None:
        """Initialize sensor offset by averaging values over a period of time or a number of samples."""
        print("Initializing sensor offsets...")

        collected_data = []

        while len(collected_data) < max_samples :
            try:
                data = self.read()
                values = np.array(data.split(','), dtype=float)  # Use NumPy for faster conversion
                if values.size == 8:
                    collected_data.append(values)
            except ValueError:
                print(f"Skipping invalid data: {data}")
            time.sleep(1.0 / self.hz)  # Wait for the next sample based on the sensor frequency

        collected_data = np.stack(collected_data)  # Stack into a 2D NumPy array
        self.offsets = np.mean(collected_data, axis=0)  # Compute the mean for each column
        print(f"Sensor offsets initialized to: {self.offsets}")

    def split_read(self) -> Tuple[np.ndarray, np.ndarray]:
        data = self.read()
        try:
            values = np.array(data.split(','), dtype=float) - self.offsets
            sensor1 = -1* values.reshape(4, 2)[:, 0]  # Extract even indices [0,2,4,6]
            sensor2 = -1* values.reshape(4, 2)[:, 1]  # Extract odd indices [1,3,5,7]
            return sensor1, sensor2
        except ValueError:
            print(f"Invalid sensor data: {data}")
            return np.zeros(4), np.zeros(4)  # Return safe fallback

    def read_force(self) -> Tuple[np.ndarray, np.ndarray]:
        sensor_data1, sensor_data2 = self.split_read()
        force_R, force_L = self.calibration.predict_force(sensor_data1, sensor_data2)
        return force_R, force_L

    def close(self) -> None:
        """Close the serial port"""
        print("Port Closed.")
        self.mcu.close()


class SensorForce():
    def __init__(self):
        # Load Model and Scaler
        sensor_cal_path = "./sensor_calibration"
        self.model_1 = self.load_model(f"{sensor_cal_path}/mlp_model_1.pth")
        self.scaler_1 = joblib.load(f"{sensor_cal_path}/scaler_1.pkl")
        self.model_2 = self.load_model(f"{sensor_cal_path}/mlp_model_2.pth")
        self.scaler_2 = joblib.load(f"{sensor_cal_path}/scaler_2.pkl")
        print("Sensor Calibration Model is loaded")

    def load_model(self, model_path):
        model = MLPModel()
        model.load_state_dict(torch.load(model_path))
        model.eval() # Set model to evaluation mode
        return model
    
    def predict_force(self, sensor1_data: np.ndarray, sensor2_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict forces using models"""
        def predict(sensor_values: np.ndarray, scaler, model) -> np.ndarray:
            scaled_values = scaler.transform(sensor_values.reshape(1,-1))
            tensor_values = torch.tensor(scaled_values, dtype=torch.float32)
            return model(tensor_values).detach().numpy()[0]

        force_R = predict(sensor1_data, self.scaler_1, self.model_1) # force_R1, force_R2, force_R3
        force_L = predict(sensor2_data, self.scaler_2, self.model_2) # force_L1, force_L2, force_L3
        return force_R, force_L



def main():
    np.set_printoptions(precision=1)
    start_time = time.time()

    fsensor = FingerSensor(port='/dev/ttyUSB0')
    try:
        while True:
            sensor_data1, sensor_data2 = fsensor.read_force()
            print(f"{time.time()-start_time:.3f} {sensor_data1} {sensor_data2}")
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Exiting program")
    except Exception as e:
        print(f"Error Exception: {e}")
    finally:
        fsensor.close()


if __name__ == "__main__":
    main()

