# Embedded-AI
This project implements a real-time face mask detection system using a deep learning model in Python with OpenCV and TensorFlow, combined with hardware feedback via STM32 microcontroller. The system detects whether a person is wearing a mask and sends a signal over UART to an STM32G431MBT6 board, which controls green (Mask) and red (No Mask) LEDs.
# 😷 Face Mask Detection with STM32 UART LED Feedback

## 📌 Overview
This project is a real-time face mask detection system that uses deep learning for detecting whether a person is wearing a face mask or not. It integrates with an STM32G431MBT6 microcontroller to provide **hardware feedback via LEDs** using UART communication.

- ✅ **Green LED** → Mask detected  
- ❌ **Red LED** → No Mask detected

---

## 🧠 Features
- Real-time mask detection using webcam
- Shows confidence score (e.g., `Mask: 98.45%`)
- Sends result via UART to STM32
- STM32 blinks appropriate LED based on detection

---

## 🛠 Tech Stack

### 💻 Software
- Python
- TensorFlow / Keras
- OpenCV
- pySerial
- STM32CubeIDE

### 🔧 Hardware
- STM32G431MBT6 microcontroller
- LEDs connected to PB0 (Green) and PB1 (Red)
- USB–UART connection (USART1)

---
