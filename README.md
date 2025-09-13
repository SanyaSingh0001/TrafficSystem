A hardware-integrated AI simulation system designed to optimize urban traffic flow using Reinforcement Learning (Q-Learning), computer vision, and IoT, reducing average commute time by dynamically controlling signal timings.

✨ Highlights

· 🤖 AI-Powered: Uses a Q-Learning RL model to make intelligent traffic signal decisions.
· 🎯 Simulation & Real-World Bridge: Connects the SUMO traffic simulator to physical hardware (ESP32/Arduino).
· 👁️ Computer Vision: Includes live camera feed detection for real-time vehicle counting.
· 🔧 Hardware Integration: Controls RGB LED traffic lights and 7-segment displays based on simulation data.
· 📊 Data-Driven: Analyzes traffic patterns to predict and mitigate congestion bottlenecks.

📁 Repository Structure

├── 📄 RL_MODEL.py                 # Reinforcement Learning (Q-Learning) model training and inference
├── 📄 SUMO_TO_ARDUINO.ipynb       # Jupyter Notebook: Main bridge between SUMO sim and Arduino
├── 📄 Arduino.ino                 # Code for ESP32/Arduino to control LEDs & display
├── 📁 Camera-detection/           # OpenCV scripts for real-time vehicle detection

⚙️ How It Works

1. Simulation (SUMO)

The SUMO simulator generates realistic urban traffic scenarios. Data like vehicle count, speed, and queue length are extracted in real-time using the TraCI API.

2. AI Decision Making (RL Model)

The RL_MODEL.py implements a Q-Learning algorithm. It takes the live traffic data from SUMO, predicts potential congestion, and decides the optimal signal timing (Green/Red light duration) to minimize overall wait time.

3. Hardware Control (Arduino/ESP32)

The decision from the AI model is sent via serial communication to the microcontroller (Arduino.ino), which controls:

· 🚦 RGB LEDs acting as traffic signals.
· ⏰ 7-Segment Displays showing countdown timers.

4.  Real-Time Camera Feed

The Camera-detection module uses OpenCV and Haar Cascades to count vehicles from a live camera feed, which can be used as an alternative input to the AI model.


🚀 Getting Started

Prerequisites

· Python 3.8+
· Arduino IDE
· SUMO (Simulation of Urban Mobility)
· An ESP32 or Arduino Board
· RGB LEDs, 7-Segment Displays, Resistors

Installation & Setup

1. Clone the repo
   ```bash
   git clone https://github.com/SanyaSingh0001/TrafficSystem
   cd TrafficSystem
   ```
2. Install Python dependencies
   ```bash
   pip install -r requirements.txt
   # Main libraries: traci, numpy, opencv-python, pyserial, pandas
   ```
3. Set up SUMO
   · Download and install SUMO from here.
   · Ensure the SUMO_HOME environment variable is set.
4. Upload the Arduino Code
   · Open Arduino.ino in the Arduino IDE.
   · Select your board  and port.
   · Upload the code.
5. Run the System
   · Run the Jupyter Notebook SUMO_TO_ARDUINO.ipynb cell by cell.
   · Alternatively, run the Python scripts:
     ```bash
     python RL_MODEL.py


🧠 The AI Model: Reinforcement Learning

Our Q-Learning model is trained to maximize the reward function, which is based on:

· Negative reward for vehicles waiting at a red light.
· Positive reward for vehicles passing through a green light.
· High penalty for emergency vehicles being stuck.

The state includes the number of vehicles on each lane, and the actions are which signal to turn green.


🎯 Future Implementations

· Implement DQN (Deep Q-Network) for more complex intersections.
· Integrate V2X (Vehicle-to-Everything) communication.
· Develop a web dashboard for real-time monitoring.
· Use GPS data for city-wide traffic prediction.


<div align="center">

Built with passion for Smart India Hackathon (SIH)

⚡ Innovation | 🤖 Automation | 🚀 Technology

</div>
     
