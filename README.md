
---

# 🖐️ Phantom Touch— Turn Your Laptop into a Touchscreen (No Touch Required!)

**Phantom Touch** lets you control your mouse and draw **just by waving your hand in front of your camera**.
Using **Computer Vision** + **Hand Tracking**, it detects your gestures and simulates touch/mouse events — so you can draw in **any app** (Paint, Photoshop, browsers, etc.) without a touchscreen.

---

## ✨ Features

* 🎯 **Real-time hand tracking** with [MediaPipe](https://google.github.io/mediapipe/)
* 🖌 **Draw anywhere** — works in any drawing software or webpage
* 🤏 **Pinch gesture** to draw, release to stop
* 🖱 **Full mouse control** — move cursor with your index finger
* 🪶 Smooth motion with coordinate mapping
* 🖼 Optional **on-screen video preview** for debugging



## 🛠 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Kashina69/Phantom-Touch.git
cd Phantom-Touch
```

### 2️⃣ Install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
python main.py
```

---

## 🎮 How to Use

1. Run `main.py`
2. Keep your hand visible to your webcam.
3. **Index finger up** → move the cursor.
4. **Pinch (index + thumb)** → hold to draw, release to stop.
5. Press **`Q`** to quit.

---

## ⚙ Configuration

You can tweak:

* **Gesture sensitivity** (pinch distance)
* **Tracking area** (frame margins)
* **Smoothing** (average positions to reduce jitter)

*(Open `main.py` and adjust variables at the top.)*

---

## 📦 Dependencies

* [OpenCV](https://opencv.org/) — Image & video processing
* [MediaPipe](https://google.github.io/mediapipe/) — Hand landmark detection
* [PyAutoGUI](https://pyautogui.readthedocs.io/) — Mouse control
* [NumPy](https://numpy.org/) — Math operations

---

## 🚀 Future Plans

* ✋ Support for **multi-hand gestures**
* 🎨 In-app drawing canvas mode
* 📏 Calibration for better accuracy
* 🖱 Right/left click gestures
* 📌 Adjustable brush size via gestures

---

## 🤝 Contributing

Contributions are welcome!
If you’d like to improve gesture detection, add features, or optimize performance:

1. Fork the repo
2. Create a feature branch
3. Submit a pull request

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify.

---

## 💡 Inspiration

This project is inspired by the idea of making **touchless interaction** possible for everyone —
turning any laptop or PC into a futuristic gesture-controlled device 🚀.

---