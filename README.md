✍️ Handwritten Digit Recognition (MNIST-Style)

This project is a handwritten digit recognition web application inspired by the classic MNIST neural network demonstration.

Users can draw a digit (0–9) in the browser, and the application predicts the digit along with confidence scores using a simple, explainable neural network.

⸻

🖼️ Demo

Below is a live example of the application predicting a handwritten digit correctly, along with probability scores and the processed MNIST-style input.


⸻

🔍 What This Project Does
	•	Lets users draw digits using a mouse
	•	Converts drawings into MNIST-style 28×28 images
	•	Uses a feedforward neural network (MLP) to predict digits
	•	Displays top predictions with probabilities
	•	Matches the behavior of the original MNIST C++ demo

⸻

🧠 Model Architecture

This project intentionally avoids complex CNNs and uses a classic feedforward neural network, exactly like the original MNIST example.

Network Structure

Input Layer   : 784 neurons (28 × 28 image)
Hidden Layer  : 30 neurons (Sigmoid activation)
Output Layer  : 10 neurons (Digits 0–9)

Why this model?
	•	Simple, interpretable, and reliable
	•	No training–prediction mismatch
	•	Excellent results with correct preprocessing

⸻

🖼️ Image Preprocessing (Key to Accuracy)

Every drawn digit goes through the same preprocessing used during training:
	1.	Convert to grayscale
	2.	Invert colors (white digit on black background)
	3.	Extract bounding box of the digit
	4.	Resize digit to 20×20 pixels
	5.	Place it at the center of a 28×28 canvas
	6.	Align using center of mass of pixels
	7.	Normalize pixel values to 0–1

This ensures training and inference data are identical, which is why predictions are accurate.

⸻

📁 Project Structure

handwritten_digit_recognition/
├── train.py            # Model training script
├── app.py              # Streamlit web application
├── requirements.txt    # Python dependencies
├── digit_model.keras   # Trained model (generated)
├── assets/
│   └── demo.png        # Demo screenshot
└── README.md


⸻

📦 Installation

Create and activate a virtual environment (recommended):

python -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt


⸻

🏋️ Train the Model

Run once to train the model on the MNIST dataset:

python train.py

This will generate:

digit_model.keras


⸻

▶️ Run the Application

Start the web app:

streamlit run app.py

Open the displayed URL in your browser.

⸻

📊 Example Output

When a digit is drawn, the app displays results like:

Digit Prediction
9 : 0.994924
4 : 0.981900
0 : 0.976247

It also shows the processed 28×28 MNIST-style input image used for prediction.