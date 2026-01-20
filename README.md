✍️ Handwritten Digit Recognition (MNIST-Style)

This project is a handwritten digit recognition web application inspired by the classic MNIST demonstration.

Users can draw a digit (0–9) in the browser, and the application predicts the digit using a simple neural network, following the exact same preprocessing logic used in MNIST training.

The goal of this project is correctness, explainability, and consistency, not unnecessary model complexity.

⸻

🔍 What This Project Does
	•	Allows users to draw digits using a mouse
	•	Converts the drawing into a 28×28 MNIST-style image
	•	Uses a feedforward neural network to predict the digit
	•	Displays prediction probabilities
	•	Mimics the behavior of the original MNIST neural network demo

⸻

🧠 Model Architecture

This project intentionally uses a simple feedforward neural network, not a CNN.

Network Structure

Input Layer   : 784 neurons (28 × 28 image)
Hidden Layer  : 30 neurons (Sigmoid activation)
Output Layer  : 10 neurons (Digits 0–9)

Why this model?
	•	This architecture is identical to the original MNIST example
	•	Easy to understand and explain
	•	Very reliable when preprocessing is correct
	•	Avoids overfitting and data mismatch issues

⸻

🖼️ Image Preprocessing (Core Logic)

The accuracy of this project comes from correct preprocessing, not model complexity.

Each drawn digit goes through the following steps:
	1.	Convert drawing to grayscale
	2.	Invert colors (white digit on black background)
	3.	Extract the bounding box around the digit
	4.	Resize the digit to 20×20 pixels
	5.	Place it in the center of a 28×28 canvas
	6.	Center the digit using center of mass of pixels
	7.	Normalize pixel values to range 0–1

This process ensures that training data and prediction input are identical.

⸻

📁 Project Structure

handwritten_digit_recognition/
├── train.py            # Trains the neural network on MNIST
├── app.py              # Streamlit web application
├── requirements.txt    # Python dependencies
├── digit_model.keras   # Saved trained model
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

Run the training script once:

python train.py

This will:
	•	Download the MNIST dataset
	•	Train the neural network
	•	Save the model as:

digit_model.keras


⸻

▶️ Run the Application

Start the web application:

streamlit run app.py

Then open the displayed URL in your browser.

⸻

📊 Example Output

After drawing a digit, the application shows predictions like:

Digit Prediction:
2 : 0.999998
3 : 0.000002
6 : 0.000000

It also displays the processed 28×28 MNIST-style image used for prediction.