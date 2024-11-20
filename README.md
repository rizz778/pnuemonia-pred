Pneumonia Detection Using Convolutional Neural Networks
Overview
Pneumonia is a life-threatening respiratory condition that requires timely and accurate diagnosis. This project leverages deep learning techniques, specifically Convolutional Neural Networks (CNNs), to automate the detection of pneumonia using chest X-ray images. By assisting healthcare professionals, this tool aims to reduce diagnostic time and improve healthcare outcomes.

Features
Automated Detection: Predicts whether an X-ray image is normal or indicates pneumonia.
Deep Learning-Based: Uses CNN architecture trained on chest X-ray images for robust predictions.
User-Friendly Interface: Includes a Streamlit-based application for easy interaction.
Evaluation Metrics: Provides confusion matrix, sensitivity, specificity, and accuracy reports for evaluation.
Dataset
Source: The dataset is sourced from Kaggle’s Chest X-Ray Dataset.
Composition:
Training: Images labeled as 'NORMAL' and 'PNEUMONIA' (split into bacterial and viral pneumonia).
Validation: A subset of images for tuning model hyperparameters.
Testing: A separate set of images used for final evaluation.
Data Preprocessing:
Resized images to 180x180 pixels.
Normalized pixel values to the range [0, 1].
Applied data augmentation (e.g., rotation, flipping) to enhance generalization.
Model Architecture
Key Details
CNN Layers:
Conv2D: Feature extraction using 16 filters of size 3x3.
Pooling: MaxPooling2D layers to reduce spatial dimensions.
Dropout: Regularization to prevent overfitting.
Activation Functions: ReLU for non-linearity.
Dense Layers: Fully connected layers for classification.
Output Layer: Softmax for binary classification into 'NORMAL' and 'PNEUMONIA'.
Loss and Optimization
Loss Function: SparseCategoricalCrossEntropy.
Optimizer: Adam with learning rate scheduling using ReduceLROnPlateau.
Installation and Setup
Clone the Repository:
bash
Copy code
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
Create a Virtual Environment:
bash
Copy code
python -m venv env
source env/bin/activate  # For Linux/Mac
env\Scripts\activate  # For Windows
Install Dependencies:
bash
Copy code
pip install -r requirements.txt
Download Dataset:
Download the dataset from the Kaggle link provided above.
Place the dataset in the data/ folder in the following structure:
kotlin
Copy code
data/
├── train/
├── val/
├── test/
Usage
Notebook Testing
Run the provided Jupyter Notebook for:

Training the model from scratch.
Testing the model on new images.
Visualizing performance metrics such as confusion matrix and classification report.
Streamlit Application
Run the Streamlit app:
bash
Copy code
streamlit run app.py
Upload a chest X-ray image through the app.
View predictions and confidence scores.
Evaluation Metrics
Accuracy: 82.5% on the test set.
Confusion Matrix: Visualized to show true positives, true negatives, false positives, and false negatives.
Sensitivity (Recall): Measures the model's ability to detect pneumonia cases.
Specificity: Assesses the model's ability to detect normal cases.
ROC-AUC Curve: Demonstrates the tradeoff between true positive and false positive rates.
Limitations
Dataset bias may limit real-world generalizability.
Only binary classification (NORMAL vs. PNEUMONIA) is currently supported.
Moderate accuracy of 82.5% may need improvement for clinical deployment.
Future Directions
Expand to multi-class classification for broader lung diseases.
Test and deploy the model in real-world clinical environments.
Improve model explainability with techniques like Grad-CAM.
Integrate with mobile health applications for increased accessibility.
Dependencies
The project relies on the following libraries:

TensorFlow/Keras
NumPy
OpenCV
Matplotlib
Scikit-learn
Streamlit
To install all dependencies:

bash
Copy code
pip install -r requirements.txt
Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests with enhancements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.
