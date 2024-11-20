# Pneumonia Detection System

## 📖 Overview
Pneumonia is a leading cause of mortality worldwide, especially in children and the elderly. Its diagnosis often relies on subjective interpretations of chest X-rays, which can lead to errors and delays in treatment. This project aims to automate pneumonia detection using Convolutional Neural Networks (CNNs) to improve accuracy and ease the workload of radiologists.

---

## 🎯 Objectives
- Automate the detection of pneumonia from chest X-rays using deep learning.
- Improve diagnostic accuracy and consistency.
- Provide a scalable and deployable solution for healthcare institutions.

---

## 📂 Dataset
- **Source**: [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets).
- **Composition**:
  - Normal: 1341 images.
  - Pneumonia: 3875 images.
  - Splits: Training, Validation, Testing.

---

## 🛠️ Tools and Libraries
- **Programming Language**: Python
- **Libraries**: 
  - TensorFlow, Keras
  - NumPy, Pandas, Matplotlib, Seaborn
  - OpenCV, PIL
- **Other Tools**:
  - Jupyter Notebook for development.
  - Streamlit for deployment and user interface.

---

## 🔧 Model Development
1. **Architecture**:
   - Custom CNN with Conv2D, MaxPooling, and Dropout layers.
   - Activation Function: ReLU.
2. **Hyperparameters**:
   - Optimizer: Adam.
   - Loss Function: SparseCategoricalCrossentropy.
   - Learning Rate Scheduler: ReduceLROnPlateau.
3. **Data Augmentation**:
   - Rotation, flipping, and zoom transformations.

---

## 🚀 Training
- **Epochs**: 25
- **Batch Size**: 32
- **Callbacks**:
  - EarlyStopping to prevent overfitting.
  - ModelCheckpoint for saving the best-performing model.
- **Performance Metrics**:
  - Accuracy, Sensitivity, Specificity.

---

## 📊 Evaluation
1. **Confusion Matrix**:
   - Shows the classification performance across Normal and Pneumonia classes.
2. **Classification Report**:
   - Precision, Recall, F1-Score.
3. **ROC-AUC Curve**:
   - Demonstrates the tradeoff between sensitivity and specificity.

---

## 💡 Key Takeaways
- **Accuracy**: Achieved 82.5% on test data.
- **Strengths**:
  - High sensitivity in detecting pneumonia.
  - Lightweight and deployable on low-resource systems.
- **Impact**: Assists radiologists by providing a second opinion, improving diagnosis speed.

---

## ❌ Limitations
- Dataset imbalance: Fewer Normal cases compared to Pneumonia.
- Struggles with images containing artifacts or poor contrast.

---

## 🔮 Future Directions
- Use larger, more diverse datasets to improve generalization.
- Explore ensemble models for better accuracy.
- Integrate with real-time systems like hospital management software.

---

## 🖥️ Deployment
1. **Interface**:
   - Built with Streamlit for an interactive user experience.
   - Allows users to upload chest X-rays and receive diagnostic predictions.
2. **Steps to Run**:
   - Clone the repository.
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Start the Streamlit app:
     ```bash
     streamlit run app.py
     ```

---

## 📈 Performance Metrics
| Metric          | Value   |
|------------------|---------|
| Accuracy         | 82.5%   |
| Sensitivity      | 84.7%   |
| Specificity      | 80.2%   |

---
