INTRODUCTION

Pneumonia, a severe lung infection, is a leading cause of death worldwide, particularly among children under five and the elderly. Early and accurate detection is crucial for effective treatment, but current diagnostic methods primarily rely on radiologists’ manual interpretation of chest X-rays, which is time-consuming, subjective, and prone to error, especially in resource-limited settings. This project addresses these challenges by developing an automated pneumonia detection system using convolutional neural networks (CNNs).
The primary objective of this project is to design a deep learning-based solution that can classify chest X-ray images into two categories: Normal and Pneumonia. By automating the detection process, this system aims to improve diagnostic accuracy, reduce the workload of healthcare professionals, and provide timely support in clinical decisions.
The proposed methodology leverages a custom CNN architecture to analyze X-ray images. The dataset, sourced from Kaggle’s “Chest X-Ray Images (Pneumonia)” collection, was preprocessed to enhance model performance. Preprocessing steps included resizing images, normalizing pixel values, and applying augmentation techniques like flipping and rotation to create a robust and generalized model. The CNN architecture was carefully designed with convolutional layers, max-pooling, batch normalization, and dropout to optimize feature extraction and prevent overfitting. A sparse categorical cross-entropy loss function and Adam optimizer were employed for training, with callbacks like ModelCheckpoint and ReduceLROnPlateau to fine-tune the learning process.
The trained model achieved an accuracy of 82.5%, with a high sensitivity for detecting pneumonia cases. The confusion matrix and classification report revealed balanced precision and recall, ensuring the system's reliability in distinguishing between normal and diseased lungs.
This AI-driven system has significant implications for healthcare. By enabling automated, accurate, and consistent analysis of chest X-rays, it can serve as a diagnostic aid for radiologists and clinicians, particularly in regions with limited access to skilled professionals. Moreover, its scalability and efficiency make it a valuable tool for mass screening programs and telemedicine applications.
In conclusion, the project demonstrates the potential of deep learning in revolutionizing medical diagnostics, paving the way for faster, more accurate, and accessible healthcare solutions.
DATASET SPECIFICATION

•	Dataset is Taken from Kaggle https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 
•	The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
•	Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.
•	For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.
 

DATASET PREPROCESSING
•  Resizing: All images were resized to a fixed size of 180x180 pixels to standardize input for the model.
•  Normalization: Pixel values were scaled to the range [0, 1] by dividing by 255. This helped in faster convergence during training.
•  Augmentation: Applied random transformations such as:
•	Horizontal flipping to simulate variations.
•	Rotation (e.g., ±15 degrees) to make the model invariant to orientations.
•	Zooming for robustness against varying image sizes.
MODEL DEVELOPMENT
•  Architecture Choice
•	A custom CNN architecture was built to ensure flexibility in optimizing for binary classification.
•	Additionally, VGG19 with transfer learning was explored to leverage pre-trained weights and improve accuracy.
•  Layers and Parameters Used
•	Convolutional Layers (Conv2D): Extract features such as edges, textures, and patterns.
o	Example: 16 filters of 3x3 size with ReLU activation.
•	Pooling Layers (MaxPooling2D): Reduce spatial dimensions while retaining significant features.
•	Batch Normalization: Normalize feature maps to improve training stability and performance.
•	Dropout Layers: Reduce overfitting by randomly setting a fraction of input units to zero.
•	Dense Layers:
o	Fully connected layers for high-level decision-making.
o	Final output layer with 2 neurons (NORMAL and PNEUMONIA) using softmax activation.
•  Loss Function
•	SparseCategoricalCrossentropy: Calculated the difference between the true labels and predictions.
•  Optimization Algorithm
•	Adam Optimizer: Adaptive optimization with learning rate adjustments.
MODEL TRAINING
1.	Configuration
o	Number of epochs: 20-50 (depending on convergence).
o	Batch size: 32 (balanced between computational efficiency and memory constraints).
2.	Callbacks Used
o	EarlyStopping: Monitored validation loss to stop training if it didn’t improve for a set number of epochs.
o	ModelCheckpoint: Saved the best-performing model weights during training.
o	ReduceLROnPlateau: Reduced the learning rate dynamically if validation loss plateaued, ensuring efficient learning.
3.	Metrics Tracked
o	Accuracy: The percentage of correctly classified images.
o	Loss: The error between predictions and true labels.
o	Sensitivity: The ability of the model to detect pneumonia cases.
o	Specificity: The ability of the model to correctly classify normal cases
Validation and Testing
•  Performance Evaluation Metrics
•	Confusion Matrix: Presented the true positives, true negatives, false positives, and false negatives.
•	Classification Report: Provided precision, recall, and F1 scores for both classes.
•	ROC Curve: Plotted the true positive rate (sensitivity) against the false positive rate to measure classifier performance.
•  Insights from Testing
•	Achieved balanced performance across both NORMAL and PNEUMONIA categories.
•	Identified areas for improvement, such as augmenting data to address potential biases or fine-tuning the architecture for higher accuracy. 






CONFUSION MATRIX AND CURVE FOR VGG-NET
CONFUSION MATRIX  AND CURVE FOR CUSTOM MODEL
FRONTEND OF THE MODEL









	
Key Takeaways of the Project
1.	Problem and Solution
o	Problem: Pneumonia is a critical global health issue, especially in vulnerable populations like children and the elderly. Diagnosis often relies on radiologists interpreting chest X-rays, which can be time-consuming, subjective, and prone to error in high-pressure environments.
o	Solution: A deep learning-based system using Convolutional Neural Networks (CNNs) was developed to automate pneumonia detection from chest X-ray images. This solution aims to assist healthcare professionals by providing reliable predictions quickly and consistently.

2.	Model Strengths
o	High Accuracy: The model achieved an accuracy of 82.5%, demonstrating its capability to distinguish between normal and pneumonia-affected chest X-rays.
o	Robustness: The use of data augmentation and transfer learning improved the model's generalization, reducing overfitting and enhancing performance on unseen data.
o	Deployability: The model’s lightweight architecture and compatibility with modern deployment frameworks make it suitable for real-world applications, such as integration into mobile apps or hospital systems.

3.	Impact on Healthcare
o	Improved Diagnostic Efficiency: By automating the initial screening process, the system can save radiologists significant time, enabling them to focus on more complex cases.
o	Scalability: The solution can be deployed in low-resource settings where radiologists may not always be available, bridging gaps in medical access.
o	Early Detection: Faster and accurate detection of pneumonia can lead to earlier interventions, potentially reducing morbidity and mortality rates.

In summary, this project showcases how AI-powered systems can complement human expertise in healthcare, offering scalable, efficient, and impactful solutions to pressing medical challenges.


