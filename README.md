# Credit Card Fraud Detection using Anomaly Detection

This project leverages unsupervised learning techniques, particularly **Isolation Forest**, to detect fraudulent credit card transactions. The focus is on identifying anomalies in a highly imbalanced dataset using statistical insights and visual analytics.

## Features
- Data cleaning and transformation.
- Exploratory Data Analysis (EDA) with visualizations.
- Application of **Isolation Forest** for anomaly detection.
- Evaluation of model performance using precision, recall, and F1-score.
- Interactive visualizations of outliers and feature distributions.
  
## Installation
To run this project locally, ensure you have Python installed. It's recommended to use a virtual environment.
```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection-anomaly.git
cd fraud-detection-anomaly

# (Optional) Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## Usage
This project is implemented using Jupyter Notebook. Follow the steps below to run the notebook:
	1.	Launch Jupyter Notebook or JupyterLab.
	2.	Open the file:
Credit Card Fraud Detection using Anomaly Detection Technique.ipynb
	3.	Run the cells sequentially to:
	•	Load and preprocess the data.
	•	Perform exploratory analysis.
	•	Apply and evaluate the Isolation Forest model.

### Data Preparation
The dataset used is:
	•	creditcard.csv:
A real-world dataset containing credit card transactions made by European cardholders in September 2013.

Ensure this file is placed in a data/ directory or update the path in the notebook accordingly.

| Model             | Type         | Purpose                          |
|------------------|--------------|----------------------------------|
| Isolation Forest | Unsupervised | Detect anomalous (fraudulent) transactions |

### Evaluation Metrics:
| Metric    | Description                                 |
|-----------|---------------------------------------------|
| Precision | Measures how many detected frauds are true  |
| Recall    | Measures how many actual frauds were found  |
| F1-Score  | Harmonic mean of precision and recall       |

### Visualizations
- **Feature Distributions**: Helps understand variable importance.
- **PCA Plot**: Visual clustering patterns.
- **Fraud vs Normal Distribution**: Visual identification of anomalies.

### Custom Functions
- **plot_confusion_matrix()**: For detailed error analysis.
- **evaluate_model()**: Precision, recall, F1-score calculator.
- **plot_pca_clusters()**: Visualizes anomaly clusters in 2D space.

### Key Learnings
- Anomaly detection is crucial for highly imbalanced datasets like fraud detection.
- Unsupervised models like Isolation Forest can perform well without labeled data.
- Evaluation should focus on recall and precision, not accuracy.
- Visual techniques (like PCA) aid in interpreting high-dimensional anomaly patterns.

 ### Acknowledgments
 
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python) ![Pandas](https://img.shields.io/badge/Pandas-1.x-150458?logo=pandas) ![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy) ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-3776AB?logo=python) ![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-579ACA?logo=python) ![SciPy](https://img.shields.io/badge/SciPy-1.x-8CAAE6?logo=scipy) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?logo=scikit-learn) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow) ![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras) ![pickle](https://img.shields.io/badge/pickle-serialization-green) ![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
