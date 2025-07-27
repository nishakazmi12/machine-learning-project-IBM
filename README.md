# Loan Prediction Machine Learning Project

## Overview  
This project implements a machine learning pipeline to predict loan approval status using real-world data. The pipeline includes data preprocessing, feature engineering, dimensionality reduction with PCA, and classification with K-Nearest Neighbors (KNN) and Decision Tree classifiers.

---

## Features  
- Data loading from IBM Cloud Object Storage  
- Handling missing data with imputation  
- Encoding categorical features with One-Hot and Label Encoding  
- Feature scaling with StandardScaler  
- Dimensionality reduction using Principal Component Analysis (PCA)  
- Classification using KNN and Decision Tree with hyperparameter tuning  
- Performance evaluation using F1 Score, Jaccard Index, Log Loss, and Confusion Matrix  
- Visualization of decision boundaries and metric trends  

---

## Tech Stack  
- Python  
- Pandas, NumPy, Matplotlib  
- Scikit-learn (sklearn) for preprocessing, modeling, and evaluation  
- IBM Cloud Object Storage SDK for data access  

---

## Project Structure  

- `loan_train.csv` — dataset loaded from IBM Cloud Object Storage  
- Jupyter Notebook / Python scripts containing:  
  - Data preprocessing and imputation  
  - Feature encoding and scaling  
  - PCA dimensionality reduction  
  - Training and evaluating KNN and Decision Tree classifiers  
  - Visualization of results and decision boundaries  

---

## How to Run  

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <oath_to_folder>
````

2. Set your IBM Cloud Object Storage API key and endpoint credentials in the script or environment variables.

3. Run the main notebook or Python script to execute the pipeline.

---

## Evaluation Metrics

* **F1 Score:** Harmonic mean of precision and recall
* **Jaccard Index:** Intersection over union metric for classification
* **Log Loss:** Penalizes false classifications with probability outputs
* **Confusion Matrix:** Visualization of true vs predicted labels

---

## Visualizations

* Metric scores plotted over different hyperparameter values for KNN (`k`) and Decision Tree (`max_depth`)
* Decision boundary plots for KNN classifier in 2D PCA space
* Confusion matrix heatmaps with labels

---

## Results Summary

* KNN achieved optimal performance at `k=9` neighbors with good F1 and Jaccard scores
* Decision Tree performance was evaluated for depths 2 to 9, showing trade-offs in model complexity and accuracy
* PCA reduced feature space to 2 principal components explaining significant variance

---

## Contact

For questions or collaboration, please contact me via GitHub or LinkedIn.

---

*Happy Modeling!*

```

