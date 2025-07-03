# 🤖 Support Vector Machines (SVM) 

This repository contains an implementation of **Support Vector Machines (SVM)** for binary classification using the **Breast Cancer Dataset**. The task is part of the AI & ML Internship Program.

---

## 🎯 Objective

- Apply SVM with **Linear** and **RBF** kernels
- Understand kernel functions and decision boundaries
- Tune hyperparameters like **C** and **gamma**
- Evaluate model using **accuracy**, **confusion matrix**, and **cross-validation**

---

## 🛠 Tools & Libraries

- Python
- Scikit-learn (`sklearn`)
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 📁 Dataset

- **Breast Cancer Wisconsin Dataset** (from `sklearn.datasets`)
- 2 Classes: Malignant (0), Benign (1)
- 30 Numeric Features (e.g., mean radius, mean texture, etc.)

---

## 🧪 Steps Performed

1. **Loaded and explored** the breast cancer dataset
2. **Normalized** features using `StandardScaler`
3. **Split** data into training and testing sets
4. **Trained two SVM models**:
   - Linear Kernel (`kernel='linear'`)
   - RBF Kernel (`kernel='rbf'`)
5. **Evaluated** each model:
   - Accuracy Score
   - Classification Report
   - Confusion Matrix
6. **Performed cross-validation** to check model stability
7. **Visualized decision boundary** using first two features

---

## 📊 Results

| Model        | Test Accuracy | Cross-Validation Accuracy |
|--------------|---------------|----------------------------|
| Linear SVM   | ~97%          | ~96.5%                     |
| RBF SVM      | ~98.2%        | ~97.2%                     |

> Note: Results may vary slightly depending on train-test split.

---

## 📈 Visual Outputs

- Confusion Matrix (for both Linear and RBF kernels)
![Screenshot 2025-07-03 202715](https://github.com/user-attachments/assets/1470e7d8-d34d-4614-8957-a0b215f4bdb4)

- Decision Boundary plot using 2D projection (first 2 features)
![Screenshot 2025-07-03 202735](https://github.com/user-attachments/assets/97a4f28a-50c2-4aa1-9b66-06adecf0f642)
![Screenshot 2025-07-03 202747](https://github.com/user-attachments/assets/3e752eb0-4665-44f9-b780-f2c818b176f4)



