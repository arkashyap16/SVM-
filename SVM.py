# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='label')

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SVM with Linear Kernel
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

print("Linear Kernel SVM:")
print("Accuracy:", svm_linear.score(X_test, y_test))
print(classification_report(y_test, y_pred_linear))

# Confusion Matrix
cm_linear = confusion_matrix(y_test, y_pred_linear)
disp = ConfusionMatrixDisplay(cm_linear, display_labels=data.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Linear Kernel")
plt.show()

# SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

print("RBF Kernel SVM:")
print("Accuracy:", svm_rbf.score(X_test, y_test))
print(classification_report(y_test, y_pred_rbf))

# Confusion Matrix
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
disp = ConfusionMatrixDisplay(cm_rbf, display_labels=data.target_names)
disp.plot(cmap='Oranges')
plt.title("Confusion Matrix - RBF Kernel")
plt.show()

# Cross-Validation Scores
cv_linear = cross_val_score(svm_linear, X_scaled, y, cv=5)
cv_rbf = cross_val_score(svm_rbf, X_scaled, y, cv=5)

print("Cross-Validation Accuracy (Linear Kernel):", np.round(cv_linear.mean(), 4))
print("Cross-Validation Accuracy (RBF Kernel):", np.round(cv_rbf.mean(), 4))

# Optional: Visualize decision boundary using 2D data
# Let's take only first 2 features for visualization
X_2d = X_scaled[:, :2]
X_train2d, X_test2d, y_train2d, y_test2d = train_test_split(X_2d, y, test_size=0.2, random_state=42)

svm_vis = SVC(kernel='rbf', C=1, gamma='scale')
svm_vis.fit(X_train2d, y_train2d)

# Create meshgrid
h = 0.02
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title("SVM Decision Boundary (RBF Kernel, 2 Features)")
plt.grid(True)
plt.show()
