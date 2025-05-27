import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
import pickle

# Load dataset
matrix = np.ones((9, 9))
matrix[0,1:10] = 0
matrix[1,3:10] = 0
matrix[2,7:10] = 0
y = matrix.ravel()
m,n = np.meshgrid(np.arange(1, 10, 1), np.arange(1, 10, 1))
X = np.c_[m.ravel(),n.ravel()]/10
X_train = X
y_train = y

# Create SVM classifier
svm_classifier = SVC(kernel='rbf', C=1, probability=True)
svm_classifier.fit(X_train, y_train)

# Save model
with open('./params/SVM_model.pkl', 'wb') as f:
    pickle.dump(svm_classifier, f)

# Calculate accuracy
y_pred = svm_classifier.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)

# Prepare grid for visualization
x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001),
                     np.arange(y_min, y_max, 0.001))
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Get probability heatmap data
Z_prob = svm_classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z_prob = Z_prob[:,1].reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 8))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 36
X_train = X_train*0.8+0.1
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='crimson', label='fail',s=200)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='mediumseagreen', label='feasible',s=200)
plt.ylabel('Side length (mm)')
plt.xlabel('Height (mm)')
ax = plt.gca()
ax.tick_params(axis='both', length=8, width=1.5)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
xticks_positions = [0.2, 0.4, 0.6, 0.8]
ax.set_xticks(xticks_positions)
plt.savefig('./original data/plot/svm3241.png', bbox_inches='tight', transparent=True,dpi=300)
plt.show()
plt.close()

# Plot probability heatmap
plt.figure(figsize=(10,8))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 36
plt.pcolormesh(xx, yy, Z_prob, cmap=plt.cm.Blues_r)
cbar = plt.colorbar()
cbar.set_label('Feasible Probability')
plt.ylabel('Side length (mm)')
plt.xlabel('Height (mm)')
ax = plt.gca()
ax.tick_params(axis='both', length=8, width=1.5)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
plt.savefig('./original data/plot/svm324.png', bbox_inches='tight', transparent=True,dpi=300)
plt.show()

print(accuracy)