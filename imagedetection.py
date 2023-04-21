import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage import io, transform

input_dir = 'data'
image_size = (150, 150)

# Load and prepare data
data = []
labels = []

for label_dir in os.listdir(input_dir):
    if label_dir == 'Label':
        continue
    for image_file in os.listdir(os.path.join(input_dir, label_dir)):
        image_path = os.path.join(input_dir, label_dir, image_file)
        image = io.imread(image_path)
        image = transform.resize(image, image_size, mode='constant')
        data.append(image.flatten())
        labels.append(label_dir)

data = np.array(data)
labels = np.array(labels)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Train classifier
classifier = SVC(probability=True)

param_grid = {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001]}
grid_search = GridSearchCV(classifier, param_grid, n_jobs=-1)
grid_search.fit(x_train, y_train)

# Test performance
y_pred = grid_search.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print('{}% of samples were correctly classified'.format(str(accuracy * 100)))

# Save model
with open('models/model.p', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)
