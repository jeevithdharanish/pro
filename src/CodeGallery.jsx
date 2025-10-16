import { Copy } from 'lucide-react';
import { useState } from 'react';

export default function CodeGallery() {
  const codes = [
    {
      title: "1️ Date / Time / User Info",
      filename: "date_time_user.sh",
      code: `1
import numpy as np

# 1D Arrays
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

print("1D Arrays:")
print("a:", a)
print("b:", b)

# Basic arithmetic
print("\nAddition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)
print("Power:", a ** 2)
print("Modulus:", a % 2)

# Aggregation
print("\nSum of a:", np.sum(a))
print("Mean of b:", np.mean(b))
print("Max of a:", np.max(a))
print("Min of b:", np.min(b))

# 2D Arrays
c = np.array([[1, 2], [3, 4]])
d = np.array([[4, 3], [2, 1]])

print("\n2D Arrays:")
print("c:\n", c)
print("d:\n", d)

# Element-wise operations
print("\nElement-wise addition:\n", c + d)
print("Element-wise multiplication:\n", c * d)

# Matrix multiplication (dot product)
print("Matrix multiplication:\n", np.dot(c, d))

# Transpose
print("Transpose of c:\n", c.T)

# Broadcasting
e = np.array([1, 2])
f = 10
print("\nBroadcasting example:")
print("e + f:", e + f)



2
import pandas as pd

def findS_algorithm(filename):
    # Load dataset
    data = pd.read_csv(filename)
    print("Training Data:\n", data, "\n")
    
    # Get features and target
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values
    
    # Initialize the most specific hypothesis
    hypothesis = ['0'] * features.shape[1]
    
    # Apply FIND-S
    for i, instance in enumerate(features):
        if target[i].lower() == "yes":  # Only consider positive examples
            for j in range(len(hypothesis)):
                if hypothesis[j] == '0':
                    hypothesis[j] = instance[j]
                elif hypothesis[j] != instance[j]:
                    hypothesis[j] = '?'
    
    print("Most Specific Hypothesis Found by FIND-S Algorithm:\n", hypothesis)

# Example run
# Save your training dataset as 'training.csv' in the same folder
findS_algorithm("training.csv")
Sky,AirTemp,Humidity,Wind,Water,Forecast,EnjoySport
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes


3
import pandas as pd

def candidate_elimination(filename):
    # Load dataset
    data = pd.read_csv(filename)
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    print("Training Data:\n", data, "\n")

    # Extract features (X) and target (Y)
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values

    # Initialize S (specific boundary) as the first positive example
    S = None
    for i, val in enumerate(target):
        if val.lower() == "yes":
            S = list(features[i])
            break

    # Initialize G (general boundary)
    G = [['?'] * len(S)]

    # Process training examples
    for i, instance in enumerate(features):
        if target[i].lower() == "yes":  # Positive example
            for j in range(len(S)):
                if S[j] != instance[j]:
                    S[j] = '?'
            # Remove from G any inconsistent hypotheses
            G = [g for g in G if all(g[j] == '?' or g[j] == instance[j] for j in range(len(g)))]

        else:  # Negative example
            new_G = []
            for g in G:
                for j in range(len(S)):
                    if g[j] == '?':
                        if S[j] != '?':
                            g_new = g.copy()
                            g_new[j] = S[j]
                            if all(g_new[k] == '?' or g_new[k] == features[i][k] for k in range(len(S))):
                                continue  # still covers negative, discard
                            new_G.append(g_new)
            G = new_G

    print("Final Specific Hypothesis (S):", S)
    print("Final General Hypotheses (G):", G)

# Example run
candidate_elimination("ex2.csv")
Sky,AirTemp,Humidity,Wind,Water,Forecast,EnjoySport
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes


4
# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree Classifier using ID3 (Entropy criterion)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Predicting a new sample
new_sample = [[5.0, 3.4, 1.5, 0.2]]  # Example new data
prediction = clf.predict(new_sample)
print(f"Predicted class for {new_sample} is: {class_names[prediction[0]]}")

# Plotting the decision tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
plt.title("Decision Tree using ID3 Algorithm")
plt.show()

# Optional: Display feature importance
importance = pd.DataFrame({'Feature': feature_names, 'Importance': clf.feature_importances_})
print("\nFeature Importances:")
print(importance.sort_values(by='Importance', ascending=False))



5.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1,1)

# One-hot encode output
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Network parameters
input_neurons = 4
hidden_neurons = 5
output_neurons = 3
lr = 0.01
epochs = 500

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(input_neurons, hidden_neurons) * 0.1
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.randn(hidden_neurons, output_neurons) * 0.1
b2 = np.zeros((1, output_neurons))

# Training
for _ in range(epochs):
    # Forward pass
    hidden = sigmoid(np.dot(X_train, W1) + b1)
    output = softmax(np.dot(hidden, W2) + b2)

    # Backpropagation
    d_output = output - y_train
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden)

    # Update weights
    W2 -= lr * hidden.T.dot(d_output)
    b2 -= lr * np.sum(d_output, axis=0, keepdims=True)
    W1 -= lr * X_train.T.dot(d_hidden)
    b1 -= np.sum(d_hidden, axis=0, keepdims=True)

# Testing
hidden_test = sigmoid(np.dot(X_test, W1) + b1)
output_test = softmax(np.dot(hidden_test, W2) + b2)
y_pred = np.argmax(output_test, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict a new sample
new_sample = np.array([[5.0, 3.4, 1.5, 0.2]])
hidden_new = sigmoid(np.dot(new_sample, W1) + b1)
output_new = softmax(np.dot(hidden_new, W2) + b2)
predicted_class = np.argmax(output_new)
print(f"Predicted class for {new_sample} is: {predicted_class}")


6.
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ---------------------------
# 1. Load CSV dataset
# ---------------------------
# Replace 'data.csv' with your file path
data = pd.read_csv('ex6.csv')

# Assume all columns are features
X = data.values

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 2. EM Algorithm (Gaussian Mixture)
# ---------------------------
n_clusters = 3  # Set number of clusters, adjust according to your dataset

gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

# ---------------------------
# 3. k-Means Clustering
# ---------------------------
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# ---------------------------
# 4. Compare clustering results
# ---------------------------
# Silhouette score (higher is better)
silhouette_em = silhouette_score(X_scaled, gmm_labels)
silhouette_km = silhouette_score(X_scaled, kmeans_labels)

print(f"Silhouette Score - EM Algorithm: {silhouette_em:.3f}")
print(f"Silhouette Score - k-Means: {silhouette_km:.3f}")

# Optional: Adjusted Rand Index if true labels exist in CSV
if 'label' in data.columns:  # column named 'label'
    true_labels = data['label'].values
    ari_em = adjusted_rand_score(true_labels, gmm_labels)
    ari_km = adjusted_rand_score(true_labels, kmeans_labels)
    print(f"Adjusted Rand Index - EM: {ari_em:.3f}")
    print(f"Adjusted Rand Index - k-Means: {ari_km:.3f}")

# ---------------------------
# 5. Print cluster assignments
# ---------------------------
print("\nCluster assignments (first 10 samples):")
print("EM labels:     ", gmm_labels[:10])
print("k-Means labels:", kmeans_labels[:10])
x,y
1.0,2.0
1.2,1.8
0.8,2.2
5.0,8.0
5.2,7.8
4.8,8.2
9.0,1.0
9.2,1.2
8.8,0.8


7.
from google.colab import files
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---- STEP 1: Upload Image ----
print(" Please upload an image (e.g., Blank_image.jpg):")
uploaded = files.upload()  # Choose file from your computer

# Get uploaded file name dynamically
image_path = list(uploaded.keys())[0]
print(f" Image uploaded successfully: {image_path}")

# ---- STEP 2: Load Pre-trained Model ----
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()

# ---- STEP 3: Open and Convert Image ----
image = Image.open(image_path)
image_tensor = F.to_tensor(image)

# ---- STEP 4: Run Object Detection ----
with torch.no_grad():
    outputs = model([image_tensor])

boxes = outputs[0]['boxes']
labels = outputs[0]['labels']
scores = outputs[0]['scores']

# ---- STEP 5: Filter Detections ----
threshold = 0.8
filtered = [(box, label, score) for box, label, score in zip(boxes, labels, scores) if score > threshold]

# ---- STEP 6: Show Results ----
if not filtered:
    print(" No object detected.")
else:
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)

    for box, label, score in filtered:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'{label.item()}:{score:.2f}',
                color='yellow', fontsize=12, weight='bold', backgroundcolor='black')

    plt.axis('off')
    plt.show()



8.
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# Parameters
vocab_size = 10000
maxlen = 200

# 1. Load and preprocess the IMDb dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# 2. Build a simple LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output: 0 = negative, 1 = positive
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train the model
model.fit(x_train, y_train, epochs=2, batch_size=64, validation_split=0.2)

# 4. Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# 5. Prepare for custom review prediction
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2

def encode_review(text):
    words = text_to_word_sequence(text)
    encoded = [1]  # <START> token
    for word in words:
        encoded.append(word_index.get(word, 2))  # 2 = <UNK> for unknown words
    return pad_sequences([encoded], maxlen=maxlen)

# 6. Predict sentiment for custom review
custom_review = "The movie was boring and too long"
encoded = encode_review(custom_review)
prediction = model.predict(encoded)[0][0]

print(f"\nReview: {custom_review}")
print(f"Sentiment Score: {prediction:.4f}")
print("Prediction:", "Positive " if prediction >= 0.5 else "Negative ")



9.
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a simple dataset
# Let's assume we want to predict 'y' based on 'x'
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # feature
y = np.array([3, 4, 2, 5, 6, 7, 8, 8, 9, 10])                # target

# Step 2: Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 3: Create and train the Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(x_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted values:", y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Step 6: Visualize the results
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x, model.predict(x), color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()


10
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a synthetic dataset
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, x.shape)  # Sinusoidal data with noise
x = x.reshape(-1, 1)

# Step 2: Define the Locally Weighted Regression function
def lwlr(query_point, X, Y, tau=0.5):
    """
    Locally Weighted Linear Regression
    query_point: the point at which we want prediction
    X: training data features
    Y: training data targets
    tau: bandwidth parameter (controls weight decay)
    """
    m = X.shape[0]
    weights = np.exp(-np.sum((X - query_point)**2, axis=1) / (2 * tau**2))
    W = np.diag(weights)  # Weight matrix
    X_aug = np.hstack((np.ones((m, 1)), X))  # Add intercept term
    theta = np.linalg.pinv(X_aug.T @ W @ X_aug) @ X_aug.T @ W @ Y
    query_aug = np.array([1, query_point[0]])
    return query_aug @ theta

# Step 3: Make predictions for all points
y_pred = np.array([lwlr(xi, x, y, tau=0.5) for xi in x])

# Step 4: Plot the results
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label='LWR fit', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Locally Weighted Regression (LWR)')
plt.legend()
plt.show()`
    }
  ];

  const [copiedIndex, setCopiedIndex] = useState(null);

  const handleCopy = async (index, text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 1500);
    } catch (err) {
      console.error("Copy failed:", err);
    }
  };

  return (
    <div className="page-container">
      <div className="gallery-grid">
        {codes.map((item, index) => (
          <div key={index} className="code-card">
            <div className="card-header">
              <div>
                <h2 className="card-title">{item.title}</h2>
                <p className="card-filename">{item.filename}</p>
              </div>
              <button
                onClick={() => handleCopy(index, item.code)}
                className="copy-button"
              >
                <Copy size={16} />
                {copiedIndex === index ? "Copied!" : "Copy"}
              </button>
            </div>
            <pre className="code-block">
              <code>{item.code}</code>
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}
