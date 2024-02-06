#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install imutils')


# In[9]:


import os
from tqdm import tqdm
import imutils
import random
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from keras.layers import Input

from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D,  GlobalAveragePooling2D
from keras.applications import resnet


import warnings
warnings.filterwarnings('ignore')


# In[10]:


os.chdir("E:\dataset")


# In[11]:


WIDTH, HEIGHT, CHANNEL = (224, 224, 3)
NUM_CLASSES = len(os.listdir("Training"))
EPOCHS = 50
VERBOSE = 1


# In[12]:


modelRN = resnet.ResNet50(
      input_shape = (HEIGHT, WIDTH, CHANNEL),
      include_top = False,
      weights = 'imagenet'
    )


# In[13]:


for layers in modelRN.layers:
    layers.trainable = False


# In[14]:


# for layer in resnet50.layers:
#     if hasattr(layer, 'rate'):
#         layer.rate = 0.5


# In[15]:


modelRN.summary()


# In[16]:


top_layer = Dropout(0.5)(modelRN.output)
top_layer = Flatten()(top_layer)
top_layer = Dropout(0.5)(top_layer)
top_layer = Dense(NUM_CLASSES, activation="softmax")(top_layer)


# In[17]:


modelRN = keras.Model(modelRN.input, top_layer)
modelRN.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
modelRN.summary()


# In[18]:


X_train = []
y_train = []

os.chdir("E:\dataset\Training")

train_sub_dirs = os.listdir()[::-1]


# In[19]:


# Define the processBlock function
def processBlock(image):
    # Your processing logic here
    processed_image = cv2.resize(image, (224, 224))
    return processed_image


# In[20]:


import cv2
from tqdm import tqdm

for dirname in train_sub_dirs:
    print(dirname)
    for file_name in tqdm(os.listdir(dirname)):
        img = cv2.imread(os.path.join(dirname, file_name))
        img = processBlock(img)
        X_train.append(img)
        y_train.append(dirname)


# In[21]:


os.chdir("./../Testing")


# In[22]:


X_test = []
y_test = []

os.chdir("./../Testing")

test_sub_dirs = os.listdir()[::-1]


# In[23]:


for dirname in test_sub_dirs:
    print(dirname)
    for file_name in tqdm(os.listdir(dirname)):
        img = cv2.imread(os.path.join(dirname, file_name))
        img = processBlock(img)
        X_test.append(img)
        y_test.append(dirname)


# In[24]:


dict_out = {}

for i in range(len(test_sub_dirs)):
    dict_out.update({test_sub_dirs[i]: i})


# In[25]:


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = np.array(X_train)
X_test = np.array(X_test)


# In[26]:


print(f"Shape of images in X_train: {X_train.shape}")
print(f"Shape of images in X_test: {X_test.shape}")
print(f"Shape of images in y_train: {y_train.shape}")
print(f"Shape of images in y_test: {y_test.shape}")


# In[29]:


from keras.preprocessing.image import ImageDataGenerator

# Define the data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Choose an image for augmentation visualization
image_to_augment = X_train[0]  # Assuming X_train is defined with your training images

# Reshape the image to (1, HEIGHT, WIDTH, CHANNEL) for the flow method
image_to_augment = image_to_augment.reshape((1,) + image_to_augment.shape)

# Generate augmented images
augmented_images = []

# Generate augmented images and save them
for batch in datagen.flow(image_to_augment, batch_size=1):
    augmented_images.append(batch[0])
    if len(augmented_images) >= 5:  # Generate 5 augmented images for visualization
        break

# Plot the original and augmented images
plt.figure(figsize=(15, 5))
plt.subplot(1, 6, 1)
plt.imshow(image_to_augment[0].astype(np.uint8))  # Convert to uint8 for proper display
plt.title('Original Image')

for i, augmented_image in enumerate(augmented_images):
    plt.subplot(1, 6, i + 2)
    plt.imshow(augmented_image.astype(np.uint8))  # Convert to uint8 for proper display
    plt.title(f'Augmented #{i+1}')

plt.show()


# In[20]:


RNhistory = modelRN.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, verbose=VERBOSE)


# In[21]:


modelRN.save('E:/dataset/SeResnet50.h5')


# In[22]:


predictions = modelRN.predict(X_test)
class_predictions = np.argmax(predictions, axis=1) 
predicted_classes = [list(dict_out.keys())[list(dict_out.values()).index(val)] for val in class_predictions]


# In[23]:


y_test_values = np.argmax(y_test, axis=1)
actual_classes = [list(dict_out.keys())[list(dict_out.values()).index(val)] for val in y_test_values]


# In[24]:


con_matrix = confusion_matrix(y_test_values, class_predictions)


# In[29]:


# Assuming you have 'num_epochs' defined somewhere in your code
epochs = range(1, EPOCHS + 1)

# Plotting the accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(epochs,RNhistory.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(epochs,RNhistory.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.grid(True)
plt.show()


# In[30]:


# Plotting the loss graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, RNhistory.history['loss'], label='Training Loss', marker='o')
plt.plot(epochs,RNhistory.history['val_loss'], label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()


# In[31]:


# Predictions
predictions = modelRN.predict(X_test)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[41]:


# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'green', 'red', 'purple'])

for i, color in zip(range(NUM_CLASSES), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve (class {i}) (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Classification')
plt.legend(loc='lower right')
plt.show()


# In[42]:


# Confusion Matrix
predictions = modelRN.predict(X_test)
class_predictions = np.argmax(predictions, axis=1)
y_test_values = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_test_values, class_predictions)



# In[43]:


# Predictions
predictions = modelRN.predict(X_test)
class_predictions = np.argmax(predictions, axis=1)
y_test_values = np.argmax(y_test, axis=1)

# Calculate Precision, Recall, and F1 Score
precision = precision_score(y_test_values, class_predictions, average='weighted')
recall = recall_score(y_test_values, class_predictions, average='weighted')
f1 = f1_score(y_test_values, class_predictions, average='weighted')

print(f'Weighted Precision: {precision:.4f}')
print(f'Weighted Recall: {recall:.4f}')
print(f'Weighted F1 Score: {f1:.4f}')


# In[44]:


plt.figure(figsize=(8, 6))
sns.heatmap(con_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['glioma', 'meningioma', 'notumor', 'pituitary'], 
            yticklabels=['glioma', 'meningioma', 'notumor', 'pituitary'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()


# In[48]:


fig, axes = plt.subplots(4, 5, figsize=(15, 15))

for i in range(20):
    axes[i//5, i%5].imshow(X_test[i]) 
    axes[i//5, i%5].set_title(f"Predicted: {predicted_classes[i]} \n Actual: {actual_classes[i]}")
plt.show()


# In[58]:


# Evaluate the model on the test set
test_loss, test_accuracy = modelRN.evaluate(X_test, y_test, verbose=1)

# Print the test accuracy
print(f'Accuracy of this model is : {test_accuracy * 100:.2f}%')


# In[ ]:




