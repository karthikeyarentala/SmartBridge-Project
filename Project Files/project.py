#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sb
import scipy as sp
import tensorflow as tf
import flask as f
import warnings as w
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator as idg
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping as ES
from keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from IPython.display import display


# In[2]:


w.filterwarnings('ignore')


# In[3]:


ds = "D:/Smart Internz Internship/Fruit And Vegetable Diseases Dataset"
classes = os.listdir(ds)


# In[4]:


#creation of directories for train,test,val sets
op_dir = "Output _dataset"
os.makedirs(op_dir,exist_ok=True)
os.makedirs(os.path.join(op_dir,'train'),exist_ok=True)
os.makedirs(os.path.join(op_dir,'val'),exist_ok=True)
os.makedirs(os.path.join(op_dir,'test'),exist_ok=True)


# In[5]:


for i in classes:
    # Create train/val/test dirs for each class
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(op_dir, split, i), exist_ok=True)
    
    class_dir = os.path.join(ds, i)
    imgs = os.listdir(class_dir)[:200]
    print(i, len(imgs))

    train_val_imgs, test_imgs = tts(imgs, test_size=0.2, random_state=42)
    train_imgs, val_imgs = tts(train_val_imgs, test_size=0.25, random_state=42)

    # Copy images to their respective directories
    for img in train_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(op_dir, 'train', i, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(op_dir, 'val', i, img))
    for img in test_imgs:
        shutil.copy(os.path.join(class_dir, img), os.path.join(op_dir, 'test', i, img))


# In[11]:


print("Split the dataset into train, val, test sets:")
#define the directories
ds_dir = "D:/Smart Internz Internship/Output_dataset"

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(ds_dir, split), exist_ok=True)

train_dir = os.path.join(ds_dir,'train')
val_dir = os.path.join(ds_dir,'val')
test_dir = os.path.join(ds_dir,'test')

#define the image size
IMG_SIZE = (224,224)

#creation of image data generators
train_datageneration = idg(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_test_datageneration = idg(rescale=1./255)
train_generator = train_datageneration.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary'
)
val_generator = val_test_datageneration.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary'
)
test_generator = val_test_datageneration.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

#printing class indices for the reference purpose
print(train_generator.class_indices)
print(val_generator.class_indices)
print(test_generator.class_indices)


# In[13]:


#visualizing the data

#1. Healthy Apple
folder_path = "D:/Smart Internz Internship/Output_dataset/train/Apple__Healthy"
img_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg','.png','.jpeg'))]
selected_img = random.choice(img_files)
img_path = os.path.join(folder_path, selected_img)
img = Image.open(img_path)
display(img)


# In[15]:


#2. Rotten Guava
folder_path = "D:/Smart Internz Internship/Output_dataset/train/Guava__Rotten"
img_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg','.png','.jpeg'))]
selected_img = random.choice(img_files)
img_path = os.path.join(folder_path, selected_img)
img = Image.open(img_path)
display(img)


# In[17]:


#3. Rotten Jujube
folder_path = "D:/Smart Internz Internship/Output_dataset/train/Jujube__Rotten"
img_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg','.png','.jpeg'))]
selected_img = random.choice(img_files)
img_path = os.path.join(folder_path, selected_img)
img = Image.open(img_path)
display(img)


# In[19]:


#4. Healthy Tomato
folder_path = "D:/Smart Internz Internship/Output_dataset/train/Tomato__Healthy"
img_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg','.png','.jpeg'))]
selected_img = random.choice(img_files)
img_path = os.path.join(folder_path, selected_img)
img = Image.open(img_path)
display(img)


# In[21]:


#splitting the data for the model building
train_path = "D:/Smart Internz Internship/Output_dataset/train"
test_path = "D:/Smart Internz Internship/Output_dataset/test"


# In[23]:


train_datagen = idg(
    rescale=1./255,
    zoom_range=0.2,
    shear_range=0.2
)
test_datagen = idg(rescale=1./255)


# In[25]:


Train = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=20
)
Test = test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=20
)


# In[27]:


#Vgg16 Transfer-Learning model
vgg = VGG16(
    include_top=False,
    input_shape=(224,224,3)
)


# In[29]:


for lay in vgg.layers:
    print(lay)


# In[31]:


len(vgg.layers)


# In[33]:


for lay in vgg.layers:
    lay.trainable=False


# In[35]:


x = Flatten()(vgg.output)


# In[37]:


Output = Dense(28,activation='softmax')(x)


# In[39]:


vgg16 = Model(vgg.input,Output)


# In[41]:


vgg16.summary()


# In[43]:


opt = Adam(learning_rate=0.0001)

early_stopping = ES(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

vgg16.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

#training the model with early stopping callback
history = vgg16.fit(
    Train,
    validation_data=Test,
    epochs=15,
    steps_per_epoch=20,
    callbacks=[early_stopping]
)  


# In[45]:


#Testing the model and data prediction

#Testing 1

img_path1 = "D:/Smart Internz Internship/Output_dataset/train/Potato__Healthy/freshPotato (168).jpg"

img1 = load_img(img_path1,target_size=(224,224))
x = img_to_array(img1)
x = preprocess_input(x)
prediction1 = vgg16.predict(np.array([x]))

prediction1


# In[47]:


#Testing 2

img_path2 = "D:/Smart Internz Internship/Output_dataset/train/Pomegranate__Rotten/RottenPomegranate (21).jpg"
img2 = load_img(img_path2,target_size=(224,224))
x = img_to_array(img2)
x = preprocess_input(x)
prediction2 = vgg16.predict(np.array([x]))

prediction2


# In[49]:


#Testing 3

img_path3 = "D:/Smart Internz Internship/Output_dataset/train/Orange__Healthy/freshOrange (108).png"
img3 = load_img(img_path3,target_size=(224,224))
x = img_to_array(img3)
x = preprocess_input(x)
prediction3 = vgg16.predict(np.array([x]))

prediction3


# In[51]:


#Testing 4

img_path4 = "D:/Smart Internz Internship/Output_dataset/train/Cucumber__Rotten/rottenCucumber (172).jpg"
img4 = load_img(img_path4,target_size=(224,224))
x = img_to_array(img4)
x = preprocess_input(x)
prediction4 = vgg16.predict(np.array([x]))

prediction4


# In[65]:


# 1. First, create the index_to_class mapping (do this right after defining your generators)
class_indices = train_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

# 2. Then define your prediction function
def predict_image_class(model, img_path, target_size=(224, 224)):
    """
    Predict the class of an image using the trained model.
    
    Args:
        model: Trained Keras model
        img_path: Path to the image file
        target_size: Target size of the image (must match model input size)
        
    Returns:
        Predicted class name and confidence score
    """
    
    # Load and preprocess the image
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match training
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Get the class name
    predicted_class = index_to_class[predicted_index]
    
    # Display the image and prediction
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}")
    plt.axis('off')
    plt.show()
    
    return predicted_class, confidence

# 3. Now you can use the function
test_image_path = "D:/Smart Internz Internship/Output_dataset/train/Strawberry__Rotten/126.jpg"
predicted_class, confidence = predict_image_class(vgg16, test_image_path)
print(f"The image is predicted to be: {predicted_class}")
print(f"Confidence: {confidence:.2%}")


# In[81]:


# Example usage:
# Load your trained model (if not already loaded)
# model = load_model('your_model.h5')  # Uncomment if you need to load a saved model

# Path to your test image
test_image_path = "D:/Smart Internz Internship/Output_dataset/test/Bellpepper__Rotten/rottenPepper (114).jpg"

# Make prediction
predicted_class, confidence = predict_image_class(vgg16, test_image_path)

# Print results
print(f"The image is predicted to be: {predicted_class}")
print(f"Confidence: {confidence:.2%}")


# In[ ]:
vgg16.save("Healthy_vs_Rotten.h5")
# Save the index_to_class mapping

# Save this mapping as a dictionary (or recreate during runtime)
class_indices = train_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

import pickle
with open('index_to_class.pkl', 'wb') as f:
    pickle.dump(index_to_class, f)
# Save the model and class index mapping
vgg16.save("Healthy_vs_Rotten.h5")

