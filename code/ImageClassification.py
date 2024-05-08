#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing libraries
get_ipython().system('pip install fastai')
from fastai.vision.all import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Setting the path
x  = 'D:\\AI master class\\DeepLearningInImageClassification\\Data Sources\\Intel Image Classification\\seg_train'
path = Path(x)

# Displaying the contents of the path
print(list(path.iterdir()))

# Setting the seed for reproducibility
np.random.seed(40)

# Defining the data block
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   splitter=RandomSplitter(valid_pct=0.2, seed=42),
                   get_y=parent_label,
                   item_tfms=Resize(460),
                   batch_tfms=[*aug_transforms(size=224, min_scale=0.75), Normalize.from_stats(*imagenet_stats)])

# Creating the dataloaders
dls = dblock.dataloaders(path, num_workers=4)

# Displaying a batch of data
dls.show_batch(nrows=3, ncols=3)

# Printing the classes and their count
print(dls.vocab)
print(len(dls.vocab), dls.c)

# Creating the learner
learn = cnn_learner(dls, resnet18, metrics=accuracy)

# Finding a good learning rate
lr_min, lr_steep = learn.lr_find()

# Training the model
learn.fit_one_cycle(40, slice(lr_min,lr_steep))

# Unfreezing the model
learn.unfreeze()

# Training the entire model
learn.fit_one_cycle(20, slice(1e-4,1e-3))

# Plotting the training and validation losses
learn.recorder.plot_loss()

# Creating a ClassificationInterpretation object
interp = ClassificationInterpretation.from_learner(learn)

# Plotting the confusion matrix
interp.plot_confusion_matrix()

# Plotting the top losses
interp.plot_top_losses(6, figsize=(25,5))

# Predicting the class of an image
img = PILImage.create('/kaggle/input/intel-image-classification/seg_test/seg_test/glacier/21982.jpg')
print(learn.predict(img)[0])

# Exporting the learner
learn.export(file = Path("/kaggle/working/export.pkl"))

# Saving the model
learn.model_dir = "/kaggle/working"
learn.save("stage-1", return_path=True)


# In[ ]:




