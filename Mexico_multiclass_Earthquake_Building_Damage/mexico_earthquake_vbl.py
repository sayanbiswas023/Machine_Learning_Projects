"""
List of modules need to be installed
    1. shapely  # !pip install shapely
    2. patchify  # !pip install patchify
    3. albumentations  # !pip install albumentations
    4. segmentation-models  # !pip install segmentation-models
    
Data path=> Loaded directly from drive; ensure the presence of the folder VBL Dataset shared by Techfest IITB

Folder structure=>
    VBL Dataset
        |
    VisionBeyondLimits
        |_Images
        \    |__Image 1
        |    |__Image 2
        \    .
        |    .
        \    
        |    
        \_ Labels
            |__Label 1
            |__Label 2
            .
            .
        
    
"""

# Library imports

import os
import json
import shapely.ops as so
import matplotlib.pyplot as plt
import numpy as np
import io
import skimage
import random
import gc
import datetime
import segmentation_models as sm
import tensorflow as tf
# import geopandas as gpd

from shapely import wkt
from PIL import Image
from tqdm import tqdm
from skimage.draw import polygon
from skimage import color
from skimage import io as skio
from patchify import patchify
from albumentations import  RandomRotate90, GridDistortion,VerticalFlip,HorizontalFlip
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam

from multi_unet import multi_unet_model
# from keras_unet_collection import models, losses

# ============================================================================= # =============================================================================
# ============================================================================= # =============================================================================


## there's some issue with tf version
sm.set_framework('tf.keras')
sm.framework()

#setting data type policy
mixed_precision.set_global_policy('mixed_float16')

# setting up path
path='/content/drive/MyDrive/VBL Dataset/VisionBeyondLimits/'
im_path=path+'Images'
la_path=path+'Labels'
n_im=len(os.listdir(im_path))
n_la=len(os.listdir(la_path))

print(n_im)
print(n_la)



# specifying some parameters
IMG_HEIGHT=256
IMG_WIDTH=256
IMG_CHANNELS=3
batch_size=16

SEED=45
random.seed(SEED)

X=np.zeros((n_im,1024,1024,3),dtype=np.uint8)
Y=np.zeros((n_im,1024,1024),dtype=np.uint8)

# ============================================================================= # =============================================================================
# ============================================================================= # =============================================================================


#Transforming the jsons
lst=os.listdir(la_path)
lst.sort()

## check category types
catset=set()
for file in tqdm(lst): # files
  # print("File ",file, " under analysis; ")

  f=open(la_path+'/'+file)
  data=json.load(f)
  xy=data['features']['xy']
  # print(type(xy))

  for poly in xy:
    category=poly['properties']['subtype']
    p=wkt.loads(poly['wkt'])
    # print(category)
    catset.add(category)

print(catset)


imagelist=os.listdir(im_path)
imagelist.sort()

# ============================================================================= # =============================================================================
# ============================================================================= # =============================================================================


# Data augmentations
def augment(image,mask,augment=True):
  if augment==True:

    aug=RandomRotate90(p=1.0)
    augmented=aug(image=image,mask=mask)
    x1=augmented["image"]
    y1=augmented["mask"]

    aug=GridDistortion(p=1.0)
    augmented=aug(image=image,mask=mask)
    x2=augmented["image"]
    y2=augmented["mask"]

    aug=HorizontalFlip(p=1.0)
    augmented=aug(image=image,mask=mask)
    x3=augmented["image"]
    y3=augmented["mask"]

    aug=VerticalFlip(p=1.0)
    augmented=aug(image=image,mask=mask)
    x4=augmented["image"]
    y4=augmented["mask"]

  return [image,x1,x2,x3,x4],[mask,y1,y2,y3,y4]



# iterating through jsons and assembling polygons to labelled images
i=1
for file in tqdm(lst): # files

  imim=skio.imread(im_path+'/'+imagelist[i-1]) # the image
  imar=np.array(imim,dtype=np.uint8)
  X[i-1]=imar # coresponding np array

  print("File ",file, " under analysis; ")

  f=open(la_path+'/'+file) # coresponding json
  data=json.load(f)
  xy=data['features']['xy']
  # print(type(xy))

  shape=[]
  category=[]
  for poly in xy:
    category.append(poly['properties']['subtype'])
    p=wkt.loads(poly['wkt'])
    # p=gpd.GeoSeries(p)
    shape.append(p)

  # shape=so.cascaded_union(shape)
  # fig, axs = plt.subplots()


  mask=np.zeros((1024,1024))

  j=0
  print(len(shape)," polygons found")
  for poly in shape:
    poly_coordinates=np.array(list(poly.exterior.coords))
    rr,cc=polygon(poly_coordinates[:,0],poly_coordinates[:,1],(1024,1024));

    cat=category[j]

    if cat=='destroyed':
      val=1
    if cat=='major-damage':
      val=2
    if cat=='minor-damage':
      val=3
    if cat=='no-damage':
      val=4
    if cat=='un-classified':
      val=5
    
    mask[cc,rr]=val
    j+=1

  Y[i-1]=mask

  masked_image=Image.fromarray(255*mask)

  print("size of image is ",masked_image.size)
  # plt.imshow(masked_image)
  skimage.io.imshow(color.label2rgb(mask,image=imim,bg_label=1,kind='overlay'))
  plt.show()

  print(117-i," left")
  i+=1
    


# sanity check 1
a=random.randint(0,n_im-1)

print(X[a].shape)
skio.imshow(X[a])
plt.show()
skio.imshow(color.label2rgb(255*Y[a]))


# ============================================================================= # =============================================================================
# ============================================================================= # =============================================================================



#Splitting dataset

# train_test split
X_tr, X_test, Y_tr, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42)

# valid split
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 0.65, random_state = 42)

# ============================================================================= # =============================================================================
# ============================================================================= # =============================================================================


## one image => 16 patches => (1+(n_aug-1)) augmentations for each
n_aug=5 ## no of augmentations =>  n_aug=5 if all augmentatins on; 1 if all augmentations off
X_train=tf.Variable(tf.zeros((X_tr.shape[0]*16*n_aug,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8))
Y_train=tf.Variable(tf.zeros((Y_tr.shape[0]*16*n_aug,IMG_HEIGHT,IMG_WIDTH),dtype=np.uint8))


#Data Augmentation and Preprocessing

i=0
for im, ma in tqdm(zip(X_tr,Y_tr),total=X_tr.shape[0]):

  ## depatching
  patches_img=patchify(im,(256,256,3),step=(256))
  patches_lab=patchify(ma,(256,256),step=(256))

  # print(patches_lab.shape)

  ## augmenting and saving

  for p in range(patches_img.shape[0]):
    for q in range(patches_img.shape[1]):
      single_im_patch=patches_img[p,q,:,:,:]
      single_la_patch=patches_lab[p,q,:,:]

      single_im_patch=np.squeeze(single_im_patch)
      single_la_patch=np.squeeze(single_la_patch)

      augx,augy=augment(single_im_patch,single_la_patch) # augx & y are a list of len n_aug

      
      for j in range(n_aug):
        X_train[i+j].assign(augx[j])
      for j in range(n_aug):
        Y_train[i+j].assign(augy[j])

      i+=n_aug
    
    
# shuffle training dataset
tf.random.shuffle(X_train, seed=SEED)
tf.random.shuffle(Y_train, seed=SEED)


#sanity check 2
a=random.randint(0,X_train.shape[0])
im=X_train[a].numpy()
skio.imshow(im)
plt.show()
ma=Y_train[a].numpy()
skio.imshow(color.label2rgb(255*ma))
print(X_train[a].shape)
print(Y_train[a].shape)

#sanity check 3
a=random.randint(0,X_train.shape[0])
skimage.io.imshow(color.label2rgb(Y_train[a].numpy(),image=X_train[a].numpy(),bg_label=1,kind='overlay'))
plt.show()



n_classes=len(np.unique(Y_train)) # includes background as well # bg # destroyed # maj dam # min dam # no dam # unclassi
print(n_classes)

# converts to categorical one hot encoding
Y_train=tf.convert_to_tensor(to_categorical(Y_train,num_classes=n_classes,dtype='float16')) # returns tensor of shape [n_images,256,256,n_classes]




# processing validation data
X_valid=tf.Variable(tf.zeros((X_val.shape[0]*16,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8))
Y_valid=tf.Variable(tf.zeros((Y_val.shape[0]*16,IMG_HEIGHT,IMG_WIDTH),dtype=np.uint8))

i=0
for im, ma in tqdm(zip(X_val,Y_val),total=X_val.shape[0]):

  ## depatching
  patches_img=patchify(im,(256,256,3),step=(256))
  patches_lab=patchify(ma,(256,256),step=(256))


  for p in range(patches_img.shape[0]):
    for q in range(patches_img.shape[1]):
      single_im_patch=patches_img[p,q,:,:,:]
      single_la_patch=patches_lab[p,q,:,:]

      single_im_patch=np.squeeze(single_im_patch)
      single_la_patch=np.squeeze(single_la_patch)
      
      # print(single_im_patch.shape, "   ",single_la_patch.shape)
      X_valid[i].assign(single_im_patch)
      Y_valid[i].assign(single_la_patch)

      i+=1
      

tf.random.shuffle(X_valid, seed=SEED)
tf.random.shuffle(Y_valid, seed=SEED)



#sanity check 4
a=random.randint(0,X_valid.shape[0])
skimage.io.imshow(color.label2rgb(Y_valid[a].numpy().astype(np.uint8),image=X_valid[a].numpy().astype(np.uint8),bg_label=1,kind='overlay'))
plt.show()

# converts validation to categorical one hot encoding
Y_valid=tf.convert_to_tensor(to_categorical(Y_valid,num_classes=n_classes,dtype='float16'))

# Normalise
tf.cast(X_train,dtype=tf.float16)
X_train=(X_train/255)

tf.cast(X_valid,dtype=tf.float16)
X_valid=(X_valid/255)

tf.cast(Y_train,dtype=tf.float16)

tf.cast(Y_valid,dtype=tf.float16)



# ============================================================================= # =============================================================================
# ============================================================================= # =============================================================================


#Model assembling (From Segmentation Models)

# setting up the model
BACKBONE = 'resnet34'
# BACKBONE = 'vgg19'
epochs=30
lr=0.1

weights=weights = [0.5, 0.05, 0.05, 0.05, 0.3, 0.05]

# losss=sm.losses.bce_jaccard_loss
losss=sm.losses.CategoricalCELoss(class_weights=weights)
# losss=tversky_index_loss
# losss=(5*sm.losses.CategoricalCELoss(class_weights=weights))+sm.losses.DiceLoss(class_weights=weights)

preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
X_train = preprocess_input(X_train)
X_valid = preprocess_input(X_valid)

# define model
model = sm.Unet(BACKBONE,classes=n_classes,activation='softmax', encoder_weights='imagenet',encoder_freeze=True,input_shape=(256, 256, 3),decoder_use_batchnorm=True)

#model compiling
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=losss, metrics=[sm.metrics.iou_score])


# training the model
def lr_scheduler(epochs, lr):
    decay_rate = 0.1
    decay_step = 8
    if epochs % decay_step == 0 and epochs:
        return lr * decay_rate
    return lr

start1 = datetime.datetime.now() 

## model checkpoints
checkpointer=tf.keras.callbacks.ModelCheckpoint('unet_earthquake_30_2_epochs.h5',verbose=1,save_best_only=True)
callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=6,monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    ]

## run
history=model.fit(X_train, 
          Y_train,
          validation_data=(X_valid,Y_valid),
          batch_size=32, 
          epochs=epochs,
          verbose=1,
          callbacks=callbacks)

stop1 = datetime.datetime.now()

## Execution time of the model 
execution_time_Unet = stop1-start1
print("UNet execution time is: ", execution_time_Unet)


## saving
model.save('Earthquake_unet_30_epochs.hdf5')


## ============================================================================= # =============================================================================
## ============================================================================= # =============================================================================
## ============================================================================= # =============================================================================

# training with made from scratch unet

# lr=0.01
# epochs=30
# def get_model():
#     return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
# 
# model = get_model()
# 
# 
# def lr_scheduler(epochs, lr):
#     decay_rate = 0.1
#     decay_step = 5
#     if epochs % decay_step == 0 and epochs:
#         return lr * decay_rate
#     return lr
# 
# start1 = datetime.datetime.now() 
# 
# ## model checkpoints
# checkpointer=tf.keras.callbacks.ModelCheckpoint('unet_earthquake_30_epochs.h5',verbose=1,save_best_only=True)
# callbacks=[
#     tf.keras.callbacks.EarlyStopping(patience=4,monitor='val_loss'),
#     tf.keras.callbacks.TensorBoard(log_dir='logs'),
#     tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
#     ]
# 
# ## run
# history=model.fit(X_train, 
#           Y_train,
#           validation_data=(X_valid,Y_valid),
#           batch_size=64, 
#           epochs=epochs,
#           verbose=1,
#           callbacks=callbacks)
# 
# stop1 = datetime.datetime.now()
# 
# ## Execution time of the model 
# execution_time_Unet = stop1-start1
# print("UNet execution time is: ", execution_time_Unet)

## =============================================================================

## ============================================================================= # =============================================================================
## ============================================================================= # =============================================================================
## ============================================================================= # =============================================================================





# preprocessing test images 
# NOTE: test data is a part of data supplied
X_te=tf.Variable(tf.zeros((X_test.shape[0]*16,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8))
Y_te=tf.Variable(tf.zeros((Y_test.shape[0]*16,IMG_HEIGHT,IMG_WIDTH),dtype=np.uint8))


i=0
for im, ma in tqdm(zip(X_test,Y_test),total=X_test.shape[0]):

  ## depatching
  patches_img=patchify(im,(256,256,3),step=(256))
  patches_lab=patchify(ma,(256,256),step=(256))


  for p in range(patches_img.shape[0]):
    for q in range(patches_img.shape[1]):
      single_im_patch=patches_img[p,q,:,:,:]
      single_la_patch=patches_lab[p,q,:,:]

      single_im_patch=np.squeeze(single_im_patch)
      single_la_patch=np.squeeze(single_la_patch)
      
      # print(single_im_patch.shape, "   ",single_la_patch.shape)
      X_te[i].assign(single_im_patch)
      Y_te[i].assign(single_la_patch)

      i+=1
      

# predicting random images (in patches)
a=random.randint(0,X_te.shape[0])
fig = plt.figure(figsize=(10, 8))

fig.add_subplot(1, 3, 1)
skio.imshow(X_te[a].numpy()) # the actual image

fig.add_subplot(1, 3, 2)
skio.imshow(color.label2rgb(255*Y_te[a].numpy())) # the ground truth label

testet=tf.expand_dims(X_te[a],axis=0) 
pred = model.predict(testet, verbose=1) # predict

pred=tf.squeeze(pred)
pred=tf.math.argmax(pred,axis=2)
pred = tf.keras.backend.eval(pred)

fig.add_subplot(1, 3, 3)
skio.imshow(color.label2rgb(255*pred)) # the predicted label

plt.show()






























