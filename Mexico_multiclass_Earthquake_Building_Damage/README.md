# README

Dataset is available at link => 
[Dataset](https://drive.google.com/drive/folders/1End_chXsxfI2-_XAcv9KKJiM4TZ8rp7O?usp=sharing)

Folder structure =>


    * VBL Dataset
        |
    * VisionBeyondLimits
        **|_Images (pngs)
        **|    ***|__Image 1
        **|    ***|__Image 2
        **|    ***.
        **|    ***.
        **|    
        **|    
        **|_ Labels (jsons)
            ***|__Label 1
            ***|__Label 2
            ***.
            ***.
            
            
List of modules need to be installed:
    1. shapely  # !pip install shapely
    2. patchify  # !pip install patchify
    3. albumentations  # !pip install albumentations
    4. segmentation-models  # !pip install segmentation-models
    
 
Possible issues:
  1. may be a compatibility issue of segmentation_model with keras...do this=> 
   ```
    sm.set_framework('tf.keras')
    sm.framework()
   ```
   
   2.Some GPUs may work slow with mised precision compatibility. Turn that off unless there's memory limitations
   ```
    # mixed_precision.set_global_policy('mixed_float16')
   ```
   
   3. Ensure shortcut of dataset has been created in My Drive
   


### The Architecture
![alt text](https://github.com/sayanbiswas023/Machine_Learning_Projects/blob/main/Mexico_multiclass_Earthquake_Building_Damage/skip/unet.png)
