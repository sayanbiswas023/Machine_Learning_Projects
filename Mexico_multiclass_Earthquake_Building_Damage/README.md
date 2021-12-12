Dataset is available at link => https://drive.google.com/drive/folders/1End_chXsxfI2-_XAcv9KKJiM4TZ8rp7O?usp=sharing

Folder structure =>

Directory:
	--VisionBeyondLimits
	|
	|--Images
		|	
		|--image 1
		|--image 2
		|--image 3
	|--Labels
		|	
		|--label 1
		|--label 2
		|--label 3 
      
      
      
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
   ...
    sm.set_framework('tf.keras')
    sm.framework()
   ...
   
   2.Some GPUs may work slow with mised precision compatibility. Turn that off unless there's memory limitations
   ...
    # mixed_precision.set_global_policy('mixed_float16')
   ...
   
   3. Create a shortcut of dataset in My Drive
   
  
