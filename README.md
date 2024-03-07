# Swin transformer inflitration segmentation in UPennGBM database
Swin transformer for infiltration segmentation using 11 channels of MRI (structural and functional DSC and DTI images). Using Monai and Pytorch.

In process...

Copy in tne root directory a Dataset folder with the following structure:

Dataset_106_10_casos
    
    ├───recurrence
    │   └───images
    │       ├───UPENN-GBM-00036_21
    │       ├───UPENN-GBM-00042_11
    │       ├───UPENN-GBM-00045_21
    │       └───UPENN-GBM-00051_21
    └───test
        ├───images
        │   ├───images_DSC
        │   │   ├───UPENN-GBM-00036_11
        │   │   ├───UPENN-GBM-00042_11
        │   │   ├───UPENN-GBM-00045_11
        │   │   └───UPENN-GBM-00051_11
        │   ├───images_DTI
        │   │   ├───UPENN-GBM-00036_11
        │   │   ├───UPENN-GBM-00042_11
        │   │   ├───UPENN-GBM-00045_11
        │   │   └───UPENN-GBM-00051_11
        │   └───images_structural
        │       ├───UPENN-GBM-00036_11
        │       ├───UPENN-GBM-00042_11
        │       ├───UPENN-GBM-00045_11
        │       └───UPENN-GBM-00051_11
        └───labels
