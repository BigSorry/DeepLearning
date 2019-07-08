# DeepLearning

1. Run the StyleTransferGoogle.py to generate images. At the moment it will create 50 images (for each artist one). Where each images (128x128) contains 6 images; one content image, one style image, and 4 images outputs for the 4 different VGG's.
2. Run plot,py to get the corresponding total loss plots.


* Less images go to line 210; slice image_list[:1] (first only example)
* Different models con be configured at line 206 models = {}
* Imsize is at line 183

