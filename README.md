# GenAI-Pix2Pix

 
## Bogazici University - MIS 48B - GenAI and Deep Learning Course

# ReFace: Post-Operative Facial Outcome Simulation Using Pix2Pix GAN

**Data Source**: [1] Christian Rathgeb, Didem Dogan, Fabian Stockhardt, Maria De Marsico, Christoph Busch, „Plastic Surgery: An Obstacle for Deep Face Recognition?“, in 15th IEEE Computer Society Workshop on Biometrics (CVPRW), pp. 3510-3517, 2020


This study aims to build a post operative simulation model using Pix2Pix GAN framework.

<hr />

**Data Description**: The Dataset consists of 105 paired images. The pairs belong to the same individuals who underwent a facelift operation

**Data Preprocessing**: The images are first aligned and cropped to maintain consistency between images. Dlib library was used for this task

**code files**: 

- augmentation.py The paired images are augmented using albumntation library
- dataset.py : Custom dataset class that loads original and augmented image datasets
- generator.py : U-Net style generator comprised of seven encoder blocks and seven decoder blocks.
- discriminator.py : Patch-GAN discrimintor as originally taken from Pix2Pix framework
- utils.py : Provides essential tools to save some images generated during training, save model to disk and resume training from a saved model
- config.py : Tunes and sets the essential parameter for training. Uses GPU if available. Sets hyperparameter such as learning rate, batch size and number of epochs. These parameters were set according to standard practice for GAN training and can be adjusted as needed. In addition to hyperparameters, the loss functions weights were set in this file.
- training.py : All training process were implemented in this section. 
  *Additional loss functions: compute_identity_los and VGGPPerceptualLoss
  *train_fn: operates forward backward passes
  *PairedFaceDataset: Loads images from drive files
  *save_some_examples: saves some sample images from each epoch for evaluation 
  

**Model Training**: The models are trained using the training set. The hyperparameters of the models are tuned using the Grid Search Cross-Validation technique.

**Model Evaluation**: The models are evaluated using the SSIM and PSNR metrics. The evaluation was made on the sample results of 443. epoch
