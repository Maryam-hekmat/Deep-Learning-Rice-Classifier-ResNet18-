# Deep-Learning-Rice-Classifier-ResNet18-
Model Architecture  Backbone: ResNet18 (pre-trained on ImageNet). Modifications: Final fully connected layer adjusted for 5-class output. Techniques like dropout and gradient clipping added for stability. Optimizer: Adam with fit_one_cycle scheduler (dynamic learning rate: starts low, peaks mid-cycle, decreases for fine-tuning).
Overview
This project implements a deep learning model for classifying rice varieties using the FastAI library and a pre-trained ResNet18 architecture. The model is trained on the Rice Image Dataset from Kaggle, which includes 75,000 images across 5 classes: Arborio, Basmati, Ipsala, Jasmine, and Karacadag. It achieves high accuracy (~98% on validation) through transfer learning, data augmentation, and the fit_one_cycle scheduler. This is ideal for agricultural applications like quality control or variety identification.
Dataset

Source: Rice Image Dataset on Kaggle
Details: 75,000 RGB images (250x250 pixels), balanced with 15,000 images per class.
Preprocessing: Images resized to 224x224, normalized using ImageNet stats. Augmentation (flips, rotations) applied only to training data to reduce overfitting.

Model Architecture

Backbone: ResNet18 (pre-trained on ImageNet).
Modifications: Final fully connected layer adjusted for 5-class output. Techniques like dropout and gradient clipping added for stability.
Optimizer: Adam with fit_one_cycle scheduler (dynamic learning rate: starts low, peaks mid-cycle, decreases for fine-tuning).

Training Process

Data Loading: Used ImageDataLoaders.from_folder to split into train/validation (80/20).
Fine-Tuning: Trained for 5 epochs with learn.fine_tune or fit_one_cycle (lr_max=1e-3).
Evaluation: Monitored accuracy, loss plots, and confusion matrix. Early stopping and model checkpointing prevented overfitting.
Results: Validation accuracy ~98%, with high confidence predictions (e.g., 99.99% for correct classes).
