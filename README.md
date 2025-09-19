# Deep-Learning-Rice-Classifier-ResNet18-
Model Architecture  Backbone: ResNet18 (pre-trained on ImageNet). Modifications: Final fully connected layer adjusted for 5-class output. Techniques like dropout and gradient clipping added for stability. Optimizer: Adam with fit_one_cycle scheduler (dynamic learning rate: starts low, peaks mid-cycle, decreases for fine-tuning).
