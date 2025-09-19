# Deep Learning Rice Classifier (ResNet18)

A deep learning project for **classifying rice varieties** using **FastAI** and a pre-trained **ResNet18** model.  
Trained on the [Rice Image Dataset from Kaggle](https://www.kaggle.com) (~75,000 images, 5 classes), achieving **~98% validation accuracy**. Ideal for agricultural applications like quality control or variety identification.

---

## 🧮 Dataset

- **Source**: Rice Image Dataset on Kaggle  
- **Total images**: 75,000 RGB images (250×250 pixels)  
- **Classes**: Arborio, Basmati, Ipsala, Jasmine, Karacadag  
- **Preprocessing**:  
  - Resize to 224×224  
  - Normalize using ImageNet statistics  
  - Data augmentation (flip, rotate) applied only to training set  

---

## 🏗️ Model Architecture

- **Backbone**: ResNet18 (pre-trained on ImageNet)  
- **Modifications**: Final fully connected layer for 5-class output, dropout & gradient clipping  
- **Optimizer**: Adam with `fit_one_cycle` scheduler  

---

## 🚀 Usage

### Installation
```bash
git clone https://github.com/Maryam-hekmat/Deep-Learning-Rice-Classifier-ResNet18-.git
cd Deep-Learning-Rice-Classifier-ResNet18-
pip install fastai torch torchvision
Prepare Dataset

Download the Kaggle Rice Image Dataset and place it at:
/kaggle/input/rice-image-dataset/Rice_Image_Dataset

Training
from fastai.vision.all import *

path = Path('/kaggle/input/rice-image-dataset/Rice_Image_Dataset')
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2,
                                   item_tfms=Resize(224),
                                   batch_tfms=aug_transforms())
learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fit_one_cycle(5, lr_max=1e-3)
learn.export('rice_classifier.pkl')

Inference
learn = load_learner('rice_classifier.pkl')
pred_class, _, probs = learn.predict('/path/to/test/image.png')
print(f"Predicted: {pred_class}, Confidence: {probs.max():.4f}")

📈 Results & Visualizations
Accuracy Plot

Sample Prediction

Confusion Matrix

🔍 Challenges & Improvements

Overfitting mitigated with data augmentation & gradient clipping

Future: Use deeper models like ResNet50

Potential: Deploy as a web or mobile app for real-time rice classification

📄 License

MIT License — free to use, modify, and distribute

🙋 Contact

For questions or collaboration, you can reach me at: maryamhekmat166@gmail.com

---

✅ **What to do next:**  
1. Make sure you have an `images/` folder in your repo and upload:  
   - `accuracy_plot.png`  
   - `sample_prediction.png`  
   - `confusion_matrix.png`  
2. Copy-paste this README into `README.md` in your repo.  

This version is **concise, readable, and complete**, with all necessary sections and your email.  

If you want, I can also make an **even shorter version** (1–2 paragraphs + images + email) for quick glance viewers. Do you want me to do that?

