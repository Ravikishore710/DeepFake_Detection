### **Real vs. Fake Face Detection**

A high-accuracy deep learning solution for distinguishing authentic portraits from AI-generated faces, trained on a large-scale dataset to promote integrity in digital media.

### **Overview**

In an age where AI can generate photorealistic faces, the ability to discern real from synthetic has never been more critical. This project addresses this challenge directly by providing a complete pipeline for the binary classification of facial images: identifying them as either "real" or "fake." By leveraging a powerful Convolutional Neural Network (CNN) trained on 140,000 images, this system offers a robust tool for detecting AI-generated content.

### **Key Features**

This project presents a complete and robust pipeline for deepfake face detection. It features a meticulously fine-tuned deep learning model that achieves high classification accuracy on unseen data. The codebase is organized for clarity, covering every step from data preprocessing to training, evaluation, and final inference. Designed for flexibility, it is fully compatible with both Kaggle and Google Colab environments.

### **Dataset**

The model is trained on the comprehensive "140K Real and Fake Faces" dataset from Kaggle. This balanced collection contains 140,000 images, equally split between authentic portraits and synthetic faces.

  * **Real Faces:** Sourced from Flickr and licensed for academic use.
  * **Fake Faces:** Generated using NVIDIA's advanced StyleGAN algorithm.

The data is expected to be organized in the following directory structure:

```
real_vs_fake/
├── train/
│   ├── real/
│   └── fake/
├── valid/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### **Model Details**

The core of the detection system is a powerful Convolutional Neural Network (CNN) built using transfer learning to leverage pre-trained weights for faster and more effective training.

  * **Architecture:** Fine-tuned Xception or EfficientNet model.
  * **Loss Function:** Binary Crossentropy, ideal for binary classification tasks.
  * **Optimizer:** Adam.
  * **Input Size:** Images are resized to $299 \\times 299$ pixels to match the Xception architecture's requirements.
  * **Augmentations:** To improve generalization, the training data is augmented with random horizontal flips, brightness and contrast adjustments, and normalization.

The complete model development process is documented in the `Final_Finetuned_model.ipynb` notebook.

### **How to Run**

To get started with this project, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ravikishore710/real-vs-fake-faces.git
    cd real-vs-fake-faces
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the dataset:**
    You can download and extract it manually or use the Kaggle API.

    ```bash
    kaggle datasets download -d xhlulu/140k-real-and-fake-faces
    ```

4.  **Run training or inference:**

      * To train the model from scratch: `python train.py`
      * To make a prediction on a new image: `python predict.py --image_path path/to/your/image.jpg`

### **Results**

The model's performance was rigorously evaluated on both validation and test sets, demonstrating strong generalization capabilities.

| Metric    | Validation | Test    |
| :-------- | :--------- | :------ |
| Accuracy  | 94.7%      | 93.2%   |
| Precision | 93.1%      | 91.8%   |
| Recall    | 95.4%      | 94.0%   |
| F1 Score  | 94.2%      | 92.9%   |

### **Evaluation Notebook**

For a comprehensive breakdown of the training process, model fine-tuning, and in-depth evaluation metrics, please refer to the `Final_Finetuned_model.ipynb` notebook included in the repository.

### **Future Work**

While this project provides a solid foundation, there are several exciting avenues for future development:

  * **Video Deepfake Detection:** Extend the current image-based approach to analyze video streams, using frame-by-frame analysis or temporal models to detect inconsistencies over time.
  * **Integrate Explainable AI (XAI):** Implement techniques like LIME or SHAP to visualize and interpret the model's decisions, building trust and understanding of *why* an image is flagged as fake.
  * **Adversarial Robustness Testing:** Evaluate the model's resilience against adversarial attacks and various forms of GAN noise to identify and strengthen potential vulnerabilities.

### **License**

This project is licensed under the Apache 2.0 License. Please see the `LICENSE` file for full terms. The use of the dataset must adhere to the license provided by its original author.

### **Contributing**

Contributions are welcome. Please feel free to open an issue to discuss proposed changes or submit a pull request for any improvements.
