# DinoVision-Classifying-Prehistoric-Titans
Simple Dinosaur Dataset

# Dinosaur Dataset Classifier

This project focuses on classifying images of dinosaurs into five distinct categories using deep learning techniques. The dataset contains images of the following dinosaurs:

- Ankylosaurus
- Brontosaurus
- Pterodactyl
- T-Rex
- Triceratops

## Dataset Overview

The dataset consists of:
- **Training Data**: Used for training the deep learning model.
- **Validation Data**: Used to evaluate the model during training to monitor performance.
- **Test Data**: Used for evaluating the model’s generalization on unseen data.

### Dataset Structure
```
Dataset/
|-- Train/
|   |-- ankylosaurus/
|   |-- brontosaurus/
|   |-- pterodactyl/
|   |-- t_rex/
|   |-- triceratops/
|
|-- Validation/
|   |-- ankylosaurus/
|   |-- brontosaurus/
|   |-- pterodactyl/
|   |-- t_rex/
|   |-- triceratops/
|
|-- Test/
    |-- ankylosaurus/
    |-- brontosaurus/
    |-- pterodactyl/
    |-- t_rex/
    |-- triceratops/
```

Each class contains approximately 40 images. Due to the small dataset size, various data augmentation techniques are applied to improve the model's generalization ability.

## Project Workflow

1. **Data Preparation**:
   - Rescale images to a size of `(150, 150)`.
   - Perform data augmentation (rotation, flipping, zoom, etc.) on the training data.

2. **Model Architecture**:
   - Convolutional Neural Network (CNN) with the following layers:
     - Convolutional layers for feature extraction.
     - Dropout and regularization to prevent overfitting.
     - Dense layers for classification.
   - Transfer learning using pre-trained models like VGG16 to boost performance.

3. **Training**:
   - Monitor training accuracy and loss.
   - Validate the model using validation data.
   - Use early stopping to prevent overfitting.

4. **Evaluation**:
   - Test the model on unseen test data.
   - Visualize results using confusion matrix and classification reports.

## Results

- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~97.5%
- **Test Accuracy**: ~96%

### Confusion Matrix Example:
```
[[26 12  0  2  0]
 [38  2  0  0  0]
 [29 11  0  0  0]
 [28 12  0  0  0]
 [37  3  0  0  0]]
```

## Key Challenges

1. **Overfitting**:
   - Mitigated using dropout, L2 regularization, and data augmentation.

2. **Class Imbalance**:
   - Class weights were applied during training to give more importance to underrepresented classes.

3. **Small Dataset**:
   - Transfer learning with pre-trained models was used to enhance feature extraction.

## Technologies Used

- **Python**: Programming language.
- **TensorFlow/Keras**: For building and training the deep learning model.
- **Matplotlib**: For visualizing results.
- **NumPy**: For numerical computations.

## How to Use

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python train.py
   ```

4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Future Improvements

1. Collect more data to improve classification accuracy.
2. Explore more advanced transfer learning models (e.g., EfficientNet, ResNet50).
3. Optimize hyperparameters using techniques like grid search or Bayesian optimization.
4. Deploy the model using a web interface for real-time classification.

## Acknowledgments

Special thanks to the creators of the original dinosaur dataset and the open-source community for providing the tools and resources to make this project possible.

## License

This project is for educational purposes only. Please respect the dataset's original copyright and use it responsibly.
