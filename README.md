# Celebrity Search

## Task
The task was to develop an accurate convolutional neural network (CNN) to classify images of celebrities. The Kaggle dataset was used, which contains images of various celebrities, but for this project, we focused on only five: **Angelina Jolie**, **Leonardo DiCaprio**, **Robert Downey Jr.**, **Tom Cruise**, and **Megan Fox**.

## Dataset
The dataset used for this project is from Kaggle: [Celebrity Face Image Dataset](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset). The dataset contains images of various celebrities, and for this project, the data of only the five celebrities mentioned above was used.

## Google Colab Notebook
The model was developed and trained using Google Colab. The notebook can be accessed via this link: [Google Colab Notebook](http://bit.ly/4h6elpq).

## Libraries Used
The following libraries were used in the development of this model:

- Kaggle
- OS
- Shuffle
- TensorFlow
- Matplotlib
- Numpy
- Sklearn
- Seaborn
- Google Colab

## Loading the Data
To load the dataset into the Google Colab environment, the Kaggle API key was uploaded. The Kaggle library was then installed using `pip install kaggle`, and the API key was set up. The dataset was downloaded directly using Kaggle API commands.

## Data Exploration and Preprocessing
Upon downloading the dataset, several preprocessing steps were necessary:

1. The dataset was unzipped and renamed for easier handling.
2. A loop was used to filter and delete directories of unwanted celebrities.
3. The data was split into training (80%) and validation (20%) datasets.
4. Images were resized to 224x224 to ensure uniform input size.
5. Data augmentation was introduced to add randomness, but it increased the complexity of training, requiring more epochs and leading to minimal performance gain.

## Model Architecture and Compilation
The model architecture consists of the following layers:

1. **Convolutional Layer**: 64 neurons with a 3x3 kernel.
2. **Pooling Layer**: Max Pooling with a 2x2 kernel.
3. **Flattening Layer**: Converts the feature maps into a one-dimensional vector.
4. **Dense Layers**:
   - First dense layer with 128 neurons and ReLU activation function.
   - Output layer with 5 neurons (one for each celebrity) using the Softmax activation function.

The optimizer used is **Adam** with a learning rate of 0.0001 and a beta-1 momentum of 0.95. The loss function used is **Sparse Categorical Cross-Entropy**, which is suitable for integer labels.

## Training and Evaluation
The model was trained with an 80-20 test-train split, and the accuracy and loss were tracked over 14 epochs. The model achieved a final accuracy of 95% with a loss of 0.4907. The **early stopping** strategy was implemented to prevent overfitting, with patience set to 5 epochs.

## Experiments and Observations

### Experiment 1: Parameters: 25 Epochs, Default Momentum and Learning Rate
**Observation**: The model overfitted, performing well only on the training dataset.

### Experiment 2: Using Stochastic Gradient Descent instead of Adam
**Observation**: The accuracy dropped significantly, and the model failed to improve over epochs.

### Experiment 3: Adam Optimizer with Learning Rate = 0.01, 2 Convolutional Layers, and 2 Pooling Layers
**Observation**: The model achieved 100% accuracy, indicating overfitting.

### Experiment 4: Adjusting the Learning Rate to 0.002
**Observation**: The loss curve became uneven, indicating overfitting.

### Experiment 5: Adam Optimizer with Learning Rate 0.002 and 2 Convolutional Layers
**Observation**: The results were similar, with no smooth loss curve.

### Experiment 6: Adjusting Learning Rate to 0.0001 and Beta-1 Momentum to 0.95
**Observation**: This helped smooth the loss curve and minimized overfitting.

## Final Model Settings
After experimenting with various configurations, the following settings were chosen to achieve the best balance between accuracy, loss, and avoiding overfitting:

- **CNN Architecture**: 1 convolutional layer, 1 pooling layer, 1 flattening layer, and 2 dense layers.
- **Optimizer**: Adam with learning rate = 0.0001 and beta 1 momentum = 0.95.
- **Early Stopping**: Patience set to 5 epochs.
- **Test-Train Split**: 80-20.
- **Epochs**: 14 with final accuracy = 95% and loss = 0.4907.

## Conclusion
The final model achieved a high accuracy of 95% after 14 epochs. The use of early stopping, proper adjustments to the optimizer, and data augmentation techniques helped the model perform well while avoiding overfitting.

---

## References
1. Author Name, *Title of the Paper*, Journal Name, Year.
2. Author Name, *Title of the Book*, Publisher, Year.

