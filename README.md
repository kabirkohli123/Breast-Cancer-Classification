# Breast Cancer Classification with Neural Network

This repository contains a project for classifying breast cancer using a neural network model. The classification task is performed using a dataset that includes various features of cell nuclei present in breast cancer biopsies. The model is built using a neural network, trained and evaluated using Python's machine learning libraries.

## Project Structure

- `BreastCancerClassificationWithNeuralNetwork.ipynb`: Jupyter notebook that contains all the steps for loading the dataset, preprocessing, training, evaluating the neural network model, and making predictions.
- `BreastCancerData.csv`: The dataset used in this project for classifying breast cancer as either malignant or benign.

## Dataset

The dataset used is `BreastCancerData.csv`, which consists of the following features:
- **Mean radius**
- **Mean texture**
- **Mean perimeter**
- **Mean area**
- ... (Add any other relevant columns here)

The target variable is the diagnosis result: 
- `M` = Malignant 
- `B` = Benign

## Steps

1. **Data Preprocessing**
   - Handle missing values.
   - Feature scaling using `StandardScaler`.
   - Splitting the dataset into training and testing sets.

2. **Model Building**
   - A feedforward neural network is constructed using libraries like TensorFlow/Keras or PyTorch.
   - The architecture includes multiple dense layers, ReLU activation, and softmax for the output layer.

3. **Training and Evaluation**
   - The model is trained on the training dataset.
   - Performance is evaluated using metrics such as accuracy, precision, recall, and F1-score on the test dataset.

4. **Prediction**
   - The trained model is used to predict the class (malignant/benign) of new data.

## Requirements

To run this project, you'll need to install the following dependencies:
```bash
pip install -r requirements.txt
```

You can create a `requirements.txt` file containing:
```
numpy
pandas
scikit-learn
tensorflow or keras
matplotlib
```

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/kabirkohli123/BreastCancerClassificationWithNeuralNetwork.git
   cd BreastCancerClassificationWithNeuralNetwork
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter notebook:
   ```bash
   jupyter notebook BreastCancerClassificationWithNeuralNetwork.ipynb
   ```

4. Run all the cells in the notebook to train the model and make predictions.

## Results

The model achieves a classification accuracy of approximately XX% on the test set. (You can add more details about your results here.)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset used in this project is available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).
- Thanks to the developers of TensorFlow, Keras, and scikit-learn for providing such useful libraries.
