# Naive Bayes Classifier Visualization

This project demonstrates the application of a Naive Bayes classifier on a synthetic dataset, with visualizations of classification regions and probability contours. The project is built using `scikit-learn`, `matplotlib`, and `numpy` libraries.

## Table of Contents
- [Overview](#overview)
- [Concepts](#concepts)
- [Variables and Parameters](#variables-and-parameters)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Visualization](#visualization)
- [Applications](#applications-of-naive-bayes)
- [License](#license)

## Overview
The Naive Bayes classifier is a probabilistic classification algorithm based on Bayes' Theorem, assuming that the features are conditionally independent given the class label. In this project, we:
- Train a Gaussian Naive Bayes classifier on a synthetic 2D dataset.
- Visualize the classification regions showing how the model divides the feature space.
- Plot probability contours indicating the confidence of the model in different regions.

This project serves as an educational tool to better understand how the Naive Bayes classifier works and its decision-making process.

## Concepts

### Naive Bayes Classifier
The Naive Bayes classifier is a simple probabilistic model used for classification tasks. It is based on Bayes' Theorem:
```
P(C|X) = [P(X|C) * P(C)] / P(X)
```
Where:
- `P(C|X)` is the posterior probability (the probability of class `C` given the input data `X`).
- `P(X|C)` is the likelihood (the probability of observing `X` given class `C`).
- `P(C)` is the prior probability of class `C`.
- `P(X)` is the marginal likelihood (the total probability of observing `X` across all classes).

Naive Bayes assumes **conditional independence** between features, i.e., the value of one feature does not depend on the value of another, given the class label.

### Gaussian Naive Bayes
In the Gaussian Naive Bayes model, we assume that the data for each feature follows a normal (Gaussian) distribution:
```
P(X_i|C) = (1 / √(2πσ^2)) * exp(-(X_i - μ)^2 / 2σ^2)
```
Where `μ` and `σ^2` are the mean and variance of the feature `X_i` for class `C`. This makes Gaussian Naive Bayes suitable for continuous data.

### Decision Boundaries and Probability Contours
- **Decision boundaries** are the boundaries where the classifier changes its prediction from one class to another.
- **Probability contours** represent regions of the feature space where the classifier's confidence in predicting a certain class is high.

## Variables and Parameters

### Dataset
- **X**: Input features. In this project, the dataset contains two informative features generated synthetically using the `make_classification` function from `scikit-learn`.
- **y**: Target labels (class labels). The dataset is binary, with two classes: `0` and `1`.

### Model Parameters
- **GaussianNB()**: The Naive Bayes classifier used in this project. The Gaussian version assumes that the continuous features follow a normal distribution.
  
### Visualization Parameters
- **xx, yy**: Grid of feature values used to plot the decision boundaries and probability contours.
- **Z**: Predicted class labels or probabilities for each point on the grid, used for contour plotting.

### Training Parameters
- **X_train, X_test**: Training and testing subsets of the dataset created using `train_test_split`.
- **scaler**: The `StandardScaler` is used to normalize the dataset for better performance.

### Naive Bayes Assumptions
- **Conditional Independence**: Naive Bayes assumes that the features are conditionally independent given the class label, which simplifies the computation of the likelihood.

## Prerequisites
Before running the project, ensure that you have the following Python libraries installed:
- `scikit-learn`
- `matplotlib`
- `numpy`

You can install them using:
```bash
pip install scikit-learn matplotlib numpy
```

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/naive-bayes-visualization.git
   cd naive-bayes-visualization
   ```

2. **Run the Script**:
   To train the Naive Bayes classifier and visualize the results, run the Python script:
   ```bash
   python naive_bayes_visualization.py
   ```

3. **Modify Parameters**:
   You can modify the dataset size, Gaussian Naive Bayes parameters, and visualization settings within the script.

## Visualization

### 1. **Decision Boundaries**
The decision boundary plot visualizes how the Naive Bayes classifier divides the feature space between the two classes based on the training data.

Decision Boundary:

![Decision Boundary](https://github.com/AartiDashore/NaiveBayesClassificationVisualization/blob/main/Output1.png)

### 2. **Probability Contours**
The probability contour plot shows the predicted probability of the positive class across the feature space. The darker regions indicate higher probabilities, and the contour lines show where the probabilities are equal.

Probability Contours:

![Probability Contours](https://github.com/AartiDashore/NaiveBayesClassificationVisualization/blob/main/Output2.png)

## Applications of Naive Bayes

Naive Bayes classifiers are widely used in various real-world applications where the independence assumption holds or works well in practice. Some notable applications include:

1. **Spam Detection**: Naive Bayes is highly effective for email spam filtering by classifying emails as spam or not based on word frequency in the message content.

2. **Text Classification**: In natural language processing (NLP), Naive Bayes is often used for document classification tasks like sentiment analysis, categorizing news articles, and topic modeling.

3. **Sentiment Analysis**: Naive Bayes is frequently used for sentiment analysis on product reviews, movie reviews, or social media posts by analyzing text features and classifying them as positive, negative, or neutral.

4. **Medical Diagnosis**: In healthcare, Naive Bayes can be used to predict the likelihood of diseases based on symptoms. For example, it can help diagnose conditions like cancer, diabetes, or heart disease by considering different health indicators.

5. **Recommendation Systems**: Naive Bayes is applied in recommendation systems to predict user preferences based on past behavior, such as recommending products, movies, or content based on user interactions.

6. **Fraud Detection**: Naive Bayes can help detect fraudulent transactions by analyzing patterns in transactional data. It classifies each transaction as either fraudulent or non-fraudulent based on factors like amount, location, and merchant.

7. **Customer Classification**: Businesses use Naive Bayes to segment customers into different categories (e.g., high-value vs. low-value) based on purchasing behavior, demographics, and other factors.

### Advantages of Naive Bayes:
- **Simplicity**: Naive Bayes is easy to implement and interpret.
- **Efficiency**: It works well with high-dimensional data and is computationally efficient.
- **Robust to Irrelevant Features**: Even when the independence assumption is violated, Naive Bayes performs surprisingly well in practice.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
