\documentclass[a4paper,12pt]{report}

% Packages
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{float}

% Page layout
\geometry{top=1in, bottom=1in, left=1in, right=1in}

% Title Page
\title{Final Project Report – Celebrity Search - Dabeet}
\author{Your Name}
\date{January 2025}

\begin{document}

% Title Page
\maketitle
\newpage

% Abstract
\begin{abstract}
    This report outlines the development of a Convolutional Neural Network (CNN) aimed at classifying images of five celebrities: Angelina Jolie, Leonardo DiCaprio, Robert Downey Jr., Tom Cruise, and Megan Fox. The model utilizes TensorFlow and Keras to classify images based on the Kaggle celebrity image dataset. The report discusses data preprocessing, model architecture, experiments, and the final performance of the model.
\end{abstract}
\newpage

% Table of Contents
\tableofcontents
\newpage

% Chapter 1 - Introduction
\chapter{Introduction}
\section{Task}
The task was to develop an accurate convolutional neural network (CNN) for classifying images of celebrities. The Kaggle dataset was used, which contains images of various celebrities, but for this project, we focused on only five: Angelina Jolie, Leonardo DiCaprio, Robert Downey Jr., Tom Cruise, and Megan Fox.

\section{Dataset}
The dataset used for this project is from Kaggle: \url{https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset}. The dataset contains images of celebrities, which were filtered and preprocessed to include only the five celebrities mentioned above.

\section{Google Colab Notebook}
The model was developed and trained using Google Colab. The notebook can be accessed via this link: \url{http://bit.ly/4h6elpq}.

\chapter{Model Development}
\section{Libraries Used}
The following libraries were used in the development of this model:
\begin{itemize}
    \item Kaggle
    \item OS
    \item Shuffle
    \item TensorFlow
    \item Matplotlib
    \item Numpy
    \item Sklearn
    \item Seaborn
    \item Google Colab
\end{itemize}

\section{Loading the Data}
To load the dataset into the Google Colab environment, the Kaggle API key was uploaded. The Kaggle library was then installed using \texttt{pip install kaggle}, and the API key was set up. The dataset was downloaded directly using Kaggle API commands.

\section{Data Exploration and Preprocessing}
Upon downloading the dataset, several preprocessing steps were necessary:
\begin{itemize}
    \item The dataset was unzipped and renamed for easier handling.
    \item A loop was used to filter and delete directories of unwanted celebrities.
    \item The data was split into training (80\%) and validation (20\%) datasets.
    \item Images were resized to 224x224 to ensure uniform input size.
    \item Data augmentation was introduced to add randomness, but it increased the complexity of training, requiring more epochs and leading to minimal performance gain.
\end{itemize}

\section{Model Architecture and Compilation}
The model architecture consists of the following:
\begin{itemize}
    \item 1 Convolutional layer with 64 neurons and a 3x3 kernel
    \item 1 Pooling layer (Max Pooling with 2x2 kernel)
    \item Flattening layer
    \item 2 Dense layers: One with 128 neurons and ReLU activation, and the output layer with 5 neurons (one for each celebrity) using the Softmax activation function.
\end{itemize}

The optimizer used is Adam with a learning rate of 0.0001 and a beta-1 momentum of 0.95. The loss function is Sparse Categorical Cross-Entropy, suitable for integer labels.

\section{Training and Evaluation}
The model was trained with an 80-20 test-train split, and the accuracy and loss were tracked over 14 epochs, achieving a final accuracy of 95\% with a loss of 0.4907. The early stopping strategy was implemented to prevent overfitting, with patience set to 5 epochs.

\chapter{Experiments and Observations}
\section{Experiment 1: Parameters: 25 Epochs, Default Momentum and Learning Rate}
\textbf{Observation}: The model overfitted, performing well only on the training dataset.

\section{Experiment 2: Using Stochastic Gradient Descent instead of Adam}
\textbf{Observation}: The accuracy dropped significantly, and the model failed to improve over epochs.

\section{Experiment 3: Adam Optimizer with Learning Rate = 0.01, 2 Convolutional Layers, and 2 Pooling Layers}
\textbf{Observation}: The model achieved 100\% accuracy, indicating overfitting.

\section{Experiment 4: Adjusting the Learning Rate to 0.002}
\textbf{Observation}: The loss curve became uneven, indicating overfitting.

\section{Experiment 5: Adam Optimizer with Learning Rate 0.002 and 2 Convolutional Layers}
\textbf{Observation}: The results were similar, with no smooth loss curve.

\section{Experiment 6: Adjusting Learning Rate to 0.0001 and Beta-1 Momentum to 0.95}
\textbf{Observation}: This helped smooth the loss curve and minimized overfitting.

\chapter{Conclusion}
The final model achieved a high accuracy of 95\% after 14 epochs. The use of early stopping, proper adjustments to the optimizer, and data augmentation techniques helped the model perform well while avoiding overfitting.

\newpage

% References
\begin{thebibliography}{9}
    \bibitem{reference1} Author Name, \textit{Title of the Paper}, Journal Name, Year.
    \bibitem{reference2} Author Name, \textit{Title of the Book}, Publisher, Year.
\end{thebibliography}

\end{document}
