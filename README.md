Calculator Recognition Web App

This project is a simple web application designed for elementary school students to recognize handwritten numbers and perform basic mathematical operations. The application provides an engaging and fun way for students to learn and check their math results.
Overview

The app allows students to draw mathematical expressions (like 6 + 2) and get the result by clicking a button to check if the calculation is correct. The current version of the project focuses on recognizing and generating numbers. It includes the following functionality:

    Number Recognition: The app can recognize handwritten numbers.

    Number Generation: The app can generate numbers using a trained Generative Adversarial Network (GAN).

The project is designed for educational purposes, making math learning easier and more enjoyable.
How It Works

    Draw the Expression: The student draws a math expression, such as 6 + 2, using a drawing canvas.

    Click the "Show Result" Button: After drawing, the student clicks the "Show Result" button to get the math expression and the result.

    Result and Image Generation: The app will show the result of the mathematical expression, and the corresponding image of the number will be generated using a generator model.

Current Features

    Handwritten Digit Recognition: Based on the trained CNN model.

    Image Generation: Using the trained Generator model (a GAN) to produce a visual representation of numbers.

Requirements

Before running this app, you will need to install the following dependencies:

    Streamlit: A framework to create the web application.

    PyTorch: For running the models.

    TorchVision: For image transformations and model definitions.

    PIL (Pillow): For image processing.

    NumPy: For handling arrays.

    Matplotlib: For displaying generated images.

Installing Dependencies

You can install all dependencies using the following command:

pip install -r requirements.txt

Future Improvements

In future iterations, we plan to extend this application with additional features:


    Interactive Feedback: The app will be able to guide students through solving the problems by providing hints and suggestions.

    Audio Assistance: Integration of text-to-speech to assist students with learning through auditory feedback.
    
