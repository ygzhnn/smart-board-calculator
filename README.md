# Smart Board Calculator

A web application for recognizing and generating handwritten digits and basic mathematical expressions, designed for elementary school students. The project leverages deep learning models for digit recognition and generation, providing an interactive and educational experience.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Future Improvements](#future-improvements)

---

## Overview
This application allows students to draw mathematical expressions (e.g., `6 + 2`) on a digital canvas. The app recognizes the handwritten digits and operators, evaluates the expression, and displays the result. It also includes a generative model to create synthetic digit images, making math learning engaging and fun.

## Features
- **Handwritten Digit Recognition:** Uses a trained CNN to recognize digits from user drawings.
- **Digit Image Generation:** Employs a GAN-based generator to create new digit images.
- **Interactive Web Interface:** Built with Streamlit for easy use in classrooms or at home.
- **Educational Focus:** Designed to help students practice and verify math problems interactively.

## Project Structure
- `main.py` — Streamlit web app for drawing, recognition, and result display.
- `model.py` — Contains the CNN and GAN model definitions.
- `train.py` — Training script for both digit recognition and generation models.
- `requirements.txt` — List of required Python packages.
- `DigitClassification.ipynb` — Jupyter notebook for digit classification experiments and training.
- `DigitGeneration.ipynb` — Jupyter notebook for digit image generation experiments and training.
- `test_image.png` — Example/test image for model evaluation.

## Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd smart-board-calculator
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Train the models (optional):**
   - Run `train.py` to train or retrain the digit recognition and generation models.
   - Alternatively, use the provided Jupyter notebooks for step-by-step training and experimentation.
2. **Start the web app:**
   ```bash
   streamlit run main.py
   ```
3. **Interact:**
   - Draw a math expression on the canvas.
   - Click the "Show Result" button to see the recognized expression and its result.
   - View generated digit images for additional practice or visualization.

## Notebooks
- **DigitClassification.ipynb:**
  - Data loading, preprocessing, and CNN training for digit recognition.
  - Includes code for dataset download (Kaggle), exploration, and evaluation.
- **DigitGeneration.ipynb:**
  - GAN-based digit image generation.
  - Training and visualization of generated samples.

## Future Improvements
- **Operator Recognition:** Extend recognition to include handwritten operators (+, -, ×, ÷).
- **Step-by-Step Feedback:** Guide students through multi-step problems with hints.
- **Audio Assistance:** Integrate text-to-speech for auditory feedback.
- **Performance Improvements:** Optimize models for faster inference and better accuracy.

---
    
