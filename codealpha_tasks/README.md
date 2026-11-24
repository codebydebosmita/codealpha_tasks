# CodeAlpha Tasks

## 1. Handwritten Character Recognition (`handwriting_project/`)
CNN trained on EMNIST (balanced, 47 classes) in PyTorch, with image preprocessing,
fine-tuning on custom handwriting, and a CRNN model for word/line recognition.

    cd handwriting_project
    pip install -r requirements.txt
    python handwritten_recognition.py train --epochs 10
    python handwritten_recognition.py predict my_char.png

## 2. Credit Scoring Model (`credit_project/`)
Logistic Regression, Decision Tree and Random Forest compared with Precision, Recall,
F1 and ROC-AUC, plus a 300-850 scorecard-style credit score.

    cd credit_project
    pip install -r requirements.txt
    python credit_scoring.py train
    python credit_scoring.py predict my_applicants.csv
