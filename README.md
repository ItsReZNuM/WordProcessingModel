# 🚀 Persian Text Classifier  
A machine learning project for **automatic topic classification** of Persian text using **SVM + TF-IDF**, custom preprocessing, and a modern desktop UI built with **CustomTkinter**.

---

## ✨ Features  
- 🔤 **Advanced Persian text preprocessing**  
  - Emoji normalization  
  - Finglish → Persian conversion  
  - English → Persian replacement  
  - Number normalization  
  - Scientific term handling  
  - Link detection  
  - Typo correction  
  - Arabic → Persian character fixing  

- 🤖 **Machine Learning Models**  
  - Logistic Regression  
  - Naive Bayes  
  - **Linear SVM (Best)**  

- 🧠 **High accuracy:** 97% – 98%  
- 🖥️ **Desktop UI** using CustomTkinter  
- 📝 Fully modular project structure  
- 🧪 Includes unit tests for preprocessing  

---

## 📂 Project Structure  

```
WordProcessingModel/
│
├── data/
│   ├── Dataset.xlsx
│   └── clean_dataset.csv
│
├── models/
│   ├── svm_model.joblib
│   └── tfidf_vectorizer.joblib
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── preprocessor.py
│   ├── model.py
│   └── utils.py
│
├── tests/
│   └── test_preprocessor.py
│
├── assets/
│   └── fonts/
│       └── Vazir-FD-WOL.ttf
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🧪 Preprocessing Module  

The `PersianPreprocessor` performs:  
- Cleaning  
- Normalization  
- Emoji mapping  
- Laugh normalization  
- Scientific term replacement  
- Finglish conversion  
- More advanced NLP-safe transformations  

---

## 🤖 Training Notebook (exploration.ipynb)

This notebook includes:  
- EDA  
- Visualizations  
- TF-IDF vectorization  
- Model training  
- Evaluation  
- Export of trained models  

---

## 🖥️ Desktop Application (main.py)

A fully functional **CustomTkinter** UI:  
- Right-to-left input  
- Persian font support (Vazir FD-WOL)  
- Live topic classification  
- Clean and modern design  

---

## 🔧 Installation  

```bash
git clone https://github.com/ItsReZNuM/WordProcessingModel
cd WordProcessingModel
pip install -r requirements.txt
```

---

## ▶️ Running the Desktop App  

```bash
python main.py
```

---

## 🧪 Running Tests  

```bash
pytest
```

---

## 📈 Model Performance  

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 0.967 |
| Naive Bayes | 0.977 |
| **SVM** | **0.980** |

SVM is used as the final production model.

