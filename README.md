1# 🔍 Credit Card Fraud Detection

> ML-powered fraud detection system using Logistic Regression on real anonymized transaction data. Achieves 99.08% train accuracy with an interactive web interface deployed on GitHub Pages.

![Model](https://img.shields.io/badge/Model-Logistic%20Regression-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-99.08%25-brightgreen)
![Deploy](https://img.shields.io/badge/Deploy-GitHub%20Pages-orange)
![Python](https://img.shields.io/badge/Python-3.12-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🌐 Live Demo

**[View Live on GitHub Pages →](https://yourusername.github.io/your-repo-name)**

---

## 📸 Preview

> Interactive fraud detection interface with real-time confidence scoring and system log.

---

## 📌 About

Credit card fraud costs billions of dollars globally every year. This project builds an end-to-end machine learning pipeline that:

1. Trains on **real anonymized transaction data**
2. Handles severe **class imbalance** using undersampling
3. Deploys a **browser-based interface** anyone can use — no backend needed

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total Transactions | 11,959 |
| Legitimate | 11,906 |
| Fraudulent | 52 |
| Features | 30 (Time, V1–V28, Amount) |
| Label Column | Class (0 = Legit, 1 = Fraud) |

> **V1–V28** are PCA-transformed features. Original details (merchant, location, card number) were anonymized to protect cardholder privacy.

---

## ⚙️ How It Works

### 1. Data Preprocessing
- Load the dataset and check for missing values
- Separate legitimate (Class=0) and fraud (Class=1) transactions
- Analyze statistical distributions of both classes

### 2. Handling Class Imbalance
The dataset is heavily imbalanced — only 0.43% fraud. Training directly on this would cause the model to ignore fraud entirely.

**Solution: Undersampling**
```python
legit_sample = legit.sample(n=492)
new_dataset = pd.concat((legit_sample, fraud), axis=0)
# Result: 544 balanced samples
```

### 3. Model Training
```python
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

model = LogisticRegression()
model.fit(X_train, Y_train)
```

### 4. Prediction
```python
# 0 = Legitimate, 1 = Fraud
prediction = model.predict(input_data)
```

---

## 📈 Results

| Metric | Score |
|---|---|
| Training Accuracy | **99.08%** |
| Test Accuracy | **100%** |
| Train Samples | 435 |
| Test Samples | 109 |
| Train/Test Split | 80% / 20% stratified |

---

## 🖥️ Frontend

The `index.html` file is a fully self-contained web app that runs entirely in the browser.

**Features:**
- Input all 30 transaction features manually
- One-click load of real **Legit** or **Fraud** sample transactions
- Instant verdict: ✅ LEGITIMATE or ⚠ FRAUD DETECTED
- Animated confidence bars showing fraud vs. legitimate probability
- Live system log with timestamps
- Zero dependencies — pure HTML, CSS, JavaScript

---

## 🗂️ Project Structure

```
📦 credit-card-fraud-detection
 ┣ 📄 index.html          # Frontend web app (GitHub Pages)
 ┣ 📓 notebook.ipynb      # ML model training (Google Colab)
 ┣ 📄 README.md           # You are here
 ┗ 📄 creditcard.csv      # Dataset (add manually — too large for GitHub)
```

> ⚠️ The dataset file is not included due to size. Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root folder.

---

## 🚀 Deploy on GitHub Pages

1. Fork or clone this repository
2. Go to **Settings → Pages**
3. Set source to **Deploy from branch → main → / (root)**
4. Click **Save**
5. Your site will be live at `https://yourusername.github.io/repo-name`

---

## 🛠️ Run the Notebook Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install numpy pandas scikit-learn

# Open the notebook
jupyter notebook notebook.ipynb
```

Or run directly on **[Google Colab](https://colab.research.google.com/)** — no installation needed.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML Library | scikit-learn |
| Data | NumPy, pandas |
| Notebook | Google Colab |
| Frontend | HTML, CSS, JavaScript |
| Deployment | GitHub Pages |

---

## 🔑 Key Concepts

- **Logistic Regression** — Binary classification algorithm that outputs a probability score
- **PCA (Principal Component Analysis)** — Dimensionality reduction used to anonymize original features
- **Undersampling** — Reducing the majority class to balance the dataset
- **Stratified Split** — Ensures both train and test sets have the same fraud/legit ratio
- **Sigmoid Function** — Converts raw model output into a 0–1 probability

---

## 👤 Author

**Your Name**
- GitHub: https://github.com/SumanRaj93
- LinkedIn: https://www.linkedin.com/in/suman-raj-746727307

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

> ⭐ If you found this project helpful, please consider giving it a star!
