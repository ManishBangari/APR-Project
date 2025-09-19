# APR Project

## Logistic Regression for Diabetes Prediction

This project demonstrates building a **logistic regression model** from scratch and comparing it with scikit-learn's implementation to predict diabetes using the Pima Indians Diabetes Dataset.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ManishBangari/APR-Project.git
cd APR-Project
```

### 2. Download the dataset

Download from:
[proc\_pima\_2\_withheader.csv](https://github.com/ManishBangari/APR-Project/blob/main/Data/proc_pima_2_withheader.csv)
Place the CSV in the `Data/` folder.

### 3. Create a Python virtual environment

```bash
python3 -m venv myvenv
source myenv/bin/activate   # Linux/Mac
myenv\Scripts\activate      # Windows
```

### 4. Install required packages

```bash
pip install ipykernel jupyter pandas numpy matplotlib seaborn scikit-learn
```

---

## Running the Project

1. Navigate to the `src/` folder:

```bash
cd src
```

2. Run the main script:

```bash
python main.py
```

This will:

* Load and preprocess the dataset
* Train logistic regression from scratch and with scikit-learn
* Evaluate the models using accuracy, precision, recall, and F1-score
* Generate and save plots in `../Plots/` (Confusion Matrix, ROC Curve, Precision-Recall Curve)

---

## Exploratory Data Analysis (EDA)

* The EDA notebook is available in the `Notebooks/` folder.
* It includes:

  * Checking for missing values
  * Statistical summaries
  * Feature scaling and distribution plots
  * Target variable distribution

📌 The EDA helps understand the dataset before training the model.

---

## Results

* Scratch and scikit-learn implementations produce the same results:

  * **Accuracy:** 0.8228
  * **Precision:** 0.7778
  * **Recall:** 0.7241
  * **F1 Score:** 0.7500

* Plots for evaluation metrics are saved in the `Plots/` folder.

---

## Notes

* Ensure that the `Diabetes` column in the dataset is **binary (0 = non-diabetic, 1 = diabetic)**.
* The main script uses **gradient descent** for training the scratch model.
* Plots are saved automatically; no GUI is required.