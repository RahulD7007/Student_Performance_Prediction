# Student Performance Prediction using Decision Tree Classifier

A Machine Learning classification project that predicts whether a student will **Pass** or **Fail** based on academic and behavioral attributes.

The project demonstrates a complete Machine Learning workflow, including Exploratory Data Analysis (EDA), data visualization, Decision Tree model building, model evaluation, feature importance analysis, model optimization, and prediction on unseen student records.

This project was developed using **Python**, **Pandas**, **Matplotlib**, and **Scikit-learn** following industry-standard Machine Learning practices.

## Project Features

- Performed Exploratory Data Analysis (EDA) to understand the dataset.
- Generated statistical summaries and analyzed feature distributions.
- Visualized the dataset using histograms, scatter plots, and box plots.
- Built a Decision Tree Classifier to predict student performance.
- Evaluated the model using Accuracy Score, Confusion Matrix, and Classification Report.
- Visualized the trained Decision Tree.
- Analyzed feature importance to identify influential features.
- Conducted feature selection experiments to study model performance.
- Compared different `max_depth` values of the Decision Tree.
- Compared different `random_state` values to observe the effect of train-test splitting.
- Predicted the performance of new student records.

## Dataset Information

The project uses the **Student Performance Dataset**, which contains academic and behavioral information about students.

### Input Features

| Feature | Description |
|---------|-------------|
| StudyHours | Number of hours studied per day |
| Attendance | Student attendance percentage |
| PreviousScore | Score obtained in the previous examination |
| AssignmentsCompleted | Number of assignments completed |
| SleepHours | Average hours of sleep per day |

### Target Variable

| Target | Meaning |
|--------|---------|
| 1 | Pass |
| 0 | Fail |

## Project Structure

```text
Student-Performance-Prediction/
│
├── data/
│   └── student_performance_ml.csv
│
├── src/
│   └── student_performance_prediction.py
│
├── assets/
│   ├── study_hours_distribution.png
│   ├── study_hours_vs_previous_score.png
│   ├── attendance_boxplot.png
│   ├── assignments_vs_finalresult.png
│   ├── sleep_hours_vs_finalresult.png
│   ├── confusion_matrix.png
│   ├── decision_tree.png
│   └── feature_importance.png
│
├── README.md
└── requirements.txt
```
## Machine Learning Workflow

```text

Student Performance Dataset
            │
            ▼
Data Loading
            │
            ▼
Exploratory Data Analysis (EDA)
            │
            ▼
Statistical Analysis
            │
            ▼
Data Visualization
            │
            ▼
Feature Selection
            │
            ▼
Train-Test Split
            │
            ▼
Decision Tree Model
            │
            ▼
Model Training
            │
            ▼
Prediction
            │
            ▼
Model Evaluation
            │
            ▼
Feature Importance Analysis
            │
            ▼
Model Optimization
            │
            ▼
Prediction on New Student Records

```


## Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model | Decision Tree Classifier |
| Development Environment | Visual Studio Code |
| Version Control | Git & GitHub |

## Project Results

The Decision Tree model performed well on the given dataset and successfully classified student performance.

### Key Findings

- Attendance was identified as the most influential feature.
- Removing the **SleepHours** feature did not affect the model accuracy.
- Training the model using only **StudyHours** and **Attendance** achieved similar performance on this dataset.
- Different `random_state` values resulted in slight variations in testing accuracy because of different train-test splits.
- Increasing the tree depth did not improve the model performance, indicating that a simple Decision Tree was sufficient.

## Author

**Rahul Divekar**


