####################################################################################################
#
# Project Name : Student Performance Prediction
#
# File Name    : student_performance_prediction.py
#
# Description  :
# This project performs end-to-end Machine Learning on the Student Performance
# dataset. It includes data exploration, statistical analysis, visualization,
# Decision Tree classification, model evaluation, feature importance analysis,
# model optimization, and prediction of student academic performance.
#
# Dataset :
# student_performance_ml.csv
#
# Problem Statement :
# Predict whether a student will Pass (1) or Fail (0) based on academic and
# behavioral attributes.
#
# Features :
# • StudyHours
# • Attendance
# • PreviousScore
# • AssignmentsCompleted
# • SleepHours
#
# Target Variable :
# • FinalResult
#       1 → Pass
#       0 → Fail
#
# Libraries Used :
# • pandas
# • numpy
# • matplotlib
# • seaborn
# • scikit-learn
#
# Project Workflow :
# 1. Import Required Libraries
# 2. Load Dataset
# 3. Data Exploration
# 4. Statistical Analysis
# 5. Data Visualization
# 6. Data Preprocessing
# 7. Decision Tree Model Training
# 8. Model Evaluation
# 9. Feature Importance Analysis
# 10. Model Optimization
# 11. Prediction on New Data
# 12. Conclusion
#
# Author : Rahul Divekar
#
####################################################################################################

###############################################################
# Import Required Libraries
###############################################################
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
################################################################################
# Project Constants
################################################################################

Section_Separator = "=" * 80
Subsection_Separator = "-"*40
DATASET_PATH = "../data/student_performance_ml.csv"
Save_Figures = False
ASSETS_PATH = "../assets/"

################################################################################
# Step 1 : Load the Student Performance Dataset
#
# Objective:
# Read the dataset into a Pandas DataFrame.
################################################################################

print(Section_Separator)
print("STEP 1 : LOAD THE STUDENT PERFORMANCE DATASET")
print(Section_Separator)

student_df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")

print("\nFirst 5 records of Dataset :\n")
print(student_df.head())

print("\nLast 5 records of Dataset :\n")
print(student_df.tail())

################################################################################
# Step 2 : Exploratory Data Analysis (EDA)
#
# Objective:
# Understand the structure, quality, and characteristics of the dataset
################################################################################

print("\n")
print(Section_Separator)
print("STEP 2 : EXPLORATORY DATA ANALYSIS (EDA)")
print(Section_Separator)

#Display Dataset Shape
print("\nDataset Shape")
print("-"*30)

rows, columns = student_df.shape

print(f"Number of Rows     : {rows}")
print(f"Number of Columns  : {columns}")

#Display Column Names
print("\nColumn Names")
print("-"*30)

for column in student_df.columns:
    print(f"- {column}")

#Display Data Types
print("\nData Types")
print("-"*30)

print(student_df.dtypes)

#Check Missing Values
print("\nMissing Values")
print("-"*30)

print(student_df.isnull().sum())

#Display Dataset Information
print("\nDataset Information")
print("-"*30)

student_df.info()

#Statistical Summary
print("\nStatistical Summary")
print("-"*30)

print(student_df.describe())

################################################################################
# Step 3 : Statistical Analysis
#
# Objective:
# Perform statistical analysis on the dataset to understand student
# performance and feature distributions.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 3 : STATISTICAL ANALYSIS")
print(Section_Separator)

#Total Students
print("\n Total Students")
print(Subsection_Separator)

total_students = len(student_df)

print(f"Total Students : {total_students}")

#Pass-Fail count
print("\nStudent Result Distribution")
print(Subsection_Separator)

passed_students = (student_df["FinalResult"]==1).sum()
failed_students = (student_df["FinalResult"]==0).sum()

print(f"Passes students : {passed_students}")
print(f"Failed students : {failed_students}")

#Avg study Hrs
print("\nAverage Study Hours")
print(Subsection_Separator)

avg_study_hrs = student_df["StudyHours"].mean()

print(f"Average study Hours : {avg_study_hrs:.2f}")

#Avg Attendence
print("\nAverage Attendence")
print(Subsection_Separator)

avg_attendence = student_df["Attendance"].mean()

print(f"Average Attendence : {avg_attendence:.2f}%")

#Maximum previous Score
print("\nHighest Previous Score")
print(Subsection_Separator)

highest_score = student_df["PreviousScore"].max()

print(f"Highest Previous Score : {highest_score}")

#Lowest sleep hrs
print("\nLowest Sleep Hours")
print(Subsection_Separator)

minimum_sleep = student_df["SleepHours"].min()

print(f"Minimum Sleep Hours : {minimum_sleep}")

#Result Distribution
print("\nFinal Result Distribution")
print(Subsection_Separator)

result_distribution = student_df["FinalResult"].value_counts()

print(result_distribution)

#Dataset Balance
print("\nDataset Balance Analysis")
print(Subsection_Separator)

if abs(passed_students - failed_students) <=10:
    print("The dataset is approximately balanced")
else:
    print("The dataset is imbalanced")

################################################################################
# Step 4 : Data Visualization
#
# Objective:
# Visualize the distribution of important features and identify
# relationships between variables affecting student performance.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 4 : DATA VISUALIZATION")
print(Section_Separator)

#Visualization 1 : Histogram
print(f"\n{' Histogram - Study Hours Distribution ':=^80}")

plt.figure(figsize=(8, 5))

plt.hist(
    student_df["StudyHours"],
    bins=10,
    edgecolor="black"
)

plt.title("Study Hours Distribution")
plt.xlabel("Study Hours")
plt.ylabel("Number of Students")

plt.grid(alpha=0.3)

plt.tight_layout()

if Save_Figures:
    plt.savefig(
        "../assets/study_hours_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()

################################################################################
# Observation:
#
# The histogram shows how study hours are distributed among students.
# It helps identify whether most students study for fewer or greater
# numbers of hours each day.
################################################################################

#Visualization 2 : Scatter Plot
print(f"\n{' Scatter Plot - Study Hours vs Previous Score ':=^80}")

plt.figure(figsize=(8,5))

plt.scatter(
    student_df["StudyHours"],
    student_df["PreviousScore"]
)

plt.title("Study Hours vs Previous Score")

plt.xlabel("Study Hours")

plt.ylabel("Previous Score")

plt.grid(alpha=0.3)

plt.tight_layout()

if Save_Figures:
    plt.savefig(
        ASSETS_PATH + "study_hours_vs_previous_score.png",
        dpi=300,
        bbox_inches="tight"
    )
plt.show()


#Visualization 3 : Box Plot
print(f"\n{' Box Plot - Attendance ':=^80}")

plt.figure(figsize=(8,5))

plt.boxplot(student_df["Attendance"])

plt.title("Attendance Box Plot")
plt.ylabel("Attendance")

plt.grid(alpha=0.3)

plt.tight_layout()

if Save_Figures:
    plt.savefig(
        ASSETS_PATH + "attendance_boxplot.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()

print("\nObservation")
print(Subsection_Separator)

print("- The box plot helps identify the distribution of attendance values.")
print("- It also helps detect any potential outliers in attendance.")


#Visualization 4 : Assignments Completed vs Final Result
print(f"\n{' Assignments Completed vs Final Result ':=^80}")

assignment_result = student_df.groupby("FinalResult")["AssignmentsCompleted"].mean()

plt.figure(figsize=(8,5))

plt.bar(
    ["Fail","Pass"],
    assignment_result
)

plt.title("Assignments Completed vs Final Result")
plt.xlabel("Final Result")
plt.ylabel("Average Assignments Completed")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

if Save_Figures:
    plt.savefig(
        ASSETS_PATH + "assignments_vs_finalresult.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()

print("\nObservation")
print(Subsection_Separator)

print("- Students who passed generally completed more assignments.")


# Visualization 5 : Sleep Hours vs Final Result
print(f"\n{' Sleep Hours vs Final Result ':=^80}")

sleep_result = student_df.groupby("FinalResult")["SleepHours"].mean()

plt.figure(figsize=(8,5))

plt.bar(
    ["Fail","Pass"],
    sleep_result
)

plt.title("Sleep Hours vs Final Result")
plt.xlabel("Final Result")
plt.ylabel("Average Sleep Hours")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

if Save_Figures:
    plt.savefig(
        ASSETS_PATH + "sleep_hours_vs_finalresult.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()

print("\nObservation")
print(Subsection_Separator)

print("- Sleep hours alone do not show a strong impact on student performance.")


################################################################################
# Step 5 : Feature Selection
#
# Objective:
# Separate the independent (input) features and the dependent
# (target) variable for Machine Learning.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 5 : FEATURE SELECTION")
print(Section_Separator)

# Independent variables (Features)
feature_columns = [
    "StudyHours",
    "Attendance",
    "PreviousScore",
    "AssignmentsCompleted",
    "SleepHours"
]

# Dependent variable (Target)
target_column = "FinalResult"

X = student_df[feature_columns]

Y = student_df[target_column]

print("\nIndependent Variables (Features)")
print(Subsection_Separator)

print(feature_columns)

print("\nTarget Variable")
print(Subsection_Separator)

print(target_column)

print("\nDataset Shape After Feature Selection")
print(Subsection_Separator)

print(f"Features Shape : {X.shape}")
print(f"Target Shape   : {Y.shape}")


################################################################################
# Step 6 : Split the Dataset into Training and Testing Sets
#
# Objective:
# Split the dataset into training and testing subsets to evaluate the
# performance of the Machine Learning model on unseen data.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 6 : TRAIN-TEST SPLIT")
print(Section_Separator)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=42)

print("\nDataset Split Summary")
print(Subsection_Separator)

print(f"Original Feature Matrix : {X.shape}")
print(f"Original Target Vector  : {Y.shape}")

print()

print(f"Training Features : {X_train.shape}")
print(f"Testing Features  : {X_test.shape}")

print()

print(f"Training Target   : {Y_train.shape}")
print(f"Testing Target    : {Y_test.shape}")

################################################################################
# Step 7 : Build the Decision Tree Model
#
# Objective:
# Create a Decision Tree Classifier that will learn patterns from
# the training dataset to predict student performance.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 7 : BUILD THE DECISION TREE MODEL")
print(Section_Separator)

decision_tree_model = DecisionTreeClassifier(random_state=42)

print("Decision Tree model created successfully.")

################################################################################
# Step 8 : Train the Decision Tree Model
#
# Objective:
# Train the Decision Tree model using the training dataset.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 8 : TRAIN THE MODEL")
print(Section_Separator)

decision_tree_model.fit(X_train, Y_train)

print("Model trained successfully.")

################################################################################
# Step 9 : Predict Student Performance
#
# Objective:
# Predict the class labels for the testing dataset using the trained model.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 9 : MODEL PREDICTION")
print(Section_Separator)

Y_pred = decision_tree_model.predict(X_test)

print("Prediction completed successfully.")

print("\nActual vs Predicted Results")
print(Subsection_Separator)

comparison_df = pd.DataFrame({
    "Actual": Y_test.values,
    "Predicted": Y_pred
})

print(comparison_df)

################################################################################
# Step 10 : Evaluate the Decision Tree Model
#
# Objective:
# Evaluate the performance of the trained Decision Tree model using
# different evaluation metrics.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 10 : MODEL EVALUATION")
print(Section_Separator)

#Accuracy Score
print("\nAccuracy Score")
print(Subsection_Separator)

accuracy = accuracy_score(Y_test, Y_pred)

print(f"Model Accuracy : {accuracy:.2%}")

#Confusion matrix
print("\nConfusion Matrix")
print(Subsection_Separator)

conf_matrix = confusion_matrix(Y_test, Y_pred)

print(conf_matrix)

# ==============================
# Confusion Matrix Visualization
# ==============================

ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=["Fail", "Pass"]
).plot()

plt.title("Confusion Matrix")

plt.tight_layout()

if Save_Figures:
    plt.savefig(
        ASSETS_PATH + "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()

print("\nClassification Report")
print(Subsection_Separator)

print(classification_report(Y_test, Y_pred))


#Training Accuracy
training_accuracy = decision_tree_model.score(X_train, Y_train)

#Testing Accuracy
testing_accuracy = decision_tree_model.score(X_test, Y_test)

print("\nTraining vs Testing Accuracy")
print(Subsection_Separator)

print(f"Training Accuracy : {training_accuracy:.2%}")
print(f"Testing Accuracy  : {testing_accuracy:.2%}")

################################################################################
# Step 11 : Decision Tree Visualization
#
# Objective:
# Visualize the trained Decision Tree to understand how it makes
# classification decisions.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 11 : DECISION TREE VISUALIZATION")
print(Section_Separator)

plt.figure(figsize=(18,10))

plot_tree(
    decision_tree_model,
    feature_names=feature_columns,
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree Classifier")

plt.tight_layout()

if Save_Figures:
    plt.savefig(
        ASSETS_PATH + "decision_tree.png",
        dpi=300,
        bbox_inches="tight"
    )


plt.show()


################################################################################
# Step 12 : Feature Importance Analysis
#
# Objective:
# Identify which features contribute the most to predicting
# student performance.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 12 : FEATURE IMPORTANCE")
print(Section_Separator)

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": decision_tree_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance)

plt.figure(figsize=(8,5))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=20)

plt.tight_layout()

if Save_Figures:
    plt.savefig(
        ASSETS_PATH + "feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()


################################################################################
# Step 13 : Feature Selection Experiment
#
# Objective:
# Remove the SleepHours feature and evaluate whether it affects
# the model's prediction performance.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 13 : REMOVE 'SleepHours' FEATURE")
print(Section_Separator)

# Create a new feature matrix without SleepHours

X_without_sleep = student_df.drop(
    columns=["SleepHours", "FinalResult"]
)

y_without_sleep = student_df["FinalResult"]

#Train-Test Split Again
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_without_sleep,
    y_without_sleep,
    test_size=0.20,
    random_state=42
)

new_decision_tree = DecisionTreeClassifier(random_state=42)

new_decision_tree.fit(X_train_new, y_train_new)

new_accuracy = new_decision_tree.score(
    X_test_new,
    y_test_new
)

print(f"\nAccuracy without SleepHours : {new_accuracy:.2%}")

#Compare with Previous Model

print("\nModel Comparison")
print(Subsection_Separator)

print(f"Original Model Accuracy : {testing_accuracy:.2%}")
print(f"Without SleepHours      : {new_accuracy:.2%}")


print("\nObservation")
print(Subsection_Separator)

if new_accuracy == testing_accuracy:
    print(" Removing the 'SleepHours' feature did not affect the model accuracy.")
    print(" This confirms that SleepHours has little or no influence on predicting")

else:
    print(" Removing the 'SleepHours' feature changed the model accuracy.")
    print(" Therefore, SleepHours contributes to predicting the student's final result.")

################################################################################
# Step 14 : Train the Model Using Two Features
#
# Objective:
# Train a Decision Tree model using only StudyHours and Attendance
# to evaluate whether these two features are sufficient for
# predicting student performance.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 14 : TRAIN MODEL USING TWO FEATURES")
print(Section_Separator)

# Select only two features

X_two_features = student_df[
    [
        "StudyHours",
        "Attendance"
    ]
]

y_two_features = student_df["FinalResult"]

#split the Data
X_train_two, X_test_two, y_train_two, y_test_two = train_test_split(
    X_two_features,
    y_two_features,
    test_size=0.20,
    random_state=42
)

#Creating new model 
two_feature_model = DecisionTreeClassifier(random_state=42)

#Train Model
two_feature_model.fit(
    X_train_two,
    y_train_two
)

#predict
y_pred_two = two_feature_model.predict(X_test_two)

#accuracy
two_feature_accuracy = accuracy_score(
    y_test_two,
    y_pred_two
)

print(f"\nAccuracy using only two features : {two_feature_accuracy:.2%}")

print("\nModel Comparison")
print(Subsection_Separator)

print(f"Original Model (5 Features)      : {testing_accuracy:.2%}")
print(f"Without SleepHours (4 Features)  : {new_accuracy:.2%}")
print(f"Only Two Features                : {two_feature_accuracy:.2%}")


print("\nObservation")
print(Subsection_Separator)

if two_feature_accuracy == testing_accuracy:
    print(" Using only StudyHours and Attendance achieved the same accuracy")
    print(" as the original model.")
    print(" This indicates that the remaining features are not required")
    print(" for this dataset.")
else:
    print(" Using only two features reduced the model accuracy.")
    print(" Therefore, additional features contribute to prediction.")

################################################################################
# Step 15 : Compare Different Tree Depths
#
# Objective:
# Evaluate the effect of different max_depth values on the
# performance of the Decision Tree classifier.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 15 : MAX_DEPTH COMPARISON")
print(Section_Separator)

depth_values = [1, 2, 3, 5, None]

print(f"{'Max Depth':<15}{'Training Accuracy':<25}{'Testing Accuracy'}")
print(Subsection_Separator)

for depth in depth_values:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, Y_train)

    train_accuracy = model.score(X_train, Y_train)
    test_accuracy = model.score(X_test, Y_test)

    print(f"{str(depth):<15}{train_accuracy:.2%}{'':<10}{test_accuracy:.2%}")

print("\nObservation")
print(Subsection_Separator)

print(" The dataset is easily separable using the Attendance feature.")
print(" Increasing the tree depth does not improve model accuracy.")
print(" A shallow Decision Tree is sufficient for this dataset.")

################################################################################
# Step 16 : Compare Different Random States
#
# Objective:
# Analyze the effect of different random_state values on the
# performance of the Decision Tree classifier.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 16 : RANDOM STATE COMPARISON")
print(Section_Separator)

random_states = [0, 10, 42, 50, 100]

print(f"{'Random State':<20}{'Testing Accuracy'}")
print(Subsection_Separator)

for state in random_states:

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=state
    )

    state_model = DecisionTreeClassifier(random_state=42)

    state_model.fit(X_train, Y_train)

    accuracy = state_model.score(X_test, Y_test)

    print(f"{state:<20}{accuracy:.2%}")

print("\nObservation")
print(Subsection_Separator)

print("- Changing the random_state changes the train-test split.")
print("- Different data splits resulted in different testing accuracies.")
print("- The model achieved 100% accuracy for some random states")
print("  and 83.33% for others.")
print("- This indicates that model performance is influenced by")
print("  the distribution of training and testing samples.")

################################################################################
# Step 17 : Predict New Student Records
#
# Objective:
# Use the trained Decision Tree model to predict the final result
# of new students based on their academic information.
################################################################################

print("\n")
print(Section_Separator)
print("STEP 17 : PREDICT NEW STUDENT RECORDS")
print(Section_Separator)

new_students = pd.DataFrame({

    "StudyHours":[8,3,6,4,9],

    "Attendance":[95,60,85,70,98],

    "PreviousScore":[90,55,75,60,96],

    "AssignmentsCompleted":[10,4,8,5,10],

    "SleepHours":[7,6,8,5,7]

})

#predict new record
predictions = decision_tree_model.predict(new_students)

new_students["Predicted Result"] = predictions

new_students["Predicted Result"] = new_students["Predicted Result"].map(
    {
        0:"Fail",
        1:"Pass"
    }
)

print(new_students)

print("\nObservation")
print(Subsection_Separator)

print("- The model successfully predicted the academic performance of")
print("  all new student records.")
print("- Attendance played the most significant role in determining")
print("  whether a student would Pass or Fail.")
print("- Students with attendance greater than approximately 75%")
print("  were predicted as 'Pass', while those below the threshold")
print("  were predicted as 'Fail'.")

################################################################################
# Project Conclusion
################################################################################

print("\n")
print(Section_Separator)
print("PROJECT CONCLUSION")
print(Section_Separator)

print("""
The Student Performance Prediction project successfully demonstrated
the complete Machine Learning workflow using a Decision Tree Classifier.

Key Highlights:
- Loaded and explored the student performance dataset.
- Performed Exploratory Data Analysis (EDA) and statistical analysis.
- Visualized important dataset characteristics using different plots.
- Built and trained a Decision Tree classification model.
- Evaluated the model using Accuracy Score, Confusion Matrix,
  and Classification Report.
- Analyzed feature importance to understand which attributes
  influenced student performance.
- Conducted experiments by removing features, using fewer features,
  changing max_depth, and modifying random_state values.
- Successfully predicted the performance of new student records.

""")
print(f"""
Overall Result:
The Decision Tree model achieved a testing accuracy of
{testing_accuracy:.2%}. Attendance was identified as the
most influential feature for predicting the student's
final result.
""")


