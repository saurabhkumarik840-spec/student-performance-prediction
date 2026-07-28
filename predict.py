# ============================================================
# STUDENT PERFORMANCE PREDICTION
# STEP 3.3 - PREDICTION SYSTEM
# ============================================================

import os
import joblib
import pandas as pd


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

MODEL_PATH = "models/best_model.pkl"


# ------------------------------------------------------------
# 2. LOAD TRAINED MODEL
# ------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Trained model not found!")
    print(f"Expected location: {MODEL_PATH}")
    print("\nPlease run:")
    print("python train_model.py")
    exit()


model = joblib.load(MODEL_PATH)


# ------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------

def get_integer_input(prompt, min_value=None, max_value=None):
    """
    Get a valid integer input from user.
    """

    while True:

        try:

            value = int(input(prompt))

            if min_value is not None and value < min_value:
                print(
                    f"Please enter a value >= {min_value}."
                )
                continue

            if max_value is not None and value > max_value:
                print(
                    f"Please enter a value <= {max_value}."
                )
                continue

            return value

        except ValueError:

            print(
                "Invalid input! Please enter a number."
            )


def get_choice_input(prompt, choices):
    """
    Get a valid categorical input.
    """

    while True:

        value = input(prompt).strip()

        # Case-insensitive matching
        for choice in choices:

            if value.lower() == choice.lower():
                return choice

        print(
            f"Invalid input! Please choose from: "
            f"{', '.join(choices)}"
        )


# ------------------------------------------------------------
# 4. PERFORMANCE LEVEL
# ------------------------------------------------------------

def get_performance_level(score):

    if score >= 90:
        return "Excellent"

    elif score >= 80:
        return "Very Good"

    elif score >= 70:
        return "Good"

    elif score >= 60:
        return "Average"

    else:
        return "Needs Improvement"


# ------------------------------------------------------------
# 5. MAIN PROGRAM
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("       STUDENT PERFORMANCE PREDICTION")
print("=" * 50)

print(
    "\nPlease enter the student's information."
)

print(
    "The trained Machine Learning model will "
    "predict the expected exam score."
)


# ------------------------------------------------------------
# 6. NUMERICAL INPUTS
# ------------------------------------------------------------

print("\n" + "-" * 50)
print("ACADEMIC INFORMATION")
print("-" * 50)


hours_studied = get_integer_input(
    "Hours studied: ",
    min_value=0,
    max_value=24
)


attendance = get_integer_input(
    "Attendance percentage: ",
    min_value=0,
    max_value=100
)


previous_scores = get_integer_input(
    "Previous exam score: ",
    min_value=0,
    max_value=100
)


tutoring_sessions = get_integer_input(
    "Number of tutoring sessions: ",
    min_value=0,
    max_value=20
)


# ------------------------------------------------------------
# 7. PERSONAL & LEARNING INFORMATION
# ------------------------------------------------------------

print("\n" + "-" * 50)
print("PERSONAL & LEARNING INFORMATION")
print("-" * 50)


sleep_hours = get_integer_input(
    "Sleep hours per day: ",
    min_value=0,
    max_value=24
)


physical_activity = get_integer_input(
    "Physical activity hours per week: ",
    min_value=0,
    max_value=24
)


# ------------------------------------------------------------
# 8. CATEGORICAL INPUTS
# ------------------------------------------------------------

print("\n" + "-" * 50)
print("STUDENT BACKGROUND")
print("-" * 50)


parental_involvement = get_choice_input(
    "Parental involvement (Low/Medium/High): ",
    ["Low", "Medium", "High"]
)


access_to_resources = get_choice_input(
    "Access to resources (Low/Medium/High): ",
    ["Low", "Medium", "High"]
)


extracurricular_activities = get_choice_input(
    "Extracurricular activities (Yes/No): ",
    ["Yes", "No"]
)


motivation_level = get_choice_input(
    "Motivation level (Low/Medium/High): ",
    ["Low", "Medium", "High"]
)


internet_access = get_choice_input(
    "Internet access (Yes/No): ",
    ["Yes", "No"]
)


family_income = get_choice_input(
    "Family income (Low/Medium/High): ",
    ["Low", "Medium", "High"]
)


teacher_quality = get_choice_input(
    "Teacher quality (Low/Medium/High): ",
    ["Low", "Medium", "High"]
)


school_type = get_choice_input(
    "School type (Public/Private): ",
    ["Public", "Private"]
)


peer_influence = get_choice_input(
    "Peer influence (Positive/Neutral/Negative): ",
    ["Positive", "Neutral", "Negative"]
)


learning_disabilities = get_choice_input(
    "Learning disabilities (Yes/No): ",
    ["Yes", "No"]
)


parental_education_level = get_choice_input(
    "Parental education (High School/College/Postgraduate): ",
    [
        "High School",
        "College",
        "Postgraduate"
    ]
)


distance_from_home = get_choice_input(
    "Distance from home (Near/Moderate/Far): ",
    [
        "Near",
        "Moderate",
        "Far"
    ]
)


gender = get_choice_input(
    "Gender (Male/Female): ",
    [
        "Male",
        "Female"
    ]
)


# ------------------------------------------------------------
# 9. CREATE INPUT DATAFRAME
# ------------------------------------------------------------

student_data = pd.DataFrame(
    [
        {
            "Hours_Studied": hours_studied,
            "Attendance": attendance,
            "Parental_Involvement": parental_involvement,
            "Access_to_Resources": access_to_resources,
            "Extracurricular_Activities": extracurricular_activities,
            "Sleep_Hours": sleep_hours,
            "Previous_Scores": previous_scores,
            "Motivation_Level": motivation_level,
            "Internet_Access": internet_access,
            "Tutoring_Sessions": tutoring_sessions,
            "Family_Income": family_income,
            "Teacher_Quality": teacher_quality,
            "School_Type": school_type,
            "Peer_Influence": peer_influence,
            "Physical_Activity": physical_activity,
            "Learning_Disabilities": learning_disabilities,
            "Parental_Education_Level": parental_education_level,
            "Distance_from_Home": distance_from_home,
            "Gender": gender
        }
    ]
)


# ------------------------------------------------------------
# 10. MAKE PREDICTION
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("PREDICTING EXAM SCORE...")
print("=" * 50)


try:

    prediction = model.predict(
        student_data
    )

    predicted_score = float(
        prediction[0]
    )

    # Keep score between 0 and 100
    predicted_score = max(
        0,
        min(
            100,
            predicted_score
        )
    )

    performance_level = get_performance_level(
        predicted_score
    )


    # --------------------------------------------------------
    # 11. DISPLAY RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 50)
    print("              PREDICTION RESULT")
    print("=" * 50)

    print(
        f"\nPredicted Exam Score : "
        f"{predicted_score:.2f}"
    )

    print(
        f"Performance Level    : "
        f"{performance_level}"
    )

    print(
        "\n" + "=" * 50
    )


except Exception as e:

    print(
        "\nERROR: Prediction failed!"
    )

    print(
        f"Details: {e}"
    )