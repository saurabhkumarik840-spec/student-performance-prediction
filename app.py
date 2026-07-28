import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "StudentPerformanceFactors.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pkl"
)

MODEL_INFO_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_info.pkl"
)

COMPARISON_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_comparison.csv"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main App */
    .stApp {
        background-color: #0b1120;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #164e63;
    }

    /* Main Title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #22d3ee;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 17px;
        margin-bottom: 30px;
    }

    /* Metric Card */
    .metric-card {
        background: linear-gradient(
            135deg,
            #111827,
            #172554
        );

        border: 1px solid #164e63;

        border-radius: 16px;

        padding: 22px;

        text-align: center;

        min-height: 130px;

        box-shadow:
            0 0 15px rgba(34, 211, 238, 0.08);
    }

    .metric-title {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #22d3ee;
        font-size: 30px;
        font-weight: 800;
    }

    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(
            135deg,
            #0f172a,
            #172554
        );

        border: 2px solid #22d3ee;

        border-radius: 22px;

        padding: 35px;

        text-align: center;

        margin: 25px 0;

        box-shadow:
            0 0 30px rgba(34, 211, 238, 0.20);
    }

    .prediction-title {
        color: #e2e8f0;
        font-size: 22px;
        font-weight: 600;
    }

    .prediction-score {
        color: #22d3ee;
        font-size: 60px;
        font-weight: 900;
        margin: 10px 0;
    }

    .prediction-level {
        color: #f8fafc;
        font-size: 25px;
        font-weight: 700;
    }

    /* Section Header */
    .section-header {
        color: #22d3ee;
        font-size: 26px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Info Box */
    .info-box {
        background-color: #111827;
        border-left: 4px solid #22d3ee;
        padding: 18px;
        border-radius: 10px;
        margin: 15px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px;
        margin-top: 50px;
        border-top: 1px solid #1e293b;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATA_PATH):
        return None

    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD BEST MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD MODEL INFO
# ============================================================

@st.cache_data
def load_model_info():

    if not os.path.exists(MODEL_INFO_PATH):
        return None

    try:

        return joblib.load(
            MODEL_INFO_PATH
        )

    except Exception:

        return None


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

@st.cache_data
def load_model_comparison():

    if not os.path.exists(COMPARISON_PATH):
        return None

    return pd.read_csv(
        COMPARISON_PATH
    )


# ============================================================
# LOAD FILES
# ============================================================

df = load_dataset()

model = load_model()

model_info = load_model_info()

comparison_df = load_model_comparison()


# ============================================================
# REQUIRED FEATURES
# ============================================================

FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Physical_Activity",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"
]


# ============================================================
# CHECK DATASET
# ============================================================

if df is None:

    st.error(
        "Dataset not found!"
    )

    st.code(
        "data/StudentPerformanceFactors.csv"
    )

    st.stop()


# ============================================================
# CHECK MODEL
# ============================================================

if model is None:

    st.error(
        "Best model not found!"
    )

    st.code(
        "models/best_model.pkl"
    )

    st.info(
        "Please run: python train_model.py"
    )

    st.stop()


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]


if missing_features:

    st.error(
        "Required dataset columns are missing:"
    )

    st.write(
        missing_features
    )

    st.stop()


# ============================================================
# PERFORMANCE LEVEL FUNCTION
# ============================================================

def get_performance_level(
    score
):

    if score >= 90:

        return (
            "Excellent",
            "🏆"
        )

    elif score >= 80:

        return (
            "Very Good",
            "👍"
        )

    elif score >= 70:

        return (
            "Good",
            "😊"
        )

    elif score >= 60:

        return (
            "Average",
            "⚠️"
        )

    else:

        return (
            "Needs Improvement",
            "📚"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;">

        <div style="
            font-size:55px;
            margin-bottom:5px;
        ">
        🎓
        </div>

        <h2 style="
            color:#22d3ee;
            margin-bottom:0;
        ">
        Student AI
        </h2>

        <p style="
            color:#94a3b8;
        ">
        Performance Prediction
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🎯 Predict Performance",
            "📊 Model Analysis",
            "📁 Dataset Explorer"
        ]
    )

    st.divider()

    st.caption(
        "Machine Learning Project"
    )

    st.caption(
        "Python • Pandas • Scikit-learn • Streamlit"
    )


# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">'
        '🎓 Student Performance AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        AI-powered student exam score prediction
        and performance analytics
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # KEY METRICS
    # --------------------------------------------------------

    total_students = len(
        df
    )

    average_score = df[
        "Exam_Score"
    ].mean()

    highest_score = min(
        df["Exam_Score"].max(),
        100
    )

    average_attendance = df[
        "Attendance"
    ].mean()


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            👨‍🎓 Total Students
            </div>

            <div class="metric-value">
            {total_students:,}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            📊 Average Exam Score
            </div>

            <div class="metric-value">
            {average_score:.2f}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            🏆 Highest Score
            </div>

            <div class="metric-value">
            {highest_score:.0f}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            📅 Average Attendance
            </div>

            <div class="metric-value">
            {average_attendance:.1f}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        '📈 Performance Analytics'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "📊 Exam Score Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.hist(
            df["Exam_Score"],
            bins=20
        )

        ax.set_xlabel(
            "Exam Score"
        )

        ax.set_ylabel(
            "Number of Students"
        )

        ax.set_title(
            "Exam Score Distribution"
        )

        st.pyplot(
            fig,
            clear_figure=True
        )


    with col2:

        st.subheader(
            "📚 Study Hours vs Exam Score"
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.scatter(
            df["Hours_Studied"],
            df["Exam_Score"],
            alpha=0.5
        )

        ax.set_xlabel(
            "Hours Studied"
        )

        ax.set_ylabel(
            "Exam Score"
        )

        ax.set_title(
            "Study Hours vs Exam Score"
        )

        st.pyplot(
            fig,
            clear_figure=True
        )


    st.subheader(
        "📅 Attendance vs Exam Score"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.scatter(
        df["Attendance"],
        df["Exam_Score"],
        alpha=0.4
    )

    ax.set_xlabel(
        "Attendance (%)"
    )

    ax.set_ylabel(
        "Exam Score"
    )

    ax.set_title(
        "Attendance vs Exam Score"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )


# ============================================================
# PAGE 2: PREDICT PERFORMANCE
# ============================================================

elif page == "🎯 Predict Performance":

    st.markdown(
        '<div class="main-title">'
        '🎯 Predict Student Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Enter all student information to predict
        the expected exam score.
        </div>
        """,
        unsafe_allow_html=True
    )


    with st.form(
        "prediction_form"
    ):

        # ====================================================
        # NUMERICAL FEATURES
        # ====================================================

        st.markdown(
            '<div class="section-header">'
            '📚 Academic & Personal Information'
            '</div>',
            unsafe_allow_html=True
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            hours_studied = st.number_input(
                "📚 Hours Studied",
                min_value=0,
                max_value=24,
                value=5
            )


        with col2:

            attendance = st.number_input(
                "📅 Attendance (%)",
                min_value=0,
                max_value=100,
                value=80
            )


        with col3:

            previous_scores = st.number_input(
                "📝 Previous Scores",
                min_value=0,
                max_value=100,
                value=70
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            sleep_hours = st.number_input(
                "😴 Sleep Hours",
                min_value=0,
                max_value=24,
                value=7
            )


        with col2:

            tutoring_sessions = st.number_input(
                "👨‍🏫 Tutoring Sessions",
                min_value=0,
                max_value=20,
                value=2
            )


        with col3:

            physical_activity = st.number_input(
                "🏃 Physical Activity",
                min_value=0,
                max_value=20,
                value=3
            )


        # ====================================================
        # CATEGORICAL FEATURES
        # ====================================================

        st.markdown(
            '<div class="section-header">'
            '🏫 Student Environment'
            '</div>',
            unsafe_allow_html=True
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            parental_involvement = st.selectbox(
                "👨‍👩‍👧 Parental Involvement",
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )


        with col2:

            access_to_resources = st.selectbox(
                "📚 Access to Resources",
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )


        with col3:

            extracurricular = st.selectbox(
                "⚽ Extracurricular Activities",
                [
                    "No",
                    "Yes"
                ]
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            motivation_level = st.selectbox(
                "🔥 Motivation Level",
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )


        with col2:

            internet_access = st.selectbox(
                "🌐 Internet Access",
                [
                    "No",
                    "Yes"
                ]
            )


        with col3:

            family_income = st.selectbox(
                "💰 Family Income",
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            teacher_quality = st.selectbox(
                "👨‍🏫 Teacher Quality",
                [
                    "Low",
                    "Medium",
                    "High"
                ]
            )


        with col2:

            school_type = st.selectbox(
                "🏫 School Type",
                [
                    "Public",
                    "Private"
                ]
            )


        with col3:

            peer_influence = st.selectbox(
                "👥 Peer Influence",
                [
                    "Negative",
                    "Neutral",
                    "Positive"
                ]
            )


        col1, col2, col3 = st.columns(3)


        with col1:

            learning_disabilities = st.selectbox(
                "🧠 Learning Disabilities",
                [
                    "No",
                    "Yes"
                ]
            )


        with col2:

            parental_education = st.selectbox(
                "🎓 Parental Education Level",
                [
                    "High School",
                    "College",
                    "Postgraduate"
                ]
            )


        with col3:

            distance_from_home = st.selectbox(
                "🏠 Distance from Home",
                [
                    "Near",
                    "Moderate",
                    "Far"
                ]
            )


        gender = st.selectbox(
            "👤 Gender",
            [
                "Male",
                "Female"
            ]
        )


        # ====================================================
        # PREDICT BUTTON
        # ====================================================

        predict_button = st.form_submit_button(
            "🚀 Predict Exam Score"
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        try:

            # ------------------------------------------------
            # CREATE EXACT 19 FEATURE DATAFRAME
            # ------------------------------------------------

            input_data = pd.DataFrame(
                [
                    {
                        "Hours_Studied":
                            hours_studied,

                        "Attendance":
                            attendance,

                        "Parental_Involvement":
                            parental_involvement,

                        "Access_to_Resources":
                            access_to_resources,

                        "Extracurricular_Activities":
                            extracurricular,

                        "Sleep_Hours":
                            sleep_hours,

                        "Previous_Scores":
                            previous_scores,

                        "Motivation_Level":
                            motivation_level,

                        "Internet_Access":
                            internet_access,

                        "Tutoring_Sessions":
                            tutoring_sessions,

                        "Family_Income":
                            family_income,

                        "Teacher_Quality":
                            teacher_quality,

                        "School_Type":
                            school_type,

                        "Peer_Influence":
                            peer_influence,

                        "Physical_Activity":
                            physical_activity,

                        "Learning_Disabilities":
                            learning_disabilities,

                        "Parental_Education_Level":
                            parental_education,

                        "Distance_from_Home":
                            distance_from_home,

                        "Gender":
                            gender
                    }
                ]
            )


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            prediction = model.predict(
                input_data
            )


            predicted_score = float(
                prediction[0]
            )


            # Keep score between 0 and 100

            predicted_score = max(
                0,
                min(
                    predicted_score,
                    100
                )
            )


            # ------------------------------------------------
            # PERFORMANCE LEVEL
            # ------------------------------------------------

            performance_level, emoji = (
                get_performance_level(
                    predicted_score
                )
            )


            st.success(
                "Prediction completed successfully!"
            )


            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="prediction-card">

                <div class="prediction-title">
                🎯 Predicted Exam Score
                </div>

                <div class="prediction-score">
                {predicted_score:.2f} / 100
                </div>

                <div class="prediction-level">
                Performance Level:
                {performance_level}
                {emoji}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            st.subheader(
                "📈 Performance Progress"
            )

            st.progress(
                int(
                    predicted_score
                )
            )


            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            st.markdown(
                '<div class="section-header">'
                '📋 Prediction Input Summary'
                '</div>',
                unsafe_allow_html=True
            )


            summary_col1, summary_col2 = (
                st.columns(2)
            )


            with summary_col1:

                st.write(
                    f"📚 **Hours Studied:** "
                    f"{hours_studied}"
                )

                st.write(
                    f"📅 **Attendance:** "
                    f"{attendance}%"
                )

                st.write(
                    f"📝 **Previous Score:** "
                    f"{previous_scores}"
                )

                st.write(
                    f"😴 **Sleep Hours:** "
                    f"{sleep_hours}"
                )

                st.write(
                    f"👨‍🏫 **Tutoring Sessions:** "
                    f"{tutoring_sessions}"
                )

                st.write(
                    f"🏃 **Physical Activity:** "
                    f"{physical_activity}"
                )


            with summary_col2:

                st.write(
                    f"👨‍👩‍👧 **Parental Involvement:** "
                    f"{parental_involvement}"
                )

                st.write(
                    f"📚 **Resources:** "
                    f"{access_to_resources}"
                )

                st.write(
                    f"🔥 **Motivation:** "
                    f"{motivation_level}"
                )

                st.write(
                    f"🌐 **Internet:** "
                    f"{internet_access}"
                )

                st.write(
                    f"🏫 **School Type:** "
                    f"{school_type}"
                )

                st.write(
                    f"👤 **Gender:** "
                    f"{gender}"
                )


            # ------------------------------------------------
            # DOWNLOAD REPORT
            # ------------------------------------------------

            report = pd.DataFrame(
                {
                    "Feature": FEATURES
                    + [
                        "Predicted_Exam_Score",
                        "Performance_Level"
                    ],

                    "Value": [
                        hours_studied,
                        attendance,
                        parental_involvement,
                        access_to_resources,
                        extracurricular,
                        sleep_hours,
                        previous_scores,
                        motivation_level,
                        internet_access,
                        tutoring_sessions,
                        family_income,
                        teacher_quality,
                        school_type,
                        peer_influence,
                        physical_activity,
                        learning_disabilities,
                        parental_education,
                        distance_from_home,
                        gender,
                        round(
                            predicted_score,
                            2
                        ),
                        performance_level
                    ]
                }
            )


            csv_data = report.to_csv(
                index=False
            )


            st.download_button(
                label="📥 Download Prediction Report",
                data=csv_data,
                file_name=(
                    "student_prediction_report.csv"
                ),
                mime="text/csv"
            )


        except Exception as e:

            st.error(
                "Prediction failed!"
            )

            st.exception(
                e
            )

            st.info(
                """
                Please check that your
                best_model.pkl was trained with
                the same 19 features and
                preprocessing pipeline.
                """
            )


# ============================================================
# PAGE 3: MODEL ANALYSIS
# ============================================================

elif page == "📊 Model Analysis":

    st.markdown(
        '<div class="main-title">'
        '📊 Model Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Compare Linear Regression, Random Forest
        and Gradient Boosting models.
        </div>
        """,
        unsafe_allow_html=True
    )


    if comparison_df is not None:

        st.subheader(
            "🏆 Model Comparison"
        )


        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        best_row = comparison_df.loc[
            comparison_df[
                "R2 Score"
            ].idxmax()
        ]


        best_model_name = (
            best_row["Model"]
        )

        best_r2 = (
            best_row["R2 Score"]
        )

        best_mae = (
            best_row["MAE"]
        )

        best_rmse = (
            best_row["RMSE"]
        )


        st.markdown(
            f"""
            <div class="prediction-card">

            <div class="prediction-title">
            🏆 Best Performing Model
            </div>

            <div class="prediction-score"
            style="font-size:42px;">

            {best_model_name}

            </div>

            <div class="prediction-level">

            R² Score:
            {best_r2:.4f}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "R² Score",
                f"{best_r2:.4f}"
            )


        with col2:

            st.metric(
                "MAE",
                f"{best_mae:.4f}"
            )


        with col3:

            st.metric(
                "RMSE",
                f"{best_rmse:.4f}"
            )


        # ----------------------------------------------------
        # R2 CHART
        # ----------------------------------------------------

        st.subheader(
            "📈 R² Score Comparison"
        )


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        ax.bar(
            comparison_df["Model"],
            comparison_df["R2 Score"]
        )


        ax.set_xlabel(
            "Model"
        )

        ax.set_ylabel(
            "R² Score"
        )

        ax.set_title(
            "Model Performance Comparison"
        )


        plt.xticks(
            rotation=15
        )


        st.pyplot(
            fig,
            clear_figure=True
        )


    else:

        st.warning(
            "Model comparison file not found."
        )

        st.info(
            "Run train_model.py first."
        )


# ============================================================
# PAGE 4: DATASET EXPLORER
# ============================================================

elif page == "📁 Dataset Explorer":

    st.markdown(
        '<div class="main-title">'
        '📁 Dataset Explorer'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
        Explore the Kaggle student performance dataset.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # DATASET METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )


    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with col3:

        st.metric(
            "Missing Values",
            int(
                df.isnull()
                .sum()
                .sum()
            )
        )


    with col4:

        st.metric(
            "Duplicate Rows",
            int(
                df.duplicated()
                .sum()
            )
        )


    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.subheader(
        "👀 Dataset Preview"
    )


    rows_to_show = st.slider(
        "Number of rows to display",
        min_value=5,
        max_value=50,
        value=10
    )


    st.dataframe(
        df.head(
            rows_to_show
        ),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    st.subheader(
        "⚠️ Missing Values"
    )


    missing_df = pd.DataFrame(
        {
            "Column":
                df.columns,

            "Missing Values":
                [
                    df[column]
                    .isnull()
                    .sum()

                    for column
                    in df.columns
                ]
        }
    )


    missing_df = missing_df[
        missing_df[
            "Missing Values"
        ] > 0
    ]


    if len(
        missing_df
    ) > 0:

        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No missing values found!"
        )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    st.subheader(
        "📊 Statistical Summary"
    )


    st.dataframe(
        df.describe(
            include="all"
        ),
        use_container_width=True
    )


    # --------------------------------------------------------
    # COLUMN INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "🔤 Column Information"
    )


    column_info = pd.DataFrame(
        {
            "Column":
                df.columns,

            "Data Type":
                [
                    str(
                        df[column]
                        .dtype
                    )

                    for column
                    in df.columns
                ],

            "Missing Values":
                [
                    df[column]
                    .isnull()
                    .sum()

                    for column
                    in df.columns
                ]
        }
    )


    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🎓 <b>Student Performance Prediction AI</b>

    <br><br>

    Machine Learning Project using

    <br>

    Python • Pandas • NumPy • Scikit-learn • Streamlit

    <br><br>

    © 2026 Student AI

    </div>
    """,
    unsafe_allow_html=True
)