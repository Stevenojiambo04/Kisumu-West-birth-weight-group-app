import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

st.set_page_config(
    page_title="Kisumu West Birth Weight Analysis",
    page_icon="👶",
    layout="wide"
)

st.title("The Effect of Gravidity and Parity on Birth Weight of Infants")
st.subheader("Kisumu West Sub-County, Kenya")

st.write("""
This group capstone application explores the relationship between
gravidity, parity, and infant birth weight using maternity register data.
The analysis describes the available records and does not independently
establish causation.
""")

# Load the dataset from the GitHub repository
DATA_PATH = Path(__file__).parent / "processed_maternity_data.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df = load_data(str(DATA_PATH))
except Exception as error:
    st.error("The processed maternity dataset could not be loaded.")
    st.write("Check that processed_maternity_data.csv is in the same repository as app.py.")
    st.stop()

if df.empty:
    st.error("The dataset contains no records.")
    st.stop()

# Convert available analysis columns to numeric
numeric_columns = [
    "Gravidae", "Parity_First", "Parity_Second",
    "Babies born", "Birth_weight", "Low_Birth_Weight"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

st.success("The processed maternity dataset was loaded successfully.")

# Overview
st.header("1. Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Records", f"{len(df):,}")
c2.metric(
    "Gravidity Records",
    f"{df['Gravidae'].notna().sum():,}" if "Gravidae" in df.columns else "Unavailable"
)
c3.metric(
    "Recorded Birth Weights",
    f"{df['Birth_weight'].notna().sum():,}" if "Birth_weight" in df.columns else "Unavailable"
)

with st.expander("View dataset columns and sample records"):
    st.write("Columns:", ", ".join(df.columns.astype(str)))
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("View numerical summary"):
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

# Visual analysis
st.header("2. Exploratory Data Analysis")

if "Birth_weight" in df.columns:
    weights = df.dropna(subset=["Birth_weight"])

    fig, ax = plt.subplots()
    ax.hist(weights["Birth_weight"], bins=25, edgecolor="black")
    ax.set_title("Distribution of Infant Birth Weight")
    ax.set_xlabel("Birth weight (grams)")
    ax.set_ylabel("Number of recorded infants")
    st.pyplot(fig)
    plt.close(fig)

    if "Gravidae" in df.columns:
        plot_df = df.dropna(subset=["Gravidae", "Birth_weight"])

        fig, ax = plt.subplots()
        ax.scatter(
            plot_df["Gravidae"],
            plot_df["Birth_weight"],
            alpha=0.5
        )
        ax.set_title("Gravidity and Birth Weight")
        ax.set_xlabel("Gravidae (number of pregnancies)")
        ax.set_ylabel("Birth weight (grams)")
        st.pyplot(fig)
        plt.close(fig)

    if "Babies born" in df.columns:
        plot_df = df.dropna(subset=["Babies born", "Birth_weight"])

        fig, ax = plt.subplots()
        ax.scatter(
            plot_df["Babies born"],
            plot_df["Birth_weight"],
            alpha=0.5
        )
        ax.set_title("Babies Born and Birth Weight")
        ax.set_xlabel("Babies born")
        ax.set_ylabel("Birth weight (grams)")
        st.pyplot(fig)
        plt.close(fig)

    # Low birth weight is defined as below 2500 grams
    df["Low_Birth_Weight"] = (
        df["Birth_weight"] < 2500
    ).astype(int).where(df["Birth_weight"].notna())

    st.subheader("Birth Weight Categories")
    low_count = int((df["Low_Birth_Weight"] == 1).sum())
    normal_count = int((df["Low_Birth_Weight"] == 0).sum())
    total_classified = low_count + normal_count

    a, b, c = st.columns(3)
    a.metric("Low Birth Weight (<2500 g)", f"{low_count:,}")
    b.metric("2500 g or Above", f"{normal_count:,}")
    c.metric(
        "Low Birth Weight Percentage",
        f"{100 * low_count / total_classified:.2f}%"
        if total_classified else "Unavailable"
    )

    fig, ax = plt.subplots()
    ax.bar(
        ["Low birth weight", "2500 g or above"],
        [low_count, normal_count]
    )
    ax.set_title("Low Birth Weight Classification")
    ax.set_ylabel("Number of infants")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning(
        "The dataset does not contain a Birth_weight column. "
        "Check the uploaded CSV before proceeding with the analysis."
    )

# Decision Tree regression, following the latest notebook predictors
st.header("3. Decision Tree Regression")

regression_features = ["Gravidae", "Parity_First", "Parity_Second"]

if all(col in df.columns for col in regression_features) and "Birth_weight" in df.columns:
    reg_df = df[regression_features + ["Birth_weight"]].dropna()

    if len(reg_df) >= 10:
        X = reg_df[regression_features]
        y = reg_df["Birth_weight"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        reg_model = DecisionTreeRegressor(max_depth=4, random_state=42)
        reg_model.fit(X_train, y_train)
        y_pred = reg_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        r1, r2c, r3 = st.columns(3)
        r1.metric("MAE (grams)", f"{mae:.2f}")
        r2c.metric("RMSE (grams)", f"{rmse:.2f}")
        r3.metric("R² Score", f"{r2:.4f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.set_xlabel("Actual birth weight (grams)")
        ax.set_ylabel("Predicted birth weight (grams)")
        ax.set_title("Actual vs Predicted Birth Weight")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Estimate Birth Weight")
        st.caption(
            "Enter gravidity and the two parity values represented "
            "in the latest notebook."
        )

        with st.form("birth_weight_form"):
            gravidae = st.number_input(
                "Gravidae (number of pregnancies)",
                min_value=0, max_value=30, value=1, step=1
            )
            parity_first = st.number_input(
                "Parity first value",
                min_value=0, max_value=30, value=0, step=1
            )
            parity_second = st.number_input(
                "Parity second value",
                min_value=0, max_value=30, value=0, step=1
            )
            submitted = st.form_submit_button("Estimate birth weight")

        if submitted:
            prediction = reg_model.predict(pd.DataFrame([{
                "Gravidae": gravidae,
                "Parity_First": parity_first,
                "Parity_Second": parity_second
            }]))[0]
            st.metric("Estimated birth weight", f"{prediction:.0f} grams")
            st.caption(
                "This is a model estimate, not a clinical assessment "
                "or a guarantee of an infant's actual birth weight."
            )
    else:
        st.warning("There are not enough complete records to train the regression model.")
else:
    st.warning(
        "Regression is not available because the CSV is missing one or more "
        "required columns: Gravidae, Parity_First, Parity_Second, Birth_weight. "
        "Use the dataset exported by the latest group notebook."
    )

# Logistic regression for low birth weight
st.header("4. Logistic Regression: Low Birth Weight")

if all(col in df.columns for col in regression_features) and "Birth_weight" in df.columns:
    model_df = df[
        regression_features + ["Birth_weight"]
    ].dropna().copy()

    model_df["Low_Birth_Weight"] = (
        model_df["Birth_weight"] < 2500
    ).astype(int)

    class_counts = model_df["Low_Birth_Weight"].value_counts()

    if len(class_counts) == 2 and class_counts.min() >= 2 and len(model_df) >= 10:
        X = model_df[regression_features]
        y = model_df["Low_Birth_Weight"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        log_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        )
        log_model.fit(X_train, y_train)

        predictions = log_model.predict(X_test)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy_score(y_test, predictions):.3f}")
        m2.metric(
            "Precision",
            f"{precision_score(y_test, predictions, zero_division=0):.3f}"
        )
        m3.metric(
            "Recall",
            f"{recall_score(y_test, predictions, zero_division=0):.3f}"
        )
        m4.metric(
            "F1 Score",
            f"{f1_score(y_test, predictions, zero_division=0):.3f}"
        )

        st.write("Confusion matrix (actual rows, predicted columns):")
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual: 2500 g or above", "Actual: Low birth weight"],
                columns=["Predicted: 2500 g or above", "Predicted: Low birth weight"]
            ),
            use_container_width=True
        )

        st.subheader("Estimate Low Birth Weight Probability")

        with st.form("low_birth_weight_form"):
            g2 = st.number_input(
                "Gravidae for classification",
                min_value=0, max_value=30, value=1, step=1,
                key="log_g"
            )
            p1 = st.number_input(
                "Parity first value for classification",
                min_value=0, max_value=30, value=0, step=1,
                key="log_p1"
            )
            p2 = st.number_input(
                "Parity second value for classification",
                min_value=0, max_value=30, value=0, step=1,
                key="log_p2"
            )
            classify = st.form_submit_button("Estimate probability")

        if classify:
            sample = pd.DataFrame([{
                "Gravidae": g2,
                "Parity_First": p1,
                "Parity_Second": p2
            }])
            probability = log_model.predict_proba(sample)[0][1]
            st.metric(
                "Estimated model probability of low birth weight",
                f"{probability * 100:.1f}%"
            )
            st.caption(
                "This model output is for educational analysis and is not "
                "a medical diagnosis or clinical decision."
            )
    else:
        st.warning(
            "The logistic regression model requires enough records from both "
            "birth-weight categories. The current data does not meet that requirement."
        )
else:
    st.warning(
        "Logistic regression needs Gravidae, Parity_First, Parity_Second, "
        "and Birth_weight in the processed dataset."
    )

st.divider()
st.caption(
    "Group capstone project | TICH School of Data Science and Analytics | 2026"
)



