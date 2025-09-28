# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ================================
# 1. Load and Preprocess Dataset
# ================================
@st.cache_data
def load_data():
    data = pd.read_csv("train.csv")

    # Drop duplicates
    data.drop_duplicates("Name", inplace=True)

    # Handle missing values
    data.drop("Cabin", axis=1, inplace=True)
    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
    data["Age"].fillna(data["Age"].mean(), inplace=True)

    # Drop irrelevant columns
    data.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)

    # Encode categorical variables
    LEC_Sex = LabelEncoder()
    LEC_Embarked = LabelEncoder()
    data["Sex"] = LEC_Sex.fit_transform(data["Sex"])
    data["Embarked"] = LEC_Embarked.fit_transform(data["Embarked"])

    return data, LEC_Sex, LEC_Embarked

data_df, LEC_Sex, LEC_Embarked = load_data()
X = data_df.drop("Survived", axis=1)
y = data_df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ================================
# 2. Train & Evaluate Models
# ================================
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    return model, acc

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = []
trained_models = {}
for name, model in models.items():
    trained, acc = evaluate_model(model, name)
    results.append({"Model": name, "Accuracy": acc})
    trained_models[name] = trained

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

# Save best model
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
joblib.dump(best_model, "best_model.pkl")

# ================================
# 3. Streamlit UI
# ================================
st.title("🚢 Titanic Survival Prediction")
st.write("This app predicts passenger survival using ML models.")

# Show data preview
with st.expander("📊 View Titanic Dataset"):
    st.dataframe(data_df.head())

# Show model performance
st.subheader("📈 Model Performance")
st.dataframe(results_df)

# Load best model
loaded_model = joblib.load("best_model.pkl")

# ================================
# 4. Passenger Prediction Form
# ================================
st.subheader("🔮 Predict Passenger Survival")

with st.form("prediction_form"):
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", 0, 100, 25)
    sibsp = st.number_input("Number of Siblings/Spouses (SibSp)", 0, 10, 0)
    parch = st.number_input("Number of Parents/Children (Parch)", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.2)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    submitted = st.form_submit_button("Predict")

    if submitted:
        sex_val = LEC_Sex.transform([sex])[0]
        embarked_val = LEC_Embarked.transform([embarked])[0]
        features = [[pclass, sex_val, age, sibsp, parch, fare, embarked_val]]

        prediction = loaded_model.predict(features)[0]
        result = "✅ Survived" if prediction == 1 else "❌ Did Not Survive"
        st.success(f"Prediction: {result}")

# ================================
# 5. Run Instructions
# ================================
st.sidebar.header("⚙ How to Run")
st.sidebar.write("1. Save this file as `app.py`")
st.sidebar.write("2. Run in terminal: `streamlit run app.py`")
st.sidebar.write("3. The app will open in your browser 🚀")
