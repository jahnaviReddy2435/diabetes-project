# diabetes_dashboard.py

# Install streamlit if not already installed
!pip install streamlit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import io

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Data Preprocessing
st.title("Diabetes Data Insights Dashboard")
st.header("Data Preprocessing")

# Show raw data
toggle_raw = st.checkbox("Show Raw Dataset")
if toggle_raw:
    st.dataframe(df)

# Handle missing values (replace 0s with NaN in some columns)
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")
st.subheader("Descriptive Statistics")
st.write(df.describe())

# Visualizations
st.subheader("Visualizations")

# Histogram
fig1, ax1 = plt.subplots()
df['Age'].hist(bins=20, ax=ax1, color='skyblue')
ax1.set_title('Age Distribution')
st.pyplot(fig1)

# Boxplot
fig2, ax2 = plt.subplots()
sns.boxplot(x='Outcome', y='BMI', data=df, ax=ax2)
ax2.set_title('BMI vs Outcome')
st.pyplot(fig2)

# Heatmap
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax3)
ax3.set_title('Correlation Heatmap')
st.pyplot(fig3)

# Filter
st.header("Filter Data")
age_range = st.slider("Select Age Range", int(df.Age.min()), int(df.Age.max()), (20, 50))
filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
st.write(f"Filtered Data (Age between {age_range[0]} and {age_range[1]}):", filtered_df.shape[0], "records")
st.dataframe(filtered_df.head())

# Summary
st.header("Summary Metrics")
st.write("Total records:", len(df))
st.write("Positive diabetes cases:", df[df['Outcome'] == 1].shape[0])
st.write("Negative diabetes cases:", df[df['Outcome'] == 0].shape[0])

# Bonus: Simple Predictive Model
st.header("Predictive Model (Bonus)")
features = df.drop('Outcome', axis=1)
target = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text("Model Evaluation Report:")
st.text(classification_report(y_test, y_pred))

# Export CSV
st.header("Export Insights")
if st.button("Download Filtered Data as CSV"):
    csv = filtered_df.to_csv(index=False)
    st.download_button("Click to Download", csv, "filtered_data.csv", "text/csv")

# Export chart
img_buf = io.BytesIO()
fig1.savefig(img_buf, format='png')
st.download_button("Download Age Histogram", data=img_buf.getvalue(), file_name="age_histogram.png", mime="image/png")


![image](https://github.com/user-attachments/assets/35a5bde6-eafe-4f50-8d2d-962b4d18c34c)
![image](https://github.com/user-attachments/assets/a999f232-fdd8-4597-abe0-670c91938fa7)
![image](https://github.com/user-attachments/assets/3cd3408e-b58c-4ae6-9c4d-d659a4292321)

