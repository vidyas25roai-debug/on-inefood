import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Online Food Ordering Dashboard",
    page_icon="🍔",
    layout="wide"
)

# -------------------------------------------------
# CSS
# -------------------------------------------------

st.markdown("""
<style>

.main{
background:#f5f7fb;
}

.metric{
padding:15px;
border-radius:10px;
background:white;
box-shadow:0px 0px 10px lightgray;
}

</style>
""",unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("🍔 Online Food Ordering Analytics Dashboard")

st.write("Professional Streamlit Dashboard")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():
    df=pd.read_csv("onlinefoods.csv")
    return df

df=load_data()

# -------------------------------------------------
# DATASET
# -------------------------------------------------

st.subheader("Dataset")

st.dataframe(df)

# -------------------------------------------------
# KPIs
# -------------------------------------------------

col1,col2,col3,col4=st.columns(4)

col1.metric("Total Customers",len(df))

col2.metric("Columns",len(df.columns))

col3.metric("Average Age",round(df["Age"].mean(),2))

col4.metric("Average Family Size",round(df["Family size"].mean(),2))

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("Filters")

gender=st.sidebar.multiselect(
"Gender",
df["Gender"].unique(),
default=df["Gender"].unique()
)

filtered=df[df["Gender"].isin(gender)]

# -------------------------------------------------
# AGE DISTRIBUTION
# -------------------------------------------------

fig=px.histogram(
filtered,
x="Age",
color="Gender",
title="Age Distribution"
)

st.plotly_chart(fig,use_container_width=True)

# -------------------------------------------------
# OCCUPATION
# -------------------------------------------------

fig=px.bar(
filtered["Occupation"].value_counts(),
title="Occupation"
)

st.plotly_chart(fig,use_container_width=True)

# -------------------------------------------------
# EDUCATION
# -------------------------------------------------

fig=px.pie(
filtered,
names="Educational Qualifications",
title="Education"
)

st.plotly_chart(fig,use_container_width=True)

# -------------------------------------------------
# FEEDBACK
# -------------------------------------------------

fig=px.histogram(
filtered,
x="Feedback",
color="Feedback"
)

st.plotly_chart(fig,use_container_width=True)

# -------------------------------------------------
# MARITAL STATUS
# -------------------------------------------------

fig=px.pie(
filtered,
names="Marital Status"
)

st.plotly_chart(fig,use_container_width=True)

# -------------------------------------------------
# FAMILY SIZE
# -------------------------------------------------

fig=px.box(
filtered,
y="Family size"
)

st.plotly_chart(fig,use_container_width=True)

# -------------------------------------------------
# MACHINE LEARNING
# -------------------------------------------------

st.header("Machine Learning")

data=df.copy()

encoder=LabelEncoder()

for col in data.columns:
    if data[col].dtype=="object":
        data[col]=encoder.fit_transform(data[col])

X=data.drop("Output",axis=1)

y=data["Output"]

X_train,X_test,y_train,y_test=train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

model=RandomForestClassifier()

model.fit(X_train,y_train)

pred=model.predict(X_test)

acc=accuracy_score(y_test,pred)

st.success(f"Accuracy : {acc*100:.2f}%")

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------

importance=pd.DataFrame({
"Feature":X.columns,
"Importance":model.feature_importances_
})

importance=importance.sort_values(
"Importance",
ascending=False
)

fig=px.bar(
importance,
x="Importance",
y="Feature",
orientation="h",
title="Feature Importance"
)

st.plotly_chart(fig,use_container_width=True)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

st.header("Prediction")

age=st.number_input("Age",18,60,25)

family=st.number_input("Family Size",1,10,4)

if st.button("Predict"):

    st.info("Complete prediction form can be added after encoding categorical features.")

# -------------------------------------------------
# BUSINESS INSIGHTS
# -------------------------------------------------

st.header("Insights")

st.success("""
✔ Young customers dominate online ordering.

✔ Students and employees are major users.

✔ Positive feedback increases ordering probability.

✔ Family size influences ordering behavior.

✔ Random Forest provides strong prediction accuracy.
""")

# -------------------------------------------------
# DOWNLOAD
# -------------------------------------------------

csv=filtered.to_csv(index=False)

st.download_button(
"Download Filtered Dataset",
csv,
"filtered_data.csv",
"text/csv"
)
