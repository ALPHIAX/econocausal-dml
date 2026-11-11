import streamlit as st  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]

st.set_page_config(
    page_title="EconoCausal Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 EconoCausal")
st.subheader("Dynamic Pricing using Double Machine Learning")

st.write(
    "Analyze pricing, sales, and demand through data analytics "
    "and causal inference."
)

st.divider()

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    ["Overview", "Dataset Explorer", "Causal Analysis"],
)

if page == "Overview":
    st.header("Project Overview")
    st.write(
        "EconoCausal aims to understand how pricing decisions "
        "affect sales and customer demand."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Project Status", "In Progress")
    col2.metric("Analysis", "Under Development")
    col3.metric("Version", "0.1")

    st.subheader("Project Workflow")

    st.markdown("""
1. Data Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Double Machine Learning
5. Causal Effect Estimation
""")

elif page == "Dataset Explorer":
    st.header("Dataset Explorer")
    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        col1, col2 = st.columns(2)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Missing Values")

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            st.success("No missing values found.")
        else:
            st.dataframe(missing.rename("Missing Count").to_frame())
    else:
        st.info("Upload a CSV file to explore your dataset.")

elif page == "Causal Analysis":
    st.header("Causal Analysis")
    st.info(
        "Causal estimates will appear here after integration "
        "with the project's Double Machine Learning pipeline."
    )

    st.subheader("Planned Features")

    st.markdown("""
- Price and sales relationship
- Treatment and outcome variable selection
- Causal effect estimation
- Model evaluation
""")

st.divider()
st.caption("EconoCausal | Initial Dashboard Prototype")