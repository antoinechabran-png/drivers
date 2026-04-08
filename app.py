import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import shap
import plotly.express as px
from io import BytesIO
from semopy import Model

st.set_page_config(page_title="Consumer Driver Analysis Tool", layout="wide")

# --- HELPER FUNCTIONS ---
def to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in df_dict.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
    return output.getvalue()

def run_rwa(X, y):
    """Relative Weight Analysis: Decomposition of R-squared."""
    corr_matrix = X.corr()
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    diagonal_sqrt_evals = np.diag(np.sqrt(eigenvalues))
    delta = eigenvectors @ diagonal_sqrt_evals @ eigenvectors.T
    transformed_X = np.linalg.inv(delta) @ X.T
    model = sm.OLS(y, sm.add_constant(transformed_X.T)).fit()
    raw_weights = (delta**2) @ (model.params.iloc[1:].values**2)
    rescaled_weights = (raw_weights / raw_weights.sum()) * 100
    return pd.DataFrame({'Driver': X.columns, 'Weight (%)': rescaled_weights}).sort_values(by='Weight (%)', ascending=False)

# --- UI APP ---
st.title("📊 Consumer Driver Analysis Suite")

uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

if uploaded_file:
    xl = pd.ExcelFile(uploaded_file)
    selected_sheet = st.selectbox("Select Sheet", xl.sheet_names)
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    df.columns = [str(c).replace(' ', '_').replace('.', '_') for c in df.columns]
    
    st.sidebar.header("Settings")
    target = st.sidebar.selectbox("Variable to Explain (Target)", df.columns)
    features = st.sidebar.multiselect("Explanatory Variables (Drivers)", [c for c in df.columns if c != target])
    
    analysis_types = st.sidebar.multiselect(
        "Choose Analyses to Perform", 
        ["Linear Regression", "RWA", "Shapley Values", "Penalty Analysis (CATA)", "Path Analysis"],
        default=["Linear Regression", "RWA"]
    )

    if target and features:
        data = df[[target] + features].dropna()
        y = data[target]
        X = data[features]
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        # --- HIGHLIGHTS SECTION ---
        st.info("### 💡 Significant Driver Highlights")
        p_values = model.pvalues.iloc[1:]
        significant = p_values[p_values < 0.05].index.tolist()
        if significant:
            st.write(f"Based on Linear Regression, the following variables have a **significant** ($p < 0.05$) impact on **{target}**:")
            for s in significant:
                st.markdown(f"- ✅ **{s}** (p-value: {model.pvalues[s]:.4f})")
        else:
            st.write("No variables reached the 95% significance threshold.")
        
        st.divider()

        # Display tabs based on selection
        tabs = st.tabs([a for a in analysis_types] + ["Export"])
        
        results_to_export = {}

        for i, analysis in enumerate(analysis_types):
            with tabs[i]:
                if analysis == "Linear Regression":
                    st.subheader("Linear Regression (Standardized Coefficients)")
                    # Standardize coefficients to compare scale
                    std_coefs = model.params.iloc[1:] * (X.std() / y.std())
                    reg_df = pd.DataFrame({'Driver': std_coefs.index, 'Impact Score': std_coefs.values}).sort_values(by='Impact Score', ascending=False)
                    st.plotly_chart(px.bar(reg_df, x='Impact Score', y='Driver', orientation='h', color='Impact Score'))
                    results_to_export["Regression"] = reg_df

                elif analysis == "RWA":
                    st.subheader("Relative Weight Analysis (RWA)")
                    rwa_df = run_rwa(X, y)
                    st.plotly_chart(px.pie(rwa_df, values='Weight (%)', names='Driver', title="Share of Explained Variance"))
                    st.write(rwa_df)
                    results_to_export["RWA"] = rwa_df

                elif analysis == "Shapley Values":
                    st.subheader("Shapley Value Importance")
                    try:
                        explainer = shap.LinearExplainer((model.params.iloc[1:].values, model.params.iloc[0]), X)
                        shap_values = explainer.shap_values(X)
                        shap_df = pd.DataFrame({'Driver': features, 'Importance': np.abs(shap_values).mean(0)}).sort_values(by='Importance', ascending=False)
                        st.plotly_chart(px.bar(shap_df, x='Importance', y='Driver', orientation='h', color='Importance'))
                        results_to_export["Shapley"] = shap_df
                    except Exception as e:
                        st.error(f"Shapley Error: {e}")

                elif analysis == "Penalty Analysis (CATA)":
                    st.subheader("CATA Penalty Analysis")
                    cata_format = st.radio("Data Format", ["0/1", "1/2"], key="cata_radio")
                    X_cata = X.copy() - 1 if cata_format == "1/2" else X.copy()
                    
                    pen_list = []
                    for col in features:
                        if set(X_cata[col].unique()).issubset({0, 1}):
                            diff = y[X_cata[col]==1].mean() - y[X_cata[col]==0].mean()
                            pen_list.append({'Attribute': col, 'Mean Difference': diff, '% Checked': (X_cata[col].mean()*100)})
                    
                    pen_df = pd.DataFrame(pen_list).sort_values(by='Mean Difference', ascending=False)
                    if not pen_df.empty:
                        st.plotly_chart(px.scatter(pen_df, x='% Checked', y='Mean Difference', text='Attribute', size='% Checked', color='Mean Difference'))
                        results_to_export["Penalty"] = pen_df

                elif analysis == "Path Analysis":
                    st.subheader("Structural Equation Modeling (Path)")
                    path_syntax = st.text_area("Syntax (Outcome ~ Driver)", value=f"{target} ~ {' + '.join(features[:3])}")
                    if st.button("Run Path Model"):
                        try:
                            sem = Model(path_syntax)
                            sem.fit(data)
                            res = sem.inspect().sort_values(by='Estimate', ascending=False)
                            st.write(res)
                            results_to_export["Path"] = res
                        except Exception as e:
                            st.error(f"SEM Syntax Error: {e}")

        with tabs[-1]:
            st.subheader("Download Results")
            if results_to_export:
                xlsx_data = to_excel(results_to_export)
                st.download_button("📥 Download Analysis (.xlsx)", xlsx_data, "driver_analysis.xlsx")
            else:
                st.warning("Perform an analysis first to enable export.")
