import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import shap
import plotly.express as px
import plotly.graph_objects as go
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
    """Simple Relative Weight Analysis implementation via correlation transformation."""
    corr_matrix = X.corr()
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    diagonal_sqrt_evals = np.diag(np.sqrt(eigenvalues))
    delta = eigenvectors @ diagonal_sqrt_evals @ eigenvectors.T
    transformed_X = np.linalg.inv(delta) @ X.T
    model = sm.OLS(y, sm.add_constant(transformed_X.T)).fit()
    # Use .iloc[1:] to skip the constant correctly
    raw_weights = (delta**2) @ (model.params.iloc[1:].values**2)
    rescaled_weights = (raw_weights / raw_weights.sum()) * 100
    return pd.DataFrame({'Driver': X.columns, 'Relative Weight (%)': rescaled_weights})

# --- UI APP ---
st.title("📊 Consumer Driver Analysis Suite")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file:
    xl = pd.ExcelFile(uploaded_file)
    sheet_names = xl.sheet_names
    selected_sheet = st.selectbox("Select the Sheet to Analyze", sheet_names)
    
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    # Clean names for SEM compatibility
    df.columns = [str(c).replace(' ', '_').replace('.', '_') for c in df.columns]
    
    st.write(f"### Data Preview", df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Select Target Variable", df.columns)
    with col2:
        features = st.multiselect("Select Driver Variables", [c for c in df.columns if c != target])
    
    if target and features:
        data = df[[target] + features].dropna()
        y = data[target]
        X = data[features]

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Linear Reg & RWA", "Shapley Values", "Penalty Analysis (CATA)", "Path Analysis", "Export"])

        # --- TAB 1: REGRESSION & RWA ---
        with tab1:
            st.subheader("Linear Regression & RWA")
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()
            
            rwa_results = run_rwa(X, y)
            st.plotly_chart(px.bar(rwa_results, x='Driver', y='Relative Weight (%)'))
            st.write(model.summary())

        # --- TAB 2: SHAPLEY VALUES ---
        with tab2:
            st.subheader("Shapley Value Regression")
            # FIXED: Using .iloc to access values by position to avoid KeyError
            coefs = model.params.iloc[1:].values 
            intercept = model.params.iloc[0]
            
            try:
                # We use a LinearExplainer with a custom model tuple
                explainer = shap.LinearExplainer((coefs, intercept), X)
                shap_values = explainer.shap_values(X)
                
                vals = np.abs(shap_values).mean(0)
                shap_df = pd.DataFrame({'Driver': features, 'Mean |Shapley Value|': vals}).sort_values(by='Mean |Shapley Value|', ascending=False)
                
                st.plotly_chart(px.bar(shap_df, x='Mean |Shapley Value|', y='Driver', orientation='h'))
            except Exception as e:
                st.error(f"SHAP calculation failed: {e}")
                shap_df = pd.DataFrame()

        # --- TAB 3: PENALTY ANALYSIS ---
        with tab3:
            st.subheader("Penalty Analysis for CATA")
            cata_format = st.radio("CATA Data Format", ["0 (No) / 1 (Yes)", "1 (No) / 2 (Yes)"])
            
            X_cata = X.copy()
            if cata_format == "1 (No) / 2 (Yes)":
                X_cata = X_cata - 1
            
            penalty_results = []
            for col in features:
                # Basic check to see if column is binary
                unique_vals = X_cata[col].unique()
                if 1 in unique_vals and 0 in unique_vals:
                    group_yes = y[X_cata[col] == 1]
                    group_no = y[X_cata[col] == 0]
                    mean_impact = group_yes.mean() - group_no.mean()
                    pct_checked = (X_cata[col].sum() / len(X_cata)) * 100
                    penalty_results.append({'Attribute': col, 'Mean Impact': mean_impact, '% Checked': pct_checked})
            
            penalty_df = pd.DataFrame(penalty_results)
            if not penalty_df.empty:
                fig_pen = px.scatter(penalty_df, x='% Checked', y='Mean Impact', text='Attribute')
                fig_pen.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig_pen)
            else:
                st.warning("No valid 0/1 data found in drivers for Penalty Analysis.")

        # --- TAB 4: PATH ANALYSIS ---
        with tab4:
            st.subheader("Path Analysis (SEM)")
            st.info("Example: Outcome ~ Driver1 + Driver2")
            path_syntax = st.text_area("semopy Syntax", value=f"{target} ~ {' + '.join(features[:2])}")
            
            if st.button("Run SEM"):
                try:
                    sem = Model(path_syntax)
                    sem.fit(data)
                    st.write(sem.inspect())
                except Exception as e:
                    st.error(f"SEM Syntax Error: {e}")

        # --- TAB 5: EXPORT ---
        with tab5:
            st.subheader("Export to Excel")
            # Build regression table for export
            reg_export = pd.DataFrame({
                "Coefficient": model.params,
                "P-Value": model.pvalues
            })
            
            results_dict = {
                "Regression": reg_export,
                "RWA": rwa_results,
                "Shapley": shap_df if 'shap_df' in locals() else pd.DataFrame(),
                "Penalty": penalty_df if 'penalty_df' in locals() else pd.DataFrame()
            }
            
            excel_bin = to_excel(results_dict)
            st.download_button("📥 Download Results", excel_bin, f"analysis_{selected_sheet}.xlsx")
