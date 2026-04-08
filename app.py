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
            df.to_excel(writer, sheet_name=sheet_name, index=True)
    return output.getvalue()

def run_rwa(X, y):
    """Simple Relative Weight Analysis implementation via correlation transformation."""
    # Orthogonalize predictors
    corr_matrix = X.corr()
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    # Ensure no negative eigenvalues due to precision
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    diagonal_sqrt_evals = np.diag(np.sqrt(eigenvalues))
    # Transformation matrix
    delta = eigenvectors @ diagonal_sqrt_evals @ eigenvectors.T
    transformed_X = np.linalg.inv(delta) @ X.T
    # Regression on transformed
    model = sm.OLS(y, sm.add_constant(transformed_X.T)).fit()
    raw_weights = (delta**2) @ (model.params[1:]**2)
    rescaled_weights = (raw_weights / raw_weights.sum()) * 100
    return pd.DataFrame({'Driver': X.columns, 'Relative Weight (%)': rescaled_weights})

# --- UI APP ---
st.title("📊 Consumer Driver Analysis Suite")
st.markdown("Upload your Excel file, select the sheet, and define your variables.")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file:
    # Get sheet names
    xl = pd.ExcelFile(uploaded_file)
    sheet_names = xl.sheet_names
    selected_sheet = st.selectbox("Select the Sheet to Analyze", sheet_names)
    
    # Load data
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    
    # Clean column names for SEM compatibility (semopy hates spaces)
    df.columns = [c.replace(' ', '_').replace('.', '_') for c in df.columns]
    
    st.write(f"### Data Preview: {selected_sheet}", df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Select Target Variable (Variable to explain)", df.columns)
    with col2:
        features = st.multiselect("Select Driver Variables (Explanatory variables)", [c for c in df.columns if c != target])
    
    if target and features:
        # Data Cleaning
        data = df[[target] + features].dropna()
        y = data[target]
        X = data[features]

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Linear Reg & RWA", "Shapley Values", "Penalty Analysis (CATA)", "Path Analysis", "Export"])

        # --- TAB 1: REGRESSION & RWA ---
        with tab1:
            st.subheader("Linear Regression & Relative Weight Analysis")
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()
            
            # RWA Calculation
            rwa_results = run_rwa(X, y)
            fig_rwa = px.bar(rwa_results, x='Driver', y='Relative Weight (%)', 
                             title="Relative Weight Analysis (Contribution to R²)")
            st.plotly_chart(fig_rwa)
            st.write("### RWA Summary Table", rwa_results)
            st.write("### Standard Regression Summary", model.summary())

        # --- TAB 2: SHAPLEY VALUES ---
        with tab2:
            st.subheader("Shapley Value Regression")
            # FIX: Manually passing coefs and intercept to bypass shap/statsmodels incompatibility
            coefs = model.params[1:].values 
            intercept = model.params[0]
            
            try:
                explainer = shap.LinearExplainer((coefs, intercept), X)
                shap_values = explainer.shap_values(X)
                
                vals = np.abs(shap_values).mean(0)
                shap_df = pd.DataFrame(list(zip(features, vals)), 
                                     columns=['Driver','Mean |Shapley Value|']).sort_values(by='Mean |Shapley Value|', ascending=False)
                
                fig_shap = px.bar(shap_df, x='Mean |Shapley Value|', y='Driver', 
                                 orientation='h', title="Global Feature Importance (SHAP)")
                st.plotly_chart(fig_shap)
            except Exception as e:
                st.error(f"SHAP Error: {e}")

        # --- TAB 3: PENALTY ANALYSIS ---
        with tab3:
            st.subheader("Penalty Analysis for CATA (Yes/No)")
            cata_format = st.radio("CATA Data Format", ["0 (No) / 1 (Yes)", "1 (No) / 2 (Yes)"])
            
            X_cata = X.copy()
            if cata_format == "1 (No) / 2 (Yes)":
                X_cata = X_cata - 1
            
            penalty_results = []
            for col in features:
                group_yes = y[X_cata[col] == 1]
                group_no = y[X_cata[col] == 0]
                if len(group_yes) > 2 and len(group_no) > 2:
                    mean_drop = group_yes.mean() - group_no.mean()
                    pct_checked = (X_cata[col].sum() / len(X_cata)) * 100
                    penalty_results.append({'Attribute': col, 'Mean Impact': mean_drop, '% Checked': pct_checked})
            
            penalty_df = pd.DataFrame(penalty_results)
            if not penalty_df.empty:
                fig_pen = px.scatter(penalty_df, x='% Checked', y='Mean Impact', text='Attribute', 
                                   title="Penalty/Reward Map", size_max=60)
                fig_pen.update_traces(textposition='top center')
                fig_pen.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_pen)
                st.write(penalty_df)
            else:
                st.warning("Insufficient variation in CATA data to perform analysis.")

        # --- TAB 4: PATH ANALYSIS ---
        with tab4:
            st.subheader("Path Analysis (SEM)")
            st.info("Formula syntax: `Outcome ~ Driver1 + Driver2`")
            default_path = f"{target} ~ {' + '.join(features[:3])}"
            path_syntax = st.text_area("semopy Syntax", value=default_path)
            
            if st.button("Run Path Analysis"):
                try:
                    sem = Model(path_syntax)
                    sem.fit(data)
                    estimates = sem.inspect()
                    st.write(estimates)
                except Exception as e:
                    st.error(f"SEM Error: {e}")

        # --- TAB 5: EXPORT ---
        with tab5:
            st.subheader("Export Results to Excel")
            reg_summary = pd.DataFrame({"Coeff": model.params, "P-Value": model.pvalues})
            
            results_dict = {
                "Regression": reg_summary,
                "RWA": rwa_results,
                "Shapley": shap_df if 'shap_df' in locals() else pd.DataFrame(),
                "Penalty_Analysis": penalty_df
            }
            excel_data = to_excel(results_dict)
            st.download_button(label="📥 Download All Results", data=excel_data, 
                               file_name=f"{selected_sheet}_analysis.xlsx")
