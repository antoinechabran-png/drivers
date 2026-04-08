import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import shap
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from semopy import Model
import re

st.set_page_config(page_title="Consumer Driver Analysis Tool", layout="wide")

# --- HELPER FUNCTIONS ---
def to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in df_dict.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
    return output.getvalue()

def sanitize_name(name):
    return re.sub(r'[^a-zA-Z0-9]', '_', str(name))

def run_rwa(X, y):
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
    df_raw = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    
    # Sanitize names immediately
    df = df_raw.copy()
    df.columns = [sanitize_name(c) for c in df.columns]
    
    # --- STEP 0: SUB-TARGET FILTERING ---
    st.sidebar.header("0. Sub-Target Filtering")
    filter_col = st.sidebar.selectbox("Select Filter Question", ["No Filter"] + list(df.columns))
    
    working_df = df.copy()
    if filter_col != "No Filter":
        unique_vals = sorted(df[filter_col].dropna().unique().tolist())
        selected_codes = st.sidebar.multiselect(f"Select Codes for {filter_col}", unique_vals)
        if selected_codes:
            working_df = df[df[filter_col].isin(selected_codes)]
            st.sidebar.success(f"Filter applied: {len(working_df)} rows remaining.")
        else:
            st.sidebar.warning("No codes selected: using full sample.")

    # --- STEP 1: VARIABLE SELECTION ---
    st.sidebar.header("1. Variable Selection")
    target = st.sidebar.selectbox("Variable to Explain (Target)", working_df.columns)
    
    st.sidebar.write("Select Explanatory Variables (Drivers):")
    available_drivers = [c for c in working_df.columns if c != target]
    selection_df = pd.DataFrame({"Select": [False] * len(available_drivers), "Driver_Variable": available_drivers})
    
    edited_df = st.sidebar.data_editor(
        selection_df, hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True), "Driver_Variable": st.column_config.TextColumn(disabled=True)},
        use_container_width=True
    )
    features = edited_df[edited_df["Select"] == True]["Driver_Variable"].tolist()
    
    # --- STEP 2: ANALYSIS SELECTION ---
    st.sidebar.header("2. Analysis Selection")
    analysis_options = ["Linear Regression", "RWA", "Shapley Values", "Penalty Analysis (CATA)", "Kano Analysis", "Path Analysis"]
    analysis_types = st.sidebar.multiselect("Choose Analyses", analysis_options, default=[], placeholder="Choose options...")

    if target and features and analysis_types:
        data = working_df[[target] + features].dropna()
        y = data[target]
        X = data[features]
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        st.info(f"### 💡 Insights for Sub-Target (N={len(data)})")
        p_values = model.pvalues.iloc[1:]
        significant = p_values[p_values < 0.05].sort_values()
        if not significant.empty:
            for var, pval in significant.items():
                st.markdown(f"- ✅ **{var}** (p-value: {pval:.4f})")
        else:
            st.write("No variables reached significance for this sub-target.")
        
        st.divider()

        tabs = st.tabs([a for a in analysis_types] + ["Export"])
        results_to_export = {}

        for i, analysis in enumerate(analysis_types):
            with tabs[i]:
                if analysis == "Linear Regression":
                    st.subheader("Linear Regression (Standardized Coefficients)")
                    std_coefs = model.params.iloc[1:] * (X.std() / y.std())
                    reg_df = pd.DataFrame({'Driver': std_coefs.index, 'Impact Score': std_coefs.values}).sort_values(by='Impact Score', ascending=False)
                    st.plotly_chart(px.bar(reg_df, x='Impact Score', y='Driver', orientation='h', color='Impact Score'))
                    results_to_export["Regression"] = reg_df

                elif analysis == "RWA":
                    st.subheader("Relative Weight Analysis (RWA)")
                    rwa_df = run_rwa(X, y)
                    st.plotly_chart(px.bar(rwa_df, x='Weight (%)', y='Driver', orientation='h'))
                    results_to_export["RWA"] = rwa_df

                elif analysis == "Penalty Analysis (CATA)":
                    st.subheader("CATA Penalty Analysis")
                    cata_format = st.radio("Data Format", ["0/1", "1/2"], key="cata_radio")
                    X_cata = X.copy() - 1 if cata_format == "1/2" else X.copy()
                    pen_list = []
                    for col in features:
                        if 0 in X_cata[col].values and 1 in X_cata[col].values:
                            diff = y[X_cata[col]==1].mean() - y[X_cata[col]==0].mean()
                            pen_list.append({'Attribute': col, 'Mean Difference': diff, '% Checked': (X_cata[col].mean()*100)})
                    pen_df = pd.DataFrame(pen_list).sort_values(by='Mean Difference', ascending=False)
                    if not pen_df.empty:
                        st.plotly_chart(px.scatter(pen_df, x='% Checked', y='Mean Difference', text='Attribute', size_max=40))
                        results_to_export["Penalty"] = pen_df

                elif analysis == "Kano Analysis":
                    st.subheader("Kano Strategic Classification")
                    # Calculate Potential for Satisfaction (Reward) and Dissatisfaction (Penalty)
                    kano_list = []
                    for col in features:
                        reward = y[X[col] >= X[col].median()].mean() - y.mean()
                        penalty = y.mean() - y[X[col] < X[col].median()].mean()
                        
                        # Classification logic
                        if reward > penalty and reward > 0.1: cat = "Delighter (Attractive)"
                        elif penalty > reward and penalty > 0.1: cat = "Must-have (Basic)"
                        elif abs(reward - penalty) < 0.1 and reward > 0.1: cat = "Linear (Performance)"
                        else: cat = "Indifferent"
                        
                        kano_list.append({'Driver': col, 'Reward Potential': reward, 'Penalty Potential': penalty, 'Category': cat})
                    
                    kano_df = pd.DataFrame(kano_list)
                    st.plotly_chart(px.scatter(kano_df, x='Penalty Potential', y='Reward Potential', color='Category', text='Driver', title="Kano Map"))
                    st.table(kano_df)
                    results_to_export["Kano"] = kano_df

                elif analysis == "Path Analysis":
                    st.subheader("Path Analysis (SEM)")
                    path_syntax = st.text_area("Syntax", value=f"{target} ~ {' + '.join(features)}")
                    if st.button("Run Path Model"):
                        try:
                            sem = Model(path_syntax)
                            sem.fit(data)
                            res = sem.inspect()
                            paths = res[res['op'] == '~']
                            labels = list(set(paths['lval'].tolist() + paths['rval'].tolist()))
                            fig = go.Figure(data=[go.Sankey(
                                node = dict(pad=15, thickness=20, label=labels, color="blue"),
                                link = dict(source=[labels.index(x) for x in paths['rval']],
                                           target=[labels.index(x) for x in paths['lval']],
                                           value=np.abs(paths['Estimate']).tolist(),
                                           label=paths['Estimate'].round(3).astype(str).tolist()))])
                            st.plotly_chart(fig, use_container_width=True)
                            results_to_export["Path"] = res
                        except Exception as e:
                            st.error(f"SEM Error: {e}")

        with tabs[-1]:
            st.subheader("Download Results")
            if results_to_export:
                xlsx_data = to_excel(results_to_export)
                st.download_button("📥 Download Analysis (.xlsx)", xlsx_data, "subtarget_analysis.xlsx")
    else:
        st.info("👈 Complete the sidebar steps to begin.")
