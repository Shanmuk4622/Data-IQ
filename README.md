# DataIQ 📊

### AI-Powered Automated EDA, Preprocessing & Dataset Intelligence Assistant

**DataIQ** is a production-grade, zero-configuration Streamlit web application that provides comprehensive, interactive, and AI-driven exploratory data analysis (EDA), data quality assessment, and machine learning preprocessing pipelines.

It combines the automated profiling of tools like `ydata-profiling` and `Sweetviz` with the intelligence of LLMs (powered by Groq) to deliver rich, actionable insights, interactive visualizations, and exportable, production-ready scikit-learn code.

---

## 🚀 Key Features

*   **Zero-Configuration Data Loading:** Supports CSV, Excel (`.xlsx`, `.xls`), JSON, Parquet, and TSV files up to 200MB.
*   **Comprehensive Data Profiling:** Automatically identifies 7 distinct data types, detects outliers (using IQR, Z-Score, and Isolation Forest), analyzes missing values, and checks for data consistency issues.
*   **Interactive Visualizations:** Over 14 interactive Plotly charts, including distribution plots, box/violin plots, Q-Q plots, correlation heatmaps, and pair-wise relationship grids.
*   **Advanced Analytics:** Statistical tests (ANOVA, Chi-Square), time-series decomposition (for datetime columns), and NLP-based text analysis (including word clouds and n-grams).
*   **Dataset Health Score:** A 5-dimension scoring algorithm (0-100) that rates your dataset and assigns a letter grade (A-F), highlighting the worst performing dimensions.
*   **Smart Preprocessing & Feature Engineering:** Automatically recommends cleaning steps, generates robust `scikit-learn` ColumnTransformer pipelines, and suggests advanced engineered features.
*   **Machine Learning Advisor:** Recommends optimal models, generates production-grade Python starter code, and provides cross-validation templates.
*   **Groq AI Explanations & Q&A:** A dedicated AI analyst utilizing the latest Groq models (with robust multi-model fallback) to generate professional, data-driven reports and answer custom queries.
*   **Download Center:** Export full HTML/Markdown/JSON reports, download generated Jupyter Notebooks (`.ipynb`), export cleaned datasets, and compare two datasets.

---

## 🛠️ Technology Stack

*   **Frontend & UI:** [Streamlit](https://streamlit.io/) (featuring a clean, minimalist Slate/Blue professional theme)
*   **Data Processing:** [Pandas](https://pandas.pydata.org/) (fully compatible with Pandas 3.x), [NumPy](https://numpy.org/)
*   **Plotting & Charts:** [Plotly](https://plotly.com/), [Wordcloud](https://github.com/amueller/word_cloud), [Missingno](https://github.com/ResidentMario/missingno)
*   **Machine Learning:** [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), [Imbalanced-learn](https://imbalanced-learn.org/)
*   **AI Engine:** [Groq API Python SDK](https://github.com/groq/groq-python) (featuring LLaMA 3.3 70B, LLaMA 3.1 8B, Qwen 32B, and more)
*   **Testing:** [Pytest](https://docs.pytest.org/)

---

## 📦 Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd DataIQ
    ```

2.  **Activate Environment:**
    Ensure you use the recommended Conda environment:
    ```powershell
    conda activate cv_conda
    ```

3.  **Install Dependencies:**
    Install required libraries (if not already present in the environment):
    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate Sample Datasets (Optional):**
    If the sample datasets are missing, run the generator script:
    ```bash
    python assets/generate_samples.py
    ```

---

## 🚀 Running the App

To launch the Streamlit application:
```powershell
# Activate the conda environment
conda activate cv_conda

# Run the app
streamlit run app.py
```
Open your browser and navigate to the local URL (typically `http://localhost:8501`).

---

## 🗂️ Project Structure

The project follows a highly modular structure to keep pages clean and focused:

```text
DataIQ/
├── app.py                          # Main entry point & sidebar config
├── requirements.txt                # Dependency definitions
├── assets/
│   ├── style.css                   # Custom CSS styling (clean, minimalist Slate/Blue theme)
│   └── sample_datasets/            # Pre-generated CSV files for testing/demo
├── modules/                        # Backend computation and logic
│   ├── loader.py                   # File parsing and validation
│   ├── profiler.py                 # Core column profiling
│   ├── quality.py                  # Missing values, duplicates, consistency
│   ├── statistics.py               # Statistical summaries
│   ├── outliers.py                 # Outlier detection
│   ├── categorical.py              # Class distribution and imbalance
│   ├── correlations.py             # Correlation matrix calculation
│   ├── relationships.py            # Cross-feature statistical tests (ANOVA/Chi2)
│   ├── datetime_analysis.py        # Date/Time extraction and analysis
│   ├── text_analysis.py            # Text/NLP-based processing
│   ├── target_detector.py          # Auto-target selection heuristics
│   ├── ml_advisor.py               # ML recommendations and code generation
│   ├── preprocessor.py             # Pipeline builder & code generation
│   ├── feature_engineer.py         # Feature engineering recommendations
│   ├── health_score.py             # Dataset scoring system
│   ├── groq_client.py              # API client and model fallback logic
│   ├── reporter.py                 # Exports (JSON, MD, HTML)
│   ├── notebook_generator.py       # Jupyter Notebook (.ipynb) exporter
│   └── dataset_comparator.py       # Statistical comparison between two datasets
├── pages/                          # Streamlit page scripts (01 to 09)
├── utils/                          # Common UI helper functions and formats
│   ├── type_detector.py            # Strategic datatype detection
│   ├── chart_factory.py            # Reusable Plotly charts
│   └── formatters.py               # Text/number formatters
└── tests/                          # Pytest suite
    ├── test_loader.py              # Loader test cases
    ├── test_quality.py             # Quality module test cases
    └── test_health_score.py        # Health scoring test cases
```

---

## 🧪 Testing

DataIQ comes with a robust test suite covering core data loading, quality checks, and health scoring.

Run all tests using `pytest` inside the `cv_conda` environment:
```powershell
python -m pytest tests/ -v
```

---

## 🧠 AI Integration

To use the AI-powered explanation and Q&A features, you need a **Groq API Key** (available for free at [console.groq.com](https://console.groq.com)). Enter this key in the sidebar of the application. 

The application uses a multi-model fallback chain to ensure maximum reliability:
*   **For Reports:** `llama-3.3-70b-versatile` ➡️ `llama-3.1-8b-instant` ➡️ `qwen/qwen3-32b` ➡️ `groq/compound`
*   **For Q&A:** `llama-3.1-8b-instant` ➡️ `groq/compound-mini` ➡️ `llama-3.3-70b-versatile`
