# Heart Disease Dataset Explorer (Streamlit)

Covers all 3 assigned tasks with fully manual implementations (no pandas.read_csv/.describe()/.isna()/.dropna(), no numpy stat functions):

1. **Manual CSV loading & parsing** with delimiter auto-detection, pagination, and sorting.
2. **Manual summary statistics**: mean, median, mode, min, max, std — all hand-coded.
3. **Manual missing-value detection** + 5 imputation techniques (mean, median, mode, forward-fill, linear interpolation) with a written justification report.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

The app auto-loads the bundled `heart.csv` if you don't upload a file.

## Push to GitHub / Google Drive
This sandbox has no Google Drive or GitHub push access on your behalf — download the files below and upload them yourself (drag the folder into Drive, or `git init && git add . && git commit && git push` for GitHub).
