# Heart Disease Prediction — 25 Aug 2026 Submission

This single app (`25_august_app.py`) covers ALL tasks to date, including today's 3:

- Tab 10: Store processed data to a new CSV manually (no df.to_csv() / csv.writer)
- Tab 11: Manual histograms for all numeric features (manual bin edges + frequency counts, drawn with raw ax.bar())
- Tab 12: Manual boxplots (manual quartiles/IQR/Tukey fences, drawn with raw ax.fill_between/ax.plot/ax.scatter)

## Run
```
pip install -r requirements.txt
streamlit run 25_august_app.py
```
The app auto-loads the bundled `heart.csv` if you don't upload a file.
For Task 10, use the in-app "Download processed_heart.csv" button on Tab 10 to export the processed data.
