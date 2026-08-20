"""
Heart Disease Prediction Using Machine Learning
Streamlit App — Tasks 1, 2, 3

Task 1: Load dataset WITHOUT built-in Pandas functions (manual file parsing,
        manual delimiter detection) + display with pagination and sorting.
Task 2: Summary statistics (mean, median, mode, min, max, std) computed
        MANUALLY (no numpy.mean/median/std, no pandas .describe()).
Task 3: Missing value detection WITHOUT .isna()/.dropna(), multiple manual
        imputation techniques, with a justification report.

Note: Streamlit itself is used only for the UI (widgets, layout, tables).
All data loading, parsing, and statistics are implemented from scratch in
pure Python, as required by the task description.
"""

import streamlit as st
import io
import math

st.set_page_config(page_title="Heart Disease Dataset Explorer", layout="wide")

# ============================================================================
# TASK 1: MANUAL FILE LOADING & PARSING (no pandas.read_csv, no pd functions)
# ============================================================================

def detect_delimiter(sample_lines):
    """Manually guess the delimiter by counting candidate characters
    across the first few lines and picking the most consistent one."""
    candidates = [",", ";", "\t", "|"]
    best_delim, best_score = ",", -1
    for d in candidates:
        counts = [line.count(d) for line in sample_lines if line.strip() != ""]
        if not counts:
            continue
        # consistent AND non-zero counts across lines => good delimiter
        if all(c == counts[0] for c in counts) and counts[0] > 0:
            score = counts[0]
            if score > best_score:
                best_score = score
                best_delim = d
    return best_delim


def manual_parse_line(line, delimiter):
    """Manually split a line on a delimiter, handling simple quoted fields,
    without str.split()'s built-in convenience being treated as a black box
    — we walk the string character by character ourselves."""
    fields = []
    current = []
    in_quotes = False
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == delimiter and not in_quotes:
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    fields.append("".join(current).strip())
    return fields


def try_cast(value):
    """Cast a raw string token to int/float/None manually."""
    v = value.strip()
    if v == "" or v.lower() in ("na", "nan", "null", "none", "?"):
        return None
    # manual integer check
    neg = v.startswith("-")
    body = v[1:] if neg else v
    if body.isdigit():
        return int(v)
    # manual float check
    try:
        f = float(v)
        return f
    except ValueError:
        return v  # keep as raw string (categorical/text)


def load_dataset_manually(raw_bytes):
    """Reads raw bytes, decodes, strips line endings, detects delimiter,
    parses header + rows, and casts values — all manually."""
    text = raw_bytes.decode("utf-8", errors="replace")
    # manual line splitting (handles \r\n and \n without relying on
    # pandas/csv module doing the whole job for us)
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    raw_lines = [ln for ln in raw_lines if ln.strip() != ""]

    if not raw_lines:
        return [], []

    delimiter = detect_delimiter(raw_lines[:5])
    header = manual_parse_line(raw_lines[0], delimiter)

    rows = []
    for line in raw_lines[1:]:
        parsed = manual_parse_line(line, delimiter)
        if len(parsed) != len(header):
            # pad or trim to keep table rectangular
            if len(parsed) < len(header):
                parsed += [None] * (len(header) - len(parsed))
            else:
                parsed = parsed[: len(header)]
        casted = [try_cast(v) for v in parsed]
        rows.append(casted)

    return header, rows


# ============================================================================
# TASK 2: MANUAL SUMMARY STATISTICS (no numpy/pandas stat functions)
# ============================================================================

def get_column(rows, col_idx):
    return [row[col_idx] for row in rows]


def numeric_values(col):
    """Filter out None/non-numeric, manually."""
    out = []
    for v in col:
        if isinstance(v, (int, float)) and v is not None:
            out.append(float(v))
    return out


def manual_mean(values):
    if not values:
        return None
    total = 0.0
    count = 0
    for v in values:
        total += v
        count += 1
    return total / count


def manual_median(values):
    if not values:
        return None
    vals = manual_sort(values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def manual_mode(values):
    if not values:
        return None
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    best_val, best_count = None, -1
    for v, c in freq.items():
        if c > best_count:
            best_val, best_count = v, c
    return best_val


def manual_min(values):
    if not values:
        return None
    m = values[0]
    for v in values[1:]:
        if v < m:
            m = v
    return m


def manual_max(values):
    if not values:
        return None
    m = values[0]
    for v in values[1:]:
        if v > m:
            m = v
    return m


def manual_std(values, mean_val=None):
    if not values or len(values) < 2:
        return 0.0
    if mean_val is None:
        mean_val = manual_mean(values)
    sq_diff_sum = 0.0
    for v in values:
        sq_diff_sum += (v - mean_val) ** 2
    variance = sq_diff_sum / (len(values) - 1)  # sample std
    return math.sqrt(variance)


def manual_sort(values):
    """Simple merge sort implemented manually (avoids relying on any
    'built-in statistical function' semantics; sorted() is just ordering,
    but we implement it ourselves to keep the pipeline fully manual)."""
    vals = list(values)
    if len(vals) <= 1:
        return vals
    mid = len(vals) // 2
    left = manual_sort(vals[:mid])
    right = manual_sort(vals[mid:])
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def compute_summary_stats(header, rows):
    summary = {}
    for idx, col_name in enumerate(header):
        col = get_column(rows, idx)
        nums = numeric_values(col)
        if not nums:
            continue
        mean_v = manual_mean(nums)
        summary[col_name] = {
            "count": len(nums),
            "mean": mean_v,
            "median": manual_median(nums),
            "mode": manual_mode(nums),
            "min": manual_min(nums),
            "max": manual_max(nums),
            "std": manual_std(nums, mean_v),
        }
    return summary


# ============================================================================
# TASK 3: MANUAL MISSING VALUE DETECTION + IMPUTATION
# ============================================================================

def find_missing(header, rows):
    """Detect missing values manually (None from parsing = missing),
    without .isna()/.isnull()/.dropna()."""
    missing_counts = {col: 0 for col in header}
    missing_cells = []  # (row_idx, col_idx)
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            if val is None:
                missing_counts[header[c_idx]] += 1
                missing_cells.append((r_idx, c_idx))
    total_missing = len(missing_cells)
    return missing_counts, missing_cells, total_missing


def impute_mean(rows, col_idx, mean_val):
    new_rows = [row[:] for row in rows]
    for row in new_rows:
        if row[col_idx] is None:
            row[col_idx] = round(mean_val, 3) if mean_val is not None else 0
    return new_rows


def impute_median(rows, col_idx, median_val):
    new_rows = [row[:] for row in rows]
    for row in new_rows:
        if row[col_idx] is None:
            row[col_idx] = median_val if median_val is not None else 0
    return new_rows


def impute_mode(rows, col_idx, mode_val):
    new_rows = [row[:] for row in rows]
    for row in new_rows:
        if row[col_idx] is None:
            row[col_idx] = mode_val if mode_val is not None else 0
    return new_rows


def impute_ffill(rows, col_idx):
    """Forward-fill: carry the last valid observation forward, manually."""
    new_rows = [row[:] for row in rows]
    last_valid = None
    for row in new_rows:
        if row[col_idx] is not None:
            last_valid = row[col_idx]
        elif last_valid is not None:
            row[col_idx] = last_valid
    return new_rows


def impute_linear_interpolate(rows, col_idx):
    """Linear interpolation between the nearest valid neighbors, manual."""
    new_rows = [row[:] for row in rows]
    n = len(new_rows)
    i = 0
    while i < n:
        if new_rows[i][col_idx] is None:
            # find previous valid
            prev_i = i - 1
            while prev_i >= 0 and new_rows[prev_i][col_idx] is None:
                prev_i -= 1
            # find next valid
            next_i = i
            while next_i < n and new_rows[next_i][col_idx] is None:
                next_i += 1
            if prev_i >= 0 and next_i < n:
                prev_v = new_rows[prev_i][col_idx]
                next_v = new_rows[next_i][col_idx]
                gap = next_i - prev_i
                for k in range(prev_i + 1, next_i):
                    frac = (k - prev_i) / gap
                    new_rows[k][col_idx] = prev_v + frac * (next_v - prev_v)
                i = next_i
            else:
                i += 1
        else:
            i += 1
    return new_rows


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🫀 Heart Disease Dataset Explorer")
st.caption("All parsing, statistics, missing-value detection and imputation below are implemented manually in pure Python — no pandas.read_csv, .describe(), .isna(), .dropna(), or numpy stat functions are used.")

uploaded_file = st.file_uploader("Upload heart.csv", type=["csv", "txt"])

# fallback: use bundled sample if nothing uploaded
if uploaded_file is None:
    try:
        with open("heart.csv", "rb") as f:
            raw_bytes = f.read()
        st.info("No file uploaded — using the bundled sample `heart.csv`.")
    except FileNotFoundError:
        raw_bytes = None
else:
    raw_bytes = uploaded_file.read()

if raw_bytes is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

header, rows = load_dataset_manually(raw_bytes)

if not header:
    st.error("Could not parse the file. Please check the format.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📄 Task 1: Data Viewer", "📊 Task 2: Summary Statistics", "🧩 Task 3: Missing Values & Imputation"])

# ---------------- TASK 1 ----------------
with tab1:
    st.subheader("Manually Parsed Dataset")
    st.write(f"**Rows:** {len(rows)}  |  **Columns:** {len(header)}")

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        sort_col = st.selectbox("Sort by column", options=header, key="sort_col")
    with col_b:
        sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
    with col_c:
        page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=0)

    sort_idx = header.index(sort_col)

    def sort_key(row):
        v = row[sort_idx]
        if v is None:
            return (1, 0)  # push missing values to the end
        if isinstance(v, (int, float)):
            return (0, v)
        return (0, str(v))

    sorted_rows = sorted(rows, key=sort_key, reverse=(sort_order == "Descending"))

    total_rows = len(sorted_rows)
    total_pages = max(1, math.ceil(total_rows / page_size))
    page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start = (page_num - 1) * page_size
    end = start + page_size
    page_rows = sorted_rows[start:end]

    st.table([dict(zip(header, row)) for row in page_rows])
    st.caption(f"Showing rows {start + 1}–{min(end, total_rows)} of {total_rows} | Page {page_num} of {total_pages}")

# ---------------- TASK 2 ----------------
with tab2:
    st.subheader("Manually Computed Summary Statistics")
    summary = compute_summary_stats(header, rows)

    display_rows = []
    for col_name, stats in summary.items():
        display_rows.append({
            "Column": col_name,
            "Count": stats["count"],
            "Mean": round(stats["mean"], 4) if stats["mean"] is not None else None,
            "Median": round(stats["median"], 4) if stats["median"] is not None else None,
            "Mode": stats["mode"],
            "Min": stats["min"],
            "Max": stats["max"],
            "Std Dev": round(stats["std"], 4) if stats["std"] is not None else None,
        })
    st.table(display_rows)

    with st.expander("How each statistic is computed (manual implementation)"):
        st.markdown("""
- **Mean** — running sum of values divided by count (`manual_mean`).
- **Median** — values sorted with a hand-written merge sort (`manual_sort`), then middle value(s) averaged (`manual_median`).
- **Mode** — frequency dictionary built by hand; value with highest count returned (`manual_mode`).
- **Min / Max** — single linear scan comparing each value (`manual_min`, `manual_max`).
- **Std Dev** — sample standard deviation: sum of squared deviations from the mean, divided by (n − 1), then square-rooted (`manual_std`).
        """)

# ---------------- TASK 3 ----------------
with tab3:
    st.subheader("Missing Value Detection (Manual)")
    missing_counts, missing_cells, total_missing = find_missing(header, rows)

    mc_rows = [{"Column": c, "Missing Count": n, "Missing %": round(100 * n / len(rows), 2) if rows else 0}
               for c, n in missing_counts.items()]
    st.table(mc_rows)
    st.write(f"**Total missing cells:** {total_missing} out of {len(rows) * len(header)} ({round(100*total_missing/(len(rows)*len(header)),2) if rows else 0}%)")

    st.subheader("Apply Imputation")
    cols_with_missing = [c for c, n in missing_counts.items() if n > 0]

    if not cols_with_missing:
        st.success("No missing values detected in this dataset.")
    else:
        target_col = st.selectbox("Column to impute", options=cols_with_missing)
        target_idx = header.index(target_col)
        col_vals = get_column(rows, target_idx)
        nums = numeric_values(col_vals)
        is_numeric_col = len(nums) > 0

        method_options = ["Mean", "Median", "Mode", "Forward Fill (LOCF)"]
        if is_numeric_col:
            method_options.append("Linear Interpolation")

        method = st.selectbox("Imputation technique", options=method_options)

        if st.button("Apply Imputation"):
            if method == "Mean":
                mean_v = manual_mean(nums) if is_numeric_col else None
                new_rows = impute_mean(rows, target_idx, mean_v)
            elif method == "Median":
                med_v = manual_median(nums) if is_numeric_col else None
                new_rows = impute_median(rows, target_idx, med_v)
            elif method == "Mode":
                mode_v = manual_mode(col_vals)
                new_rows = impute_mode(rows, target_idx, mode_v)
            elif method == "Forward Fill (LOCF)":
                new_rows = impute_ffill(rows, target_idx)
            elif method == "Linear Interpolation":
                new_rows = impute_linear_interpolate(rows, target_idx)

            _, _, remaining_missing = find_missing(header, new_rows)
            st.success(f"Applied **{method}** to `{target_col}`. Remaining missing cells overall: {remaining_missing}")
            st.table([dict(zip(header, r)) for r in new_rows[:15]])

    with st.expander("📋 Justification Report: Choosing an Imputation Method"):
        st.markdown("""
**Why multiple techniques are offered, and when to use each:**

| Technique | Best suited for | Why |
|---|---|---|
| **Mean imputation** | Numeric columns that are roughly symmetric / normally distributed (e.g. `trestbps`, `chol`) | Preserves the overall average of the column, but can be distorted by outliers and slightly reduces variance. |
| **Median imputation** | Numeric columns that are skewed or contain outliers (e.g. `oldpeak`, `chol` when skewed) | Robust to extreme values since it depends only on rank order, not magnitude. |
| **Mode imputation** | Categorical / discrete columns (e.g. `cp`, `thal`, `slope`, `sex`) | These columns take a small set of discrete labels, so the most frequent category is a reasonable, low-bias guess. |
| **Forward Fill (LOCF)** | Data with a meaningful row order (e.g. time-ordered or patient-sequence records) | Assumes adjacent records are similar; simple and fast, but risky if row order is arbitrary. |
| **Linear Interpolation** | Continuous numeric columns where nearby rows are expected to trend smoothly | Estimates missing values along a straight line between the nearest known points; better than mean/median when there's an underlying trend, but assumes the ordering is meaningful. |

**Recommended approach for this dataset (`heart.csv`):**
This is *tabular patient data*, not a time series, so row order carries no real meaning. That rules out Forward Fill and Linear Interpolation as principled choices here (they're included for completeness/demonstration). The recommended strategy is:
- Use **median imputation** for numeric clinical measurements (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) because clinical measurements often contain outliers, and median is more robust than mean in that case.
- Use **mode imputation** for categorical/coded columns (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`, `target`) since these represent discrete categories, not continuous quantities, and mean/median are not meaningful for them.

This hybrid approach (median for continuous, mode for categorical) minimizes distortion of each column's distribution while remaining simple, interpretable, and appropriate for the data types present.
        """)
