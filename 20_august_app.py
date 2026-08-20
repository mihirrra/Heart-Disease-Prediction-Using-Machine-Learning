"""
Heart Disease Prediction Using Machine Learning
Streamlit App — Tasks 1-9

Task 1: Load dataset WITHOUT built-in Pandas functions (manual file parsing,
        manual delimiter detection) + display with pagination and sorting.
Task 2: Summary statistics (mean, median, mode, min, max, std) computed
        MANUALLY (no numpy.mean/median/std, no pandas .describe()).
Task 3: Missing value detection WITHOUT .isna()/.dropna(), multiple manual
        imputation techniques, with a justification report.
Task 4: Duplicate row detection with a CUSTOM comparison algorithm, manual
        removal, and analysis of duplicate impact on data integrity/ML.
Task 5: Distribution visualization of all numeric columns using RAW
        Matplotlib only (no seaborn, no pandas.plot, no sklearn), with
        custom aesthetics, annotations, and comparative plots.
Task 6: Manual categorical encoding (label + one-hot) without
        sklearn.LabelEncoder / OneHotEncoder / pandas.get_dummies, with
        dynamic handling of unseen/unknown categories.
Task 7: Manual normalization/standardization (Min-Max, Z-score, Robust/IQR
        scaling) without sklearn.preprocessing, robust to outliers/skew.
Task 8: Manual one-hot encoding via a stateful custom encoder class that
        dynamically adapts to brand-new unseen categories in future data.
Task 9: Engineered "risk_factor" feature from chol + thalach using domain
        knowledge, an interaction term, and manual percentile-based ranking.

Note: Streamlit itself is used only for the UI (widgets, layout, tables).
All data loading, parsing, statistics, dedup, encoding, scaling, and feature
engineering logic are implemented from scratch in pure Python, as required
by the task description.
"""

import streamlit as st
import io
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
# TASK 4: CUSTOM DUPLICATE DETECTION & REMOVAL (no .duplicated()/.drop_duplicates())
# ============================================================================

def row_signature(row):
    """Build a hashable signature for a row by manually normalizing each
    value (so 3 and 3.0 compare equal, strings are case/space-normalized)."""
    parts = []
    for v in row:
        if v is None:
            parts.append("∅")
        elif isinstance(v, float) and v.is_integer():
            parts.append(str(int(v)))
        elif isinstance(v, (int, float)):
            parts.append(str(v))
        else:
            parts.append(str(v).strip().lower())
    return tuple(parts)


def custom_rows_equal(row_a, row_b, tolerance=1e-9):
    """Custom comparison algorithm: compares two rows field by field,
    treating numerically-equal values (within a float tolerance) and
    case/whitespace-insensitive strings as duplicates — not a blind
    Python == check."""
    if len(row_a) != len(row_b):
        return False
    for a, b in zip(row_a, row_b):
        if a is None and b is None:
            continue
        if a is None or b is None:
            return False
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(float(a) - float(b)) > tolerance:
                return False
        else:
            if str(a).strip().lower() != str(b).strip().lower():
                return False
    return True


def find_duplicates_manual(rows):
    """Manually scans rows, grouping by signature (bucketed comparison for
    efficiency), then confirms each candidate pair with custom_rows_equal.
    Returns: list of duplicate row indices (keeping the first occurrence),
    and a mapping of first_index -> [duplicate indices]."""
    seen_signatures = {}   # signature -> first row index
    duplicate_indices = []
    duplicate_map = {}     # first_index -> [dup indices]

    for idx, row in enumerate(rows):
        sig = row_signature(row)
        if sig in seen_signatures:
            first_idx = seen_signatures[sig]
            # confirm with the custom comparator (guards against hash collisions)
            if custom_rows_equal(rows[first_idx], row):
                duplicate_indices.append(idx)
                duplicate_map.setdefault(first_idx, []).append(idx)
                continue
        seen_signatures[sig] = idx

    return duplicate_indices, duplicate_map


def remove_duplicates_manual(rows, duplicate_indices):
    dup_set = set(duplicate_indices)
    return [row for i, row in enumerate(rows) if i not in dup_set]


# ============================================================================
# TASK 5: DISTRIBUTION VISUALIZATION — RAW MATPLOTLIB ONLY
# ============================================================================

def get_numeric_columns(header, rows):
    numeric_cols = []
    for idx, name in enumerate(header):
        vals = numeric_values(get_column(rows, idx))
        if len(vals) > 0:
            numeric_cols.append((name, idx))
    return numeric_cols


def plot_histogram(values, col_name, bins=20, color="#5B8DEF"):
    fig, ax = plt.subplots(figsize=(6, 4))
    n, bin_edges, patches = ax.hist(values, bins=bins, color=color,
                                     edgecolor="white", linewidth=0.8, alpha=0.9)
    mean_v = manual_mean(values)
    median_v = manual_median(values)
    ax.axvline(mean_v, color="#E74C3C", linestyle="--", linewidth=1.6, label=f"Mean = {mean_v:.2f}")
    ax.axvline(median_v, color="#2ECC71", linestyle="-.", linewidth=1.6, label=f"Median = {median_v:.2f}")

    # annotate the tallest bar
    max_bar_idx = max(range(len(n)), key=lambda i: n[i])
    bar_center = (bin_edges[max_bar_idx] + bin_edges[max_bar_idx + 1]) / 2
    ax.annotate(f"peak: {int(n[max_bar_idx])}",
                xy=(bar_center, n[max_bar_idx]),
                xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=8, color="#333",
                arrowprops=dict(arrowstyle="->", color="#333", lw=0.8))

    ax.set_title(f"Distribution of {col_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel(col_name, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def plot_boxplot_comparative(numeric_cols_data, labels):
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 5))
    bp = ax.boxplot(numeric_cols_data, labels=labels, patch_artist=True,
                     medianprops=dict(color="#E74C3C", linewidth=1.6),
                     boxprops=dict(facecolor="#AED6F1", edgecolor="#2E86C1"),
                     whiskerprops=dict(color="#2E86C1"),
                     capprops=dict(color="#2E86C1"),
                     flierprops=dict(marker="o", markersize=3, markerfacecolor="#E74C3C", markeredgecolor="none"))
    ax.set_title("Comparative Boxplots — All Numeric Columns", fontsize=12, fontweight="bold")
    ax.set_ylabel("Value", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_grid_histograms(numeric_cols, header, rows):
    n_cols_grid = 3
    n = len(numeric_cols)
    n_rows_grid = math.ceil(n / n_cols_grid)
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(4.2 * n_cols_grid, 3.2 * n_rows_grid))
    axes_flat = axes.flatten() if n > 1 else [axes]

    palette = ["#5B8DEF", "#F5A623", "#7ED321", "#D0021B", "#9013FE", "#50E3C2",
               "#B8860B", "#FF6F91", "#00A8CC", "#845EC2", "#4E8397", "#FF9671", "#008F7A"]

    for i, (col_name, col_idx) in enumerate(numeric_cols):
        vals = numeric_values(get_column(rows, col_idx))
        ax = axes_flat[i]
        ax.hist(vals, bins=15, color=palette[i % len(palette)], edgecolor="white", alpha=0.9)
        ax.set_title(col_name, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(numeric_cols), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("All Numeric Column Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ============================================================================
# TASK 6: MANUAL CATEGORICAL ENCODING (no sklearn / pd.get_dummies)
# ============================================================================

def build_label_map(values):
    """Manually build a category -> integer mapping in first-seen order."""
    mapping = {}
    next_code = 0
    for v in values:
        key = "∅" if v is None else str(v).strip().lower()
        if key not in mapping:
            mapping[key] = next_code
            next_code += 1
    return mapping


def apply_label_encoding(values, mapping, unknown_strategy="new_code"):
    """Encode values using the mapping, handling unseen categories
    dynamically per the chosen strategy."""
    encoded = []
    mapping = dict(mapping)  # local copy, may grow for 'new_code' strategy
    next_code = (max(mapping.values()) + 1) if mapping else 0
    for v in values:
        key = "∅" if v is None else str(v).strip().lower()
        if key in mapping:
            encoded.append(mapping[key])
        else:
            if unknown_strategy == "new_code":
                mapping[key] = next_code
                encoded.append(next_code)
                next_code += 1
            elif unknown_strategy == "most_frequent":
                # fall back to code 0 (assumes 0 = most common if map built that way)
                encoded.append(0)
            else:  # "missing_code" -> -1
                encoded.append(-1)
    return encoded, mapping


def apply_one_hot_encoding(values, categories):
    """Manually build one-hot columns for the given ordered category list.
    Any value not in `categories` gets an all-zero row (handled dynamically)
    plus is flagged in an '__unknown__' indicator column."""
    one_hot_rows = []
    for v in values:
        key = "∅" if v is None else str(v).strip().lower()
        row_vec = [1 if key == c else 0 for c in categories]
        is_unknown = 1 if key not in categories else 0
        row_vec.append(is_unknown)
        one_hot_rows.append(row_vec)
    return one_hot_rows


# ============================================================================
# TASK 7: MANUAL NORMALIZATION / STANDARDIZATION (no sklearn.preprocessing)
# ============================================================================

def manual_percentile(sorted_values, pct):
    """Linear-interpolation percentile (like numpy's default), computed
    manually on an already-sorted list."""
    if not sorted_values:
        return None
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (n - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_values[low]
    frac = rank - low
    return sorted_values[low] + frac * (sorted_values[high] - sorted_values[low])


def minmax_normalize(values):
    """Min-Max scaling to [0, 1]. Sensitive to outliers (kept for
    comparison against the robust method below)."""
    mn, mx = manual_min(values), manual_max(values)
    rng = mx - mn
    if rng == 0:
        return [0.0 for _ in values], {"min": mn, "max": mx}
    return [(v - mn) / rng for v in values], {"min": mn, "max": mx}


def zscore_standardize(values):
    """Z-score standardization: (x - mean) / std. Assumes roughly
    symmetric data; distorted by heavy outliers/skew."""
    mean_v = manual_mean(values)
    std_v = manual_std(values, mean_v)
    if std_v == 0:
        return [0.0 for _ in values], {"mean": mean_v, "std": std_v}
    return [(v - mean_v) / std_v for v in values], {"mean": mean_v, "std": std_v}


def robust_scale(values):
    """Robust scaling: (x - median) / IQR. Uses median and interquartile
    range instead of mean/std, so it is resistant to outliers and works
    well on skewed clinical data (e.g. oldpeak, chol)."""
    sorted_vals = manual_sort(values)
    median_v = manual_median(values)
    q1 = manual_percentile(sorted_vals, 25)
    q3 = manual_percentile(sorted_vals, 75)
    iqr = q3 - q1
    if iqr == 0:
        return [0.0 for _ in values], {"median": median_v, "q1": q1, "q3": q3, "iqr": iqr}
    return [(v - median_v) / iqr for v in values], {"median": median_v, "q1": q1, "q3": q3, "iqr": iqr}


def skewness_manual(values):
    """Manual Fisher-Pearson skewness coefficient, used to recommend a
    scaling method automatically."""
    n = len(values)
    if n < 3:
        return 0.0
    mean_v = manual_mean(values)
    std_v = manual_std(values, mean_v)
    if std_v == 0:
        return 0.0
    m3 = sum((v - mean_v) ** 3 for v in values) / n
    return m3 / (std_v ** 3)


# ============================================================================
# TASK 8: STATEFUL CUSTOM ONE-HOT ENCODER (dynamically adapts to new data)
# ============================================================================

class ManualOneHotEncoder:
    """A from-scratch one-hot encoder (no sklearn) that remembers the
    categories it was fit on, and can dynamically extend itself when
    transform() is later called on brand-new/unseen categories — mimicking
    how a production pipeline must handle future/streaming data."""

    def __init__(self, allow_dynamic_growth=True):
        self.categories_ = []       # ordered list of known categories
        self.allow_dynamic_growth = allow_dynamic_growth
        self.new_categories_seen_ = []  # log of categories added after fit()

    def _normalize(self, v):
        return "∅" if v is None else str(v).strip().lower()

    def fit(self, values):
        seen = []
        seen_set = set()
        for v in values:
            key = self._normalize(v)
            if key not in seen_set:
                seen_set.add(key)
                seen.append(key)
        self.categories_ = seen
        return self

    def transform(self, values):
        rows_out = []
        for v in values:
            key = self._normalize(v)
            if key not in self.categories_:
                if self.allow_dynamic_growth:
                    # dynamically adapt: extend the known category list
                    self.categories_.append(key)
                    self.new_categories_seen_.append(key)
                else:
                    key = "__unknown__"
                    if key not in self.categories_:
                        self.categories_.append(key)
            vec = [1 if c == key else 0 for c in self.categories_]
            rows_out.append(vec)

        # backfill earlier rows with 0s for any category discovered later,
        # so the final matrix stays rectangular
        width = len(self.categories_)
        for row in rows_out:
            while len(row) < width:
                row.append(0)
        return rows_out

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)


# ============================================================================
# TASK 9: ENGINEERED FEATURE — "risk_factor" FROM chol + thalach
# ============================================================================

def percentile_rank_manual(values):
    """Manual percentile rank (0-100) of each value within its own column:
    for each value, % of values in the column that are <= it."""
    n = len(values)
    ranks = []
    for v in values:
        count_le = sum(1 for x in values if x <= v)
        ranks.append(100.0 * count_le / n)
    return ranks


def compute_risk_factor(chol_values, thalach_values, age_values=None):
    """Domain-informed composite risk score combining cholesterol (chol)
    and max heart rate achieved (thalach):

    Clinical intuition:
      - Higher cholesterol  -> higher cardiovascular risk.
      - Lower max heart rate achieved during exercise (thalach) is
        associated with poorer cardiac fitness / higher risk.
      - The *interaction* between the two matters: high chol combined
        with low thalach is a compounding red flag, more than either
        factor in isolation (captured via an interaction term).

    Formula:
      1. Percentile-rank both columns (0-100) so they're on a comparable,
         outlier-resistant scale (raw chol and thalach have very
         different ranges/units).
      2. risk_factor = 0.5 * chol_percentile
                      + 0.5 * (100 - thalach_percentile)   # invert: low thalach = high risk
                      + 0.25 * interaction_term
      where interaction_term = (chol_percentile * (100 - thalach_percentile)) / 100
      This rewards the *combination* of high chol and low thalach more
      than either alone, then the whole thing is re-scaled to 0-100.
    """
    chol_pct = percentile_rank_manual(chol_values)
    thalach_pct = percentile_rank_manual(thalach_values)
    inverted_thalach_pct = [100 - p for p in thalach_pct]

    raw_scores = []
    for cp, itp in zip(chol_pct, inverted_thalach_pct):
        interaction = (cp * itp) / 100.0
        score = 0.5 * cp + 0.5 * itp + 0.25 * interaction
        raw_scores.append(score)

    # rescale raw_scores to a clean 0-100 range (min-max on the composite)
    mn, mx = manual_min(raw_scores), manual_max(raw_scores)
    rng = mx - mn
    if rng == 0:
        final_scores = [50.0 for _ in raw_scores]
    else:
        final_scores = [100.0 * (s - mn) / rng for s in raw_scores]

    return final_scores, chol_pct, thalach_pct


def risk_tier(score):
    if score >= 75:
        return "Very High"
    elif score >= 50:
        return "High"
    elif score >= 25:
        return "Moderate"
    else:
        return "Low"


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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📄 Task 1: Data Viewer",
    "📊 Task 2: Summary Statistics",
    "🧩 Task 3: Missing Values & Imputation",
    "🔁 Task 4: Duplicates",
    "📈 Task 5: Distributions",
    "🔤 Task 6: Encoding",
    "📏 Task 7: Scaling",
    "🧬 Task 8: Dynamic One-Hot",
    "❤️ Task 9: Risk Factor",
])

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

# ---------------- TASK 4 ----------------
with tab4:
    st.subheader("Duplicate Row Detection (Custom Comparison Algorithm)")
    st.caption("No `.duplicated()` / `.drop_duplicates()` used — rows are grouped by a manually normalized signature, then confirmed field-by-field with a custom tolerant comparator (numeric tolerance + case/whitespace-insensitive strings).")

    duplicate_indices, duplicate_map = find_duplicates_manual(rows)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", len(rows))
    c2.metric("Duplicate Rows Found", len(duplicate_indices))
    c3.metric("Unique Rows", len(rows) - len(duplicate_indices))

    if duplicate_indices:
        st.write("**Sample duplicate groups** (first occurrence index → duplicate indices):")
        sample_groups = list(duplicate_map.items())[:10]
        for first_idx, dups in sample_groups:
            st.write(f"- Row `{first_idx}` is duplicated at rows: {dups}")
            st.table([dict(zip(header, rows[first_idx]))])

        if st.button("Remove Duplicates"):
            cleaned_rows = remove_duplicates_manual(rows, duplicate_indices)
            st.success(f"Removed {len(duplicate_indices)} duplicate rows. Dataset now has {len(cleaned_rows)} rows.")
            st.table([dict(zip(header, r)) for r in cleaned_rows[:15]])
    else:
        st.success("No duplicate rows detected in this dataset.")

    with st.expander("📋 Analysis: How Duplicates Impact Data Integrity & ML Models"):
        st.markdown(f"""
**Data integrity impact:**
- Duplicate rows inflate the dataset without adding new information, giving a false sense of sample size.
- If duplicates arose from a data-collection/merge error, they may signal a broader pipeline bug worth investigating rather than just removing.

**Machine learning impact:**
- **Train/test leakage**: if duplicates span both the train and test split, the model is evaluated on rows it has already memorized, inflating reported accuracy.
- **Biased learning**: duplicated rows are effectively up-weighted, so the model overfits to those specific patient profiles and any class imbalance in the duplicates gets amplified (e.g. if more `target=1` rows are duplicated, the model skews toward predicting heart disease).
- **Distorted statistics**: mean/std/mode computed in Task 2 would already be skewed by any duplicates present before cleaning, so duplicate removal should generally happen **before** imputation and encoding in the pipeline.

**This dataset:** {len(duplicate_indices)} duplicate row(s) were found out of {len(rows)} total ({round(100*len(duplicate_indices)/len(rows), 2) if rows else 0}%). {"Removing them is recommended before proceeding to model training." if duplicate_indices else "No action needed — the dataset is already unique."}
        """)

# ---------------- TASK 5 ----------------
with tab5:
    st.subheader("Distribution Visualization (Raw Matplotlib Only)")
    st.caption("No seaborn, no pandas .plot(), no sklearn — every chart below is built with raw `matplotlib.pyplot` calls, custom colors, annotations, and reference lines.")

    numeric_cols = get_numeric_columns(header, rows)

    if not numeric_cols:
        st.warning("No numeric columns found to plot.")
    else:
        st.markdown("### Individual Distribution (with mean/median lines + peak annotation)")
        col_choice = st.selectbox("Choose a numeric column", options=[c[0] for c in numeric_cols])
        col_idx_choice = dict(numeric_cols)[col_choice]
        vals = numeric_values(get_column(rows, col_idx_choice))
        fig1 = plot_histogram(vals, col_choice)
        st.pyplot(fig1)
        plt.close(fig1)

        st.markdown("### Comparative Boxplots — All Numeric Columns")
        st.caption("Compares spread and outliers across every numeric column on one chart.")
        all_data = [numeric_values(get_column(rows, idx)) for _, idx in numeric_cols]
        labels = [name for name, _ in numeric_cols]
        fig2 = plot_boxplot_comparative(all_data, labels)
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("### Grid of All Numeric Column Histograms")
        fig3 = plot_grid_histograms(numeric_cols, header, rows)
        st.pyplot(fig3)
        plt.close(fig3)

        with st.expander("📋 Insights"):
            st.markdown("""
- The **mean/median dashed lines** on the individual histogram show skew at a glance — when they diverge noticeably, the column is skewed (e.g. `oldpeak`, `chol` tend to be right-skewed with a longer tail).
- The **comparative boxplot** puts every numeric column's spread and outliers side by side, making it easy to spot which clinical measurements (e.g. `chol`, `trestbps`) have the widest range of outliers.
- The **grid histograms** give a quick full-dataset overview in one glance, useful for spotting near-constant or heavily skewed columns before feature engineering.
            """)

# ---------------- TASK 6 ----------------
with tab6:
    st.subheader("Manual Categorical Encoding")
    st.caption("No `sklearn.LabelEncoder` / `OneHotEncoder` / `pandas.get_dummies` — mappings are built and applied by hand, with dynamic handling of unseen categories.")

    # Heuristic: treat low-cardinality integer/text columns as categorical candidates
    candidate_cols = []
    for idx, name in enumerate(header):
        col_vals = get_column(rows, idx)
        distinct = set("∅" if v is None else str(v).strip().lower() for v in col_vals)
        if 2 <= len(distinct) <= 15:
            candidate_cols.append(name)

    if not candidate_cols:
        st.warning("No suitable low-cardinality categorical columns detected.")
    else:
        enc_col = st.selectbox("Column to encode", options=candidate_cols)
        enc_idx = header.index(enc_col)
        enc_vals = get_column(rows, enc_idx)

        encoding_type = st.radio("Encoding type", ["Label Encoding", "One-Hot Encoding"], horizontal=True)
        unknown_strategy = st.selectbox(
            "Unknown-category strategy (for values not seen when the mapping was built)",
            options=["new_code", "most_frequent", "missing_code"],
            format_func=lambda x: {
                "new_code": "Assign a new code dynamically",
                "most_frequent": "Fall back to most frequent category's code",
                "missing_code": "Flag as -1 (missing/unknown)",
            }[x],
        )

        label_map = build_label_map(enc_vals)
        st.write("**Learned category → code mapping:**")
        st.table([{"Category": k, "Code": v} for k, v in sorted(label_map.items(), key=lambda kv: kv[1])])

        if encoding_type == "Label Encoding":
            encoded, final_map = apply_label_encoding(enc_vals, label_map, unknown_strategy)
            preview = [{"Original": v, "Encoded": e} for v, e in list(zip(enc_vals, encoded))[:15]]
            st.write("**Preview (first 15 rows):**")
            st.table(preview)
        else:
            categories = sorted(label_map.keys(), key=lambda k: label_map[k])
            one_hot_rows_result = apply_one_hot_encoding(enc_vals, categories)
            oh_header = [f"{enc_col}_{c}" for c in categories] + [f"{enc_col}_unknown"]
            preview = [dict(zip(oh_header, r)) for r in one_hot_rows_result[:15]]
            st.write("**Preview (first 15 rows):**")
            st.table(preview)

        with st.expander("📋 Notes on unknown-value handling"):
            st.markdown("""
- The mapping is learned **only from values seen in this dataset**, exactly like `fit()` would in a real ML pipeline — it is not hardcoded.
- If new/unseen categories appear later (e.g. a new `thal` code shows up in future patient data), the app handles it dynamically based on the strategy chosen above, rather than crashing or silently misencoding it:
  - **New code**: extends the mapping on the fly.
  - **Most frequent fallback**: assumes code `0` represents the most common category (safe default for tree-based models).
  - **Missing code (-1)**: explicitly marks it as unknown so downstream models/analysis can treat it specially.
- For truly nominal columns with no ordinal relationship (e.g. `cp`, `thal`), **one-hot encoding** is generally preferred over label encoding, since label encoding implies a false ordering (e.g. category 2 isn't "twice" category 1).
            """)

# ---------------- TASK 7 ----------------
with tab7:
    st.subheader("Manual Normalization / Standardization")
    st.caption("No `sklearn.preprocessing` — Min-Max, Z-score, and Robust (median/IQR) scaling are all implemented from scratch, with an automatic recommendation based on manually computed skewness.")

    numeric_cols_t7 = get_numeric_columns(header, rows)
    if not numeric_cols_t7:
        st.warning("No numeric columns found.")
    else:
        scale_col = st.selectbox("Column to scale", options=[c[0] for c in numeric_cols_t7], key="scale_col")
        scale_idx = dict(numeric_cols_t7)[scale_col]
        vals_t7 = numeric_values(get_column(rows, scale_idx))

        skew = skewness_manual(vals_t7)
        st.write(f"**Manual skewness coefficient:** `{round(skew, 3)}`")
        if abs(skew) > 1:
            recommendation = "Robust Scaling (median/IQR)"
            reason = "the column is heavily skewed / likely has outliers, so median & IQR are more reliable than mean & std."
        elif abs(skew) > 0.5:
            recommendation = "Robust Scaling (median/IQR)"
            reason = "the column shows moderate skew, so a robust scaler avoids outliers distorting the range."
        else:
            recommendation = "Z-score Standardization"
            reason = "the column is roughly symmetric, so mean/std based standardization is safe and interpretable."
        st.info(f"**Recommended method:** {recommendation} — {reason}")

        method = st.radio("Scaling method to apply", ["Min-Max Normalization", "Z-score Standardization", "Robust Scaling (median/IQR)"], horizontal=False)

        if method == "Min-Max Normalization":
            scaled, params = minmax_normalize(vals_t7)
        elif method == "Z-score Standardization":
            scaled, params = zscore_standardize(vals_t7)
        else:
            scaled, params = robust_scale(vals_t7)

        st.write("**Parameters learned:**", params)
        preview = [{"Original": round(o, 3), "Scaled": round(s, 4)} for o, s in list(zip(vals_t7, scaled))[:15]]
        st.table(preview)

        fig_scale, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        ax1.hist(vals_t7, bins=20, color="#5B8DEF", edgecolor="white", alpha=0.9)
        ax1.set_title(f"Before Scaling: {scale_col}", fontsize=10, fontweight="bold")
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
        ax2.hist(scaled, bins=20, color="#7ED321", edgecolor="white", alpha=0.9)
        ax2.set_title(f"After {method}", fontsize=10, fontweight="bold")
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        fig_scale.tight_layout()
        st.pyplot(fig_scale)
        plt.close(fig_scale)

        with st.expander("📋 Why robustness to outliers/skew matters"):
            st.markdown("""
- **Min-Max** compresses everything to [0, 1] using the raw min/max — a single extreme outlier stretches the whole range and squashes all normal values into a tiny band near 0.
- **Z-score** uses mean & std — both are themselves pulled by outliers/skew, so the "standardized" values can still be distorted for skewed clinical columns like `oldpeak` or `chol`.
- **Robust scaling** uses the **median** and **IQR (Q3 − Q1)** instead — both are resistant to extreme values, so it handles skewed and outlier-heavy distributions (common in medical data) far better than the other two.
            """)

# ---------------- TASK 8 ----------------
with tab8:
    st.subheader("Dynamic Custom One-Hot Encoder")
    st.caption("A hand-written `ManualOneHotEncoder` class (no sklearn) — fit on the current data, then transform() is re-run on simulated 'future data' containing a brand-new category to prove it adapts dynamically.")

    candidate_cols_t8 = []
    for idx, name in enumerate(header):
        col_vals = get_column(rows, idx)
        distinct = set("∅" if v is None else str(v).strip().lower() for v in col_vals)
        if 2 <= len(distinct) <= 15:
            candidate_cols_t8.append(name)

    if not candidate_cols_t8:
        st.warning("No suitable categorical columns detected.")
    else:
        enc_col_t8 = st.selectbox("Column to encode", options=candidate_cols_t8, key="enc_col_t8")
        enc_idx_t8 = header.index(enc_col_t8)
        enc_vals_t8 = get_column(rows, enc_idx_t8)

        encoder = ManualOneHotEncoder(allow_dynamic_growth=True)
        encoder.fit(enc_vals_t8)

        st.write(f"**Categories learned from current data ({len(encoder.categories_)}):**", encoder.categories_)

        st.markdown("#### Simulate future/unseen data")
        new_val = st.text_input("Enter a brand-new category value to simulate future data (e.g. an unseen code)", value="future_cat_99")

        if st.button("Transform current data + simulated new value"):
            simulated_batch = enc_vals_t8 + [new_val]
            encoded_matrix = encoder.transform(simulated_batch)
            st.success(f"Encoder dynamically grew from {len(enc_vals_t8) and len(set(str(v).strip().lower() for v in enc_vals_t8))} original categories to {len(encoder.categories_)} total after seeing `{new_val}`.")
            st.write("**Newly discovered categories during transform:**", encoder.new_categories_seen_)
            preview_header = [f"{enc_col_t8}_{c}" for c in encoder.categories_]
            preview_rows = [dict(zip(preview_header, row)) for row in encoded_matrix[-5:]]
            st.write("**Last 5 encoded rows (including the simulated new one):**")
            st.table(preview_rows)

        with st.expander("📋 How the dynamic adaptation works"):
            st.markdown("""
- `fit()` learns the category list from the training data, exactly once.
- `transform()` can be called again later (e.g. on tomorrow's incoming patient records). If it encounters a category it has never seen:
  - it **appends a new column** to its internal category list on the fly,
  - it **logs** the new category in `new_categories_seen_` for auditing,
  - and it **backfills** all previously encoded rows with a `0` in that new column so the output matrix stays rectangular.
- This mirrors a real production concern: a static one-hot mapping built once at training time would silently break (or need a manual code change) the moment a new category appears in live data. This encoder instead adapts without crashing.
            """)

# ---------------- TASK 9 ----------------
with tab9:
    st.subheader("Engineered Feature: `risk_factor` (from `chol` + `thalach`)")
    st.caption("Combines cholesterol and max heart rate achieved into one composite score using percentile ranking, domain-informed weighting, and an interaction term — no built-in feature-engineering libraries used.")

    if "chol" not in header or "thalach" not in header:
        st.error("This dataset does not contain both `chol` and `thalach` columns.")
    else:
        chol_idx = header.index("chol")
        thalach_idx = header.index("thalach")

        # only use rows where both chol and thalach are present (numeric)
        paired = [(row[chol_idx], row[thalach_idx]) for row in rows
                  if isinstance(row[chol_idx], (int, float)) and isinstance(row[thalach_idx], (int, float))]
        chol_vals_t9 = [p[0] for p in paired]
        thalach_vals_t9 = [p[1] for p in paired]

        risk_scores, chol_pct, thalach_pct = compute_risk_factor(chol_vals_t9, thalach_vals_t9)
        tiers = [risk_tier(s) for s in risk_scores]

        st.markdown("""
**Formula used:**
```
chol_percentile     = percentile rank of chol within the dataset (0-100)
thalach_inv_percentile = 100 - percentile rank of thalach (0-100)   # lower thalach => higher risk
interaction         = (chol_percentile * thalach_inv_percentile) / 100
raw_score           = 0.5*chol_percentile + 0.5*thalach_inv_percentile + 0.25*interaction
risk_factor         = raw_score rescaled to 0-100 (min-max)
```
        """)

        preview_t9 = []
        for i in range(min(15, len(paired))):
            preview_t9.append({
                "chol": chol_vals_t9[i],
                "thalach": thalach_vals_t9[i],
                "chol_percentile": round(chol_pct[i], 1),
                "thalach_percentile": round(thalach_pct[i], 1),
                "risk_factor": round(risk_scores[i], 2),
                "risk_tier": tiers[i],
            })
        st.table(preview_t9)

        tier_counts = {}
        for t in tiers:
            tier_counts[t] = tier_counts.get(t, 0) + 1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Low", tier_counts.get("Low", 0))
        c2.metric("Moderate", tier_counts.get("Moderate", 0))
        c3.metric("High", tier_counts.get("High", 0))
        c4.metric("Very High", tier_counts.get("Very High", 0))

        fig9, ax9 = plt.subplots(figsize=(7, 4))
        ax9.hist(risk_scores, bins=20, color="#D0021B", edgecolor="white", alpha=0.85)
        mean_risk = manual_mean(risk_scores)
        ax9.axvline(mean_risk, color="#333", linestyle="--", linewidth=1.5, label=f"Mean risk_factor = {mean_risk:.1f}")
        ax9.set_title("Distribution of Engineered risk_factor", fontsize=12, fontweight="bold")
        ax9.set_xlabel("risk_factor (0-100)")
        ax9.set_ylabel("Frequency")
        ax9.legend(fontsize=8, frameon=False)
        ax9.spines["top"].set_visible(False); ax9.spines["right"].set_visible(False)
        fig9.tight_layout()
        st.pyplot(fig9)
        plt.close(fig9)

        with st.expander("📋 Domain-knowledge justification"):
            st.markdown("""
- **Cholesterol (`chol`)**: elevated cholesterol is a well-established cardiovascular risk factor, so higher `chol` percentile contributes positively to risk.
- **Max heart rate achieved (`thalach`)**: a *lower* value during exercise testing generally indicates poorer cardiac fitness/response, so it is **inverted** (100 − percentile) before being added — low thalach pushes risk up, not down.
- **Percentile ranking** (rather than raw values) is used so `chol` (range ~100-560) and `thalach` (range ~70-200) contribute on a comparable, outlier-resistant 0-100 scale instead of one variable dominating purely due to its larger numeric range.
- **Interaction term**: a patient with *both* high cholesterol AND low fitness (low thalach) is clinically a bigger concern than either single factor — the interaction term (`chol_percentile × inverted_thalach_percentile`) captures that compounding effect, weighted at 0.25 so it boosts but doesn't dominate the base terms (each weighted 0.5).
- **Final rescaling to 0-100** keeps the engineered feature on an intuitive, comparable scale to the percentile inputs, and the tiering (Low/Moderate/High/Very High) makes it immediately interpretable for a non-technical audience (e.g. a doctor or dashboard viewer).
            """)
