import io
import re
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

st.set_page_config(page_title="Excel Name Matcher (Exact + Fuzzy)", page_icon="🔎", layout="wide")

# ---------------------------
# helpers
# ---------------------------
def clean_header(s):
    if s is None: return s
    return str(s).replace("\u00A0", " ").strip()

def normalize_name(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).lower()
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)  # keep alnum+space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def excel_bytes_from_df(df, sheet_name="Matched"):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    out.seek(0)
    return out

# ---------------------------
# UI - Sidebar Controls
# ---------------------------
st.sidebar.header("Matching Options")

use_fuzzy = st.sidebar.checkbox("Enable fuzzy matching (RapidFuzz)", value=True)
threshold = st.sidebar.slider("Fuzzy threshold", min_value=50, max_value=100, value=92, step=1)
scorer_name = st.sidebar.selectbox(
    "Fuzzy scorer",
    ["token_sort_ratio", "WRatio", "ratio", "token_set_ratio"],
    index=0
)

scorers = {
    "token_sort_ratio": fuzz.token_sort_ratio,
    "WRatio": fuzz.WRatio,
    "ratio": fuzz.ratio,
    "token_set_ratio": fuzz.token_set_ratio,
}
scorer = scorers[scorer_name]

st.title("🔎 Excel Name Matcher — Exact + Fuzzy")
st.write(
    "Upload **two Excel files** to match names:\n"
    "- **File 1**: the list to classify (e.g., has *Name of Entity*)\n"
    "- **File 2**: the reference/mapping (e.g., has *Name* and *Category/Type*)\n"
)

# ---------------------------
# File Uploads
# ---------------------------
colA, colB = st.columns(2)

with colA:
    file1 = st.file_uploader("📄 Upload FIRST Excel (list to classify)", type=["xlsx"], key="f1")
with colB:
    file2 = st.file_uploader("📑 Upload SECOND Excel (reference with Name + Category/Type)", type=["xlsx"], key="f2")

if not file1 or not file2:
    st.info("Upload both files to continue.")
    st.stop()

# ---------------------------
# Sheet selection
# ---------------------------
try:
    x1 = pd.ExcelFile(file1)
    x2 = pd.ExcelFile(file2)
except Exception as e:
    st.error(f"Failed to open one of the files: {e}")
    st.stop()

with colA:
    sheet1 = st.selectbox("Sheet in FIRST file", options=x1.sheet_names, key="s1")
with colB:
    sheet2 = st.selectbox("Sheet in SECOND file", options=x2.sheet_names, key="s2")

try:
    df1 = x1.parse(sheet_name=sheet1)
    df2 = x2.parse(sheet_name=sheet2)
except Exception as e:
    st.error(f"Failed to read selected sheet: {e}")
    st.stop()

# Clean headers
df1.columns = [clean_header(c) for c in df1.columns]
df2.columns = [clean_header(c) for c in df2.columns]

# ---------------------------
# Column selection
# ---------------------------
name_cands1 = [c for c in df1.columns if "name" in c.lower()]
name_cands2 = [c for c in df2.columns if "name" in c.lower()]
cat_cands2 = [c for c in df2.columns if any(k in c.lower() for k in ["category","type","class","segment","classification","cat"])]

if not name_cands1:
    st.error("Couldn't find a name-like column in FIRST sheet (look for 'name of entity', etc.).")
    st.write("Columns found:", list(df1.columns))
    st.stop()
if not name_cands2:
    st.error("Couldn't find a name-like column in SECOND sheet.")
    st.write("Columns found:", list(df2.columns))
    st.stop()
if not cat_cands2:
    # Let user pick any column, and we will just mark Found/Not found if no true category
    st.warning("No obvious Category/Type column detected in SECOND sheet. Pick one manually, or choose '(No category column)'.")
    cat_cands2 = ["(No category column)"] + list(df2.columns)

col1, col2, col3 = st.columns(3)
with col1:
    col_name1 = st.selectbox("Name column in FIRST file", options=name_cands1, index=0)
with col2:
    col_name2 = st.selectbox("Name column in SECOND file", options=name_cands2, index=0)
with col3:
    col_cat2 = st.selectbox("Category/Type column in SECOND file", options=cat_cands2, index=0)

# ---------------------------
# Run matching
# ---------------------------
run = st.button("🔁 Run Matching")

if run:
    work1 = df1.copy()
    work2 = df2.copy()

    work1["_norm"] = work1[col_name1].map(normalize_name)
    work2["_norm"] = work2[col_name2].map(normalize_name)

    # Build reference map (allow no category column)
    if col_cat2 == "(No category column)":
        ref_map = (
            work2[["_norm", col_name2]]
            .dropna(subset=["_norm"])
            .drop_duplicates("_norm", keep="first")
            .rename(columns={col_name2: "RefName"})
        )
        ref_map["RefCategory"] = None
    else:
        ref_map = (
            work2[["_norm", col_name2, col_cat2]]
            .dropna(subset=["_norm"])
            .drop_duplicates("_norm", keep="first")
            .rename(columns={col_name2: "RefName", col_cat2: "RefCategory"})
        )

    # Exact merge
    out = work1.merge(ref_map, on="_norm", how="left")
    out["Category"] = out["RefCategory"]
    out["MatchedTo"] = out["RefName"]
    out["MatchMethod"] = out["Category"].apply(lambda x: "Exact" if pd.notna(x) else None)
    out["MatchScore"] = out["Category"].apply(lambda x: 100.0 if pd.notna(x) else None)

    # Fuzzy
    if use_fuzzy:
        need = out["MatchMethod"].isna() & out["_norm"].ne("")
        queries = out.loc[need, "_norm"]

        ref = ref_map.dropna(subset=["_norm"]).drop_duplicates("_norm")
        ref_names = ref["_norm"].tolist()
        ref_cat_by_norm = dict(zip(ref["_norm"], ref["RefCategory"]))
        ref_raw_by_norm = dict(zip(ref["_norm"], ref["RefName"]))

        hits = []
        for idx, q in queries.items():
            best = process.extractOne(q, ref_names, scorer=scorer)
            if best:
                cand_norm, score, _ = best[0], best[1], best[2]
                if score >= threshold:
                    hits.append((idx, cand_norm, score))
        for idx, cand_norm, score in hits:
            out.at[idx, "Category"]    = ref_cat_by_norm.get(cand_norm)
            out.at[idx, "MatchedTo"]   = ref_raw_by_norm.get(cand_norm)
            out.at[idx, "MatchMethod"] = "Fuzzy"
            out.at[idx, "MatchScore"]  = float(score)

    # Not found
    if col_cat2 == "(No category column)":
        # Just mark Found/Not found
        out["Category"] = out["MatchedTo"].apply(lambda x: "Found" if pd.notna(x) else "Not found")
        out["MatchMethod"] = out["MatchMethod"].fillna("Not found")
    else:
        out["Category"] = out["Category"].fillna("Not found")
        out["MatchMethod"] = out["MatchMethod"].fillna("Not found")

    out["MatchScore"] = out["MatchScore"].fillna(0.0)

    # Reorder columns: place results after name
    cols = list(out.columns)
    for c in ["RefName","RefCategory","_norm"]:
        if c in cols: cols.remove(c)
    for c in ["Category","MatchedTo","MatchMethod","MatchScore"]:
        if c in cols: cols.remove(c)
    insert_at = cols.index(col_name1) + 1 if col_name1 in cols else 1
    new_cols = cols[:insert_at] + ["Category","MatchedTo","MatchMethod","MatchScore"] + cols[insert_at:]
    out = out[new_cols]

    # Summary
    total = len(out)
    exact = int((out["MatchMethod"] == "Exact").sum())
    fuzzy_cnt = int((out["MatchMethod"] == "Fuzzy").sum())
    notfound = int((out["MatchMethod"] == "Not found").sum())

    st.success("Matching complete.")
    st.metric("Total rows", total)
    m1, m2, m3 = st.columns(3)
    m1.metric("Exact", exact)
    m2.metric("Fuzzy", fuzzy_cnt)
    m3.metric("Not found", notfound)

    with st.expander("Preview (first 200 rows)"):
        st.dataframe(out.head(200), use_container_width=True)

    # Download
    xbytes = excel_bytes_from_df(out)
    st.download_button(
        label="⬇️ Download matched_output.xlsx",
        data=xbytes,
        file_name="matched_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Show a few unmatched to help debug
    if notfound:
        with st.expander("Unmatched sample (first 20)"):
            st.dataframe(out.loc[out["MatchMethod"]=="Not found", [col_name1]].head(20), use_container_width=True)
