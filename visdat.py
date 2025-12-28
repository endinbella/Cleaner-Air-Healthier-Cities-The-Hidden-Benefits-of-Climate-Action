#!/usr/bin/env python
# coding: utf-8

# # LEVEL 1 (SPASIAL)
# 
# Dataset: Level 1 – UK Co-Benefits Atlas
# 
# 1.   Fokus : Air Quality Co-benefit
# 2.   Tool: Google Colab

# ## 1️⃣ IMPORT LIBRARY

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Palet non-default (tanpa biru default)
COL = {
    "orange": "tab:orange",
    "green":  "tab:green",
    "purple": "tab:purple",
    "red":    "tab:red",
    "gray":   "0.25",
}

def hist_pretty(series, title, xlabel, ylabel, bins=40, clip_q=0.99, fmt="{x:,.2f}",
                bar_color=COL["orange"], mean_color=COL["gray"], median_color=COL["green"]):
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()

    # zoom p99 biar outlier/tail nggak “narik” histogram
    if clip_q is not None:
        lo, hi = s.min(), s.quantile(clip_q)
        s_plot = s.clip(lo, hi)
        xlim = (lo, hi)
    else:
        s_plot = s
        xlim = None

    mean, med = s.mean(), s.median()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(s_plot, bins=bins, color=bar_color, edgecolor="white", linewidth=0.6, alpha=0.9)

    ax.axvline(mean, linestyle="--", linewidth=1.2, color=mean_color, label=f"Mean = {mean:.3f}")
    ax.axvline(med,  linestyle="-",  linewidth=1.2, color=median_color, label=f"Median = {med:.3f}")

    ax.set_title(title, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter(fmt))
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.text(
        0.98, 0.95,
        f"N={len(s):,}\nmin={s.min():.3f}\nmax={s.max():.3f}",
        transform=ax.transAxes, ha="right", va="top"
    )

    plt.tight_layout()
    plt.show()


# ## 2️⃣ LOAD DATASET LEVEL 1

# In[ ]:


df_lvl1 = pd.read_excel("Level_1.xlsx")

display(df_lvl1.head())
print("Shape Level 1:", df_lvl1.shape)


# ## 3️⃣ CEK & BERSIHKAN DATA

# In[ ]:


print("\nINFO Level 1:")
df_lvl1.info()

print("\nMissing values per kolom Level 1:")
display(df_lvl1.isna().sum())


# ## 4️⃣ FILTER CO-BENEFIT: AIR QUALITY

# In[ ]:


df_air_lvl1 = df_lvl1.loc[:, ["small_area", "air_quality"]].copy()

df_air_lvl1["air_quality"] = pd.to_numeric(df_air_lvl1["air_quality"], errors="coerce")
df_air_lvl1 = df_air_lvl1.dropna(subset=["small_area", "air_quality"]).reset_index(drop=True)

print("\nINFO df_air_lvl1:")
df_air_lvl1.info()

print("\nDESKRIPTIF df_air_lvl1:")
display(df_air_lvl1["air_quality"].describe())


# ## 5️⃣ ANALISIS SPASIAL SEDERHANA

# In[ ]:


mean_val = df_air_lvl1["air_quality"].mean()
median_val = df_air_lvl1["air_quality"].median()
min_val = df_air_lvl1["air_quality"].min()
max_val = df_air_lvl1["air_quality"].max()

print("\nRingkasan Level 1:")
print(f"Mean   : {mean_val:.6f}")
print(f"Median : {median_val:.6f}")
print(f"Min    : {min_val:.6f}")
print(f"Max    : {max_val:.6f}")
print(f"Gap(all): {(max_val - min_val):.6f}")


# ## 6️⃣ VISUALISASI DISTRIBUSI (HISTOGRAM) — SELURUH UK

# In[ ]:


hist_pretty(
    df_air_lvl1["air_quality"],
    title="Distribusi Air Quality Co-Benefit Index\nSeluruh UK Small Areas",
    xlabel="Air Quality Co-Benefit Index",
    ylabel="Jumlah Small Areas",
    bins=40,
    clip_q=0.99,
    fmt="{x:,.2f}",
    bar_color=COL["orange"]     # <- warna bar
)


# # LEVEL 2 (CO-BENEFIT PENDUKUNG)
# 
# Dataset: Level 2 – UK Co-Benefits Atlas
# 1. Fungsi: Pendalaman & relasi antar co-benefits
# 2. Output: Visual pendukung + insight korelasi
# 3. Tool: Google Colab

# ## 1️⃣ Load Dataset

# In[7]:


df_lvl2 = pd.read_excel("Level_2.xlsx")

display(df_lvl2.head())
print("Shape Level 2:", df_lvl2.shape)


# ## 2️⃣ Cek dan Bersihkan Data

# In[8]:


print("\nINFO Level 2:")
df_lvl2.info()

print("\nMissing values per kolom Level 2:")
display(df_lvl2.isna().sum())


# ## 3️⃣ Filter CO-Benefit

# In[9]:


df_filtered_level2 = df_lvl2[df_lvl2["small_area"] == "E01028521"].copy()

print("\nINFO df_filtered_level2:")
df_filtered_level2.info()

# contoh: lihat row air_quality
air_quality_data = df_filtered_level2[df_filtered_level2["co-benefit_type"] == "air_quality"]
display(air_quality_data)


# ## 4️⃣ Ambil data air_quality (Level 2) + pilih kolom tahun 2025–2050

# In[10]:


df_air_lvl2 = df_lvl2[df_lvl2["co-benefit_type"] == "air_quality"].copy()

# ambil kolom tahun (kadang Excel kebaca int, kadang string)
year_cols = []
for c in df_air_lvl2.columns:
    if isinstance(c, int) and 2025 <= c <= 2050:
        year_cols.append(c)
    elif isinstance(c, str) and c.isdigit() and 2025 <= int(c) <= 2050:
        year_cols.append(c)

year_cols = sorted(year_cols, key=lambda x: int(x))
print("Jumlah kolom tahun:", len(year_cols), "| contoh:", year_cols[:5], "...", year_cols[-3:])


# ## 5️⃣ (A) Histogram nilai tahunan 2025–2050 (tanpa sum)

# In[11]:


# 5️⃣ (A) Histogram nilai tahunan 2025–2050 (tanpa sum)
df_air_long = df_air_lvl2.melt(
    id_vars=["small_area"],
    value_vars=year_cols,
    var_name="year",
    value_name="air_quality_value"
)

# optional tapi aman
df_air_long["air_quality_value"] = pd.to_numeric(df_air_long["air_quality_value"], errors="coerce")


# In[12]:


hist_pretty(
    df_air_long["air_quality_value"],
    title="Distribusi Air Quality (Tahunan 2025–2050)\nSeluruh UK Small Areas",
    xlabel="Air Quality (annual value)",
    ylabel="Jumlah data (small_area x tahun)",
    bins=40,
    clip_q=0.99,
    fmt="{x:,.3f}",
    bar_color=COL["green"]
)


# ## 6️⃣ (B) Histogram sum Level 2 (total 2025–2050)

# In[13]:


hist_pretty(
    df_air_lvl2["sum"],
    title="Distribusi Air Quality SUM (2025–2050)\nSeluruh UK Small Areas",
    xlabel="Air Quality (sum 2025–2050)",
    ylabel="Jumlah Small Areas",
    bins=40,
    clip_q=0.99,
    fmt="{x:,.2f}",
    bar_color=COL["purple"]
)


# ## 7️⃣ GAP per small_area (Level 2) — khusus antar tahun (tanpa sum)

# In[14]:


tmp = df_air_lvl2.set_index("small_area")[year_cols]
gap_small_area_year = pd.DataFrame({
    "min": tmp.min(axis=1),
    "max": tmp.max(axis=1),
})
gap_small_area_year["gap"] = gap_small_area_year["max"] - gap_small_area_year["min"]

print("\nContoh gap antar tahun (Level 2) per small_area:")
display(gap_small_area_year.head())


# # FINAL (INTEGRASI LEVEL 1 + LEVEL 2)
# Tujuan: Menggabungkan ringkasan Level 1 dengan time-series Level 2 supaya siap untuk:
# - ranking & segmentasi prioritas
# - analisis tren (2025–2050)
# - korelasi antar co-benefits
# - output final (CSV/Excel)
# 

# ## 1️⃣ BENTUK TABEL WIDE LEVEL 2 + MERGE KE LEVEL 1
# 

# In[15]:


df_air_lvl2_wide = df_air_lvl2[["small_area", "sum"] + year_cols].copy()
df_air_lvl2_wide = df_air_lvl2_wide.rename(columns={"sum": "air_quality_sum_2025_2050"})

# safety kalau ada duplikat
if df_air_lvl2_wide["small_area"].duplicated().any():
    df_air_lvl2_wide = df_air_lvl2_wide.groupby("small_area", as_index=False).first()

df_master = df_lvl1.merge(df_air_lvl2_wide, on="small_area", how="left")

print("Shape df_master:", df_master.shape)
print("Null air_quality_sum_2025_2050:", df_master["air_quality_sum_2025_2050"].isna().sum())
display(df_master.head())


# ## 1️⃣ BENTUK TABEL WIDE LEVEL 2 + MERGE KE LEVEL 1
# 

# In[16]:


df_master["diff_L1_vs_L2sum"] = df_master["air_quality"] - df_master["air_quality_sum_2025_2050"]

display(df_master["diff_L1_vs_L2sum"].describe())
print("Jumlah diff != 0:", (df_master["diff_L1_vs_L2sum"].abs() > 1e-9).sum())


# ## 3️⃣ FITUR TURUNAN: MIN/MAX/GAP + YEAR OF MAX + SLOPE TREND
# 

# In[17]:


# 0) pastikan year_cols yang dipakai memang ada di df_master
year_cols_ok = [c for c in year_cols if c in df_master.columns]
missing = set(year_cols) - set(year_cols_ok)
if missing:
    print("WARNING: year_cols tidak ada di df_master:", sorted(list(missing))[:10], "...")
print("Jumlah kolom tahun dipakai:", len(year_cols_ok))

# 1) ambil time-series TANPA set_index (biar assignment aman)
ts = df_master[year_cols_ok].apply(pd.to_numeric, errors="coerce")

# 2) MIN / MAX / GAP
df_master["min_annual"] = ts.min(axis=1, skipna=True)
df_master["max_annual"] = ts.max(axis=1, skipna=True)
df_master["gap_annual"] = df_master["max_annual"] - df_master["min_annual"]

# 3) YEAR OF MAX / MIN
df_master["year_of_max"] = ts.idxmax(axis=1)
df_master["year_of_min"] = ts.idxmin(axis=1)

# rapihin jadi int year (kalau label kolom tahun string/int)
df_master["year_of_max"] = df_master["year_of_max"].apply(lambda v: int(str(v)) if pd.notna(v) else np.nan)
df_master["year_of_min"] = df_master["year_of_min"].apply(lambda v: int(str(v)) if pd.notna(v) else np.nan)

# 4) SLOPE TREND (OLS slope sederhana, robust untuk NaN)
Y = ts.to_numpy(dtype=float)
x = np.arange(len(year_cols_ok), dtype=float)
x_c = x - x.mean()
den = np.sum(x_c**2)

valid_n = np.sum(~np.isnan(Y), axis=1)
y_mean = np.nanmean(Y, axis=1, keepdims=True)

slope = np.nansum((Y - y_mean) * x_c, axis=1) / den
slope[valid_n < 2] = np.nan  # kalau valid tahun < 2, slope tidak meaningful

df_master["slope_trend"] = slope

# 5) cek hasil
display(df_master[["small_area","air_quality","min_annual","max_annual","gap_annual","year_of_max","year_of_min","slope_trend"]].head())
print("NaN check:",
      "min", df_master["min_annual"].isna().sum(),
      "| max", df_master["max_annual"].isna().sum(),
      "| year_max", df_master["year_of_max"].isna().sum(),
      "| slope", df_master["slope_trend"].isna().sum())


# ## 4️⃣ SEGMENTASI PRIORITAS: QUADRANT TOTAL vs TREND
# 

# In[18]:


cut_total = df_master["air_quality"].median()
cut_trend = df_master["slope_trend"].median()

def quadrant(row):
    total = "HighTotal" if row["air_quality"] >= cut_total else "LowTotal"
    trend = "HighTrend" if row["slope_trend"] >= cut_trend else "LowTrend"
    return f"{total}-{trend}"

df_master["quadrant"] = df_master.apply(quadrant, axis=1)

display(df_master["quadrant"].value_counts())


# ## 5️⃣ PLOT SCATTER QUADRANT
# 

# In[19]:


cut_total = df_master["air_quality"].median()
cut_trend = df_master["slope_trend"].median()

x = pd.to_numeric(df_master["air_quality"], errors="coerce")
y = pd.to_numeric(df_master["slope_trend"], errors="coerce")
m = x.notna() & y.notna()

fig, ax = plt.subplots(figsize=(9, 6))
hb = ax.hexbin(x[m], y[m], gridsize=60, mincnt=1, cmap="plasma")  # <- non-biru
fig.colorbar(hb, ax=ax, label="Count")

ax.axvline(cut_total, linewidth=1.2, color=COL["gray"])
ax.axhline(cut_trend, linewidth=1.2, color=COL["gray"])

ax.set_title("Quadrant: Total vs Trend (Air Quality)", weight="bold")
ax.set_xlabel("Air Quality SUM (2025–2050)")
ax.set_ylabel("Trend (slope) 2025–2050")
plt.tight_layout()
plt.show()


# ## 6️⃣ OUTPUT FINAL: MASTER (WIDE) + PANEL (LONG) + SAVE
# 

# In[20]:


# (13 tahun + 13 tahun) biar muat Excel per sheet
year_blocks = [(2025, 2037), (2038, 2050)]

# id_vars: semua kolom Level 1 kebawa di panel
id_vars = df_lvl1.columns.tolist()

df_panel_full = df_master.melt(
    id_vars=id_vars,
    value_vars=year_cols,
    var_name="year",
    value_name="air_quality_annual"
)
df_panel_full["year"] = df_panel_full["year"].astype(int, errors="ignore")

# Simpan CSV
df_master.to_csv("UK_air_quality_master.csv", index=False)
df_panel_full.to_csv("UK_air_quality_panel_FULL.csv", index=False)

# Simpan Excel: master_wide + panel dipecah jadi beberapa sheet
with pd.ExcelWriter("UK_air_quality_outputs.xlsx", engine="openpyxl") as w:
    df_master.to_excel(w, sheet_name="master_wide", index=False)

    for start, end in year_blocks:
        cols_block = [c for c in year_cols if start <= int(c) <= end]

        df_panel_part = df_master.melt(
            id_vars=id_vars,
            value_vars=cols_block,
            var_name="year",
            value_name="air_quality_annual"
        )
        df_panel_part["year"] = df_panel_part["year"].astype(int, errors="ignore")

        df_panel_part.to_excel(w, sheet_name=f"panel_{start}_{end}", index=False)

print("Saved:")
print("- UK_air_quality_master.csv")
print("- UK_air_quality_panel_FULL.csv")
print("- UK_air_quality_outputs.xlsx (master_wide + panel_2025_2037 + panel_2038_2050)")
print("Rows panel full:", len(df_panel_full))


# ## 7️⃣ VISUALISASI HASIL INTEGRASI
# 

# ### Histogram Trend

# In[21]:


hist_pretty(
    df_master["slope_trend"],
    title="Distribusi Trend (Slope) Air Quality",
    xlabel="Slope Trend (2025–2050)",
    ylabel="Jumlah Small Areas",
    bins=40,
    clip_q=0.99,
    fmt="{x:,.5f}",
    bar_color=COL["red"]
)


# ### Histogram Gap

# In[22]:


hist_pretty(
    df_master["gap_annual"],
    title="Distribusi Gap Antar Tahun Air Quality",
    xlabel="Gap Annual (max-min) 2025–2050",
    ylabel="Jumlah Small Areas",
    bins=40,
    clip_q=0.99,
    fmt="{x:,.3f}",
    bar_color=COL["orange"]
)

