import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore

temps = [22, 23, np.nan, np.nan, np.nan, 25, 26, 58, 24, 23,
         22, 21, 23, 24, 25]
df = pd.DataFrame({"Hour": range(1, 16), "Temperature": temps})
# (a) Stats BEFORE cleaning
raw = df["Temperature"].dropna()
mean_before = df["Temperature"].mean()
std_before = df["Temperature"].std()
raw_median = df["Temperature"].median()

print("Stats BEFORE cleaning:")
print(df["Temperature"].describe())
print("Mean:", mean_before, "Std:", std_before)
print("Skewness:", stats.skew(raw))
print("Kurtosis:", stats.kurtosis(raw))
# (b) Linear interpolation for the 3 consecutive missing readings
df["Temp_clean"] = df["Temperature"].interpolate(method="linear")
print("After interpolation, missing values:", df["Temp_clean"].isnull().sum())
# (c) Z-score outlier detection (threshold = 2), replace only flagged value
z = np.abs(zscore(df["Temp_clean"]))
mask = z > 2
for _, row in df[mask].iterrows():
    print("Z-score(>2) outlier detected at Hour:", int(row["Hour"]),
          "(value:", row["Temp_clean"], ")")
print("Replaced with column median:", raw_median)
df.loc[mask, "Temp_clean"] = raw_median
# (d) Z-score standardization of cleaned column
df["Temp_Zscore"] = zscore(df["Temp_clean"])

print("Before/After comparison:")
table = df[["Hour", "Temperature", "Temp_clean", "Temp_Zscore"]]
print(table.rename(columns={"Temperature": "Temp_raw"}))

print("Stats AFTER cleaning:")
print("Mean:", df["Temp_clean"].mean(), "Std:", df["Temp_clean"].std())
print("(compare to BEFORE: Mean", round(mean_before, 2),
      "Std", round(std_before, 2), ")")