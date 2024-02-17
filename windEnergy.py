import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

string = open("dataRaw/Realisierte_Erzeugung_201701010000_202312242359_Stunde.csv", "r", encoding="utf-8").read()
string = string.replace(".", "").replace(",", ".").replace("-", "0")

dfWind = pd.read_csv(StringIO(string), delimiter=";", header=0)

dfWind["Wind"] = pd.to_numeric(dfWind["Wind Offshore [MWh] Berechnete Auflösungen"]) + pd.to_numeric(dfWind["Wind Onshore [MWh] Berechnete Auflösungen"])
dfWind["Zeit"] = dfWind["Datum"].astype(str) + dfWind["Anfang"].astype(str)
dfWind = dfWind.loc[:, ["Zeit", "Wind"]]

string2 = open("dataRaw/Realisierter_Stromverbrauch_201701010000_202312242359_Stunde.csv", "r", encoding="utf-8").read()
string2 = string2.replace(".", "").replace(",", ".").replace("-", "0")


dfConsumption = pd.read_csv(StringIO(string2), delimiter=";", header=0)
dfConsumption["Zeit"] = dfConsumption["Datum"].astype(str) + dfConsumption["Anfang"].astype(str)
dfConsumption = dfConsumption.rename({"Gesamt (Netzlast) [MWh] Berechnete Auflösungen": "Netzlast"}, axis="columns")

dfConsumption = dfConsumption.loc[:, ["Zeit", "Netzlast"]]


print(dfWind)
print(dfConsumption)

merged_df = pd.merge(dfWind, dfConsumption, on="Zeit")
filtered_df = merged_df[merged_df["Wind"] > (merged_df["Netzlast"] * 0.5)]

number_of_days = filtered_df["Zeit"].nunique() 
print(f"Stunden an denen Windkraft mehr als 50% der Netzlast gedeckt hat {number_of_days}")

filtered_df2 = merged_df[merged_df["Wind"] < (merged_df["Netzlast"] * 0.1)]

number_of_days2 = filtered_df2["Zeit"].nunique() 
print(f"Stunden an denen Windkraft weniger als 10% der Netzlast gedeckt hat {number_of_days2}")

rangeList = []

for i in range(10):
    lo = i/10
    hi = (i+1)/10 

    filtered_df = merged_df[
        (merged_df["Wind"] >= (merged_df["Netzlast"] * lo))
    ][
        (merged_df["Wind"] < (merged_df["Netzlast"] * hi))
    ]

    hours = filtered_df["Zeit"].nunique() 

    rangeList.append((lo, hours))

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "8"

print(rangeList)

plt.plot(
    [i[0] for i in rangeList],
    [i[1] for i in rangeList],
)
plt.xlabel("Anteil der Netzlast")
plt.ylabel("Stunden")
plt.legend()

plt.show()