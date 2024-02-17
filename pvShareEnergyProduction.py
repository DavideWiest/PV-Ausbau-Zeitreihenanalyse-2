import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

string = open("dataRaw/Realisierte_Erzeugung_201701010000_202312242359_Stunde.csv", "r", encoding="utf-8").read()
string = string.replace(".", "").replace(",", ".").replace("-", "0")

dfProduction = pd.read_csv(StringIO(string), delimiter=";", header=0)

dfProduction["PV"] = pd.to_numeric(dfProduction["Photovoltaik [MWh] Berechnete Auflösungen"])
dfProduction["FlexbileConventional"] = \
    pd.to_numeric(dfProduction["Braunkohle [MWh] Berechnete Auflösungen"]) + \
    pd.to_numeric(dfProduction["Steinkohle [MWh] Berechnete Auflösungen"]) + \
    pd.to_numeric(dfProduction["Erdgas [MWh] Berechnete Auflösungen"]) + \
    pd.to_numeric(dfProduction["Pumpspeicher [MWh] Berechnete Auflösungen"]) + \
    pd.to_numeric(dfProduction["Sonstige Konventionelle [MWh] Berechnete Auflösungen"])

dfProduction["Zeit"] = dfProduction["Datum"].astype(str) + dfProduction["Anfang"].astype(str)
dfProduction = dfProduction.loc[:, ["Zeit", "PV", "FlexbileConventional"]]

string2 = open("dataRaw/Realisierter_Stromverbrauch_201701010000_202312242359_Stunde.csv", "r", encoding="utf-8").read()
string2 = string2.replace(".", "").replace(",", ".").replace("-", "0")


dfConsumption = pd.read_csv(StringIO(string2), delimiter=";", header=0)
dfConsumption["Zeit"] = dfConsumption["Datum"].astype(str) + dfConsumption["Anfang"].astype(str)
dfConsumption = dfConsumption.rename({"Gesamt (Netzlast) [MWh] Berechnete Auflösungen": "Netzlast"}, axis="columns")

dfConsumption = dfConsumption.loc[:, ["Zeit", "Netzlast"]]


print(dfProduction)
print(dfConsumption)

print(dfProduction.shape)
print(dfConsumption.shape)

merged_df = pd.merge(dfProduction, dfConsumption, on="Zeit")
rangeList = []

for i in range(10):
    growthFactor = (i+1)/10

    loss = (
        dfConsumption + dfProduction["FlexbileConventional"] - dfProduction["PV"] * growthFactor
    ).sum()

    rangeList.append((growthFactor, 1)) # loss/1000 # convert from MWh to GWh

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "8"

print(rangeList)

plt.plot(
    [i[0] for i in rangeList],
    [i[1] for i in rangeList],
)

plt.xlabel("Wachstum der PV-Stromerzeugung in %")
plt.ylabel("Stromverluste (GWh)")
plt.legend()

plt.show()