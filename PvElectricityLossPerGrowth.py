import pandas as pd
from io import StringIO
from matplotlib import pyplot as plt

from darts.utils import timeseries_generation as tg

string = open("dataRaw/Realisierte_Erzeugung_201701010000_202312242359_Stunde.csv", "r", encoding="utf-8").read()
string = string.replace(".", "").replace(",", ".").replace("-", "0")

dfProduction = pd.read_csv(StringIO(string), delimiter=";", header=0)

dfProduction["PV"] = pd.to_numeric(dfProduction["Photovoltaik [MWh] Berechnete Auflösungen"])
dfProduction["OtherRenewables"] = \
    pd.to_numeric(dfProduction["Wasserkraft [MWh] Berechnete Auflösungen"]) + \
    pd.to_numeric(dfProduction["Wind Offshore [MWh] Berechnete Auflösungen"]) + \
    pd.to_numeric(dfProduction["Wind Onshore [MWh] Berechnete Auflösungen"]) + \
    pd.to_numeric(dfProduction["Sonstige Erneuerbare [MWh] Berechnete Auflösungen"])

dfProduction["Zeit"] = dfProduction["Datum"].astype(str) + dfProduction["Anfang"].astype(str)
dfProduction = dfProduction.loc[:, ["Zeit", "PV", "OtherRenewables"]]

string2 = open("dataRaw/Realisierter_Stromverbrauch_201701010000_202312242359_Stunde.csv", "r", encoding="utf-8").read()
string2 = string2.replace(".", "").replace(",", ".").replace("-", "0")


dfConsumption = pd.read_csv(StringIO(string2), delimiter=";", header=0)
dfConsumption["Zeit"] = dfConsumption["Datum"].astype(str) + dfConsumption["Anfang"].astype(str)
dfConsumption = dfConsumption.rename({"Gesamt (Netzlast) [MWh] Berechnete Auflösungen": "Netzlast"}, axis="columns")

dfConsumption = dfConsumption.loc[:, ["Zeit", "Netzlast"]]

# for testing
dfConsumption = dfConsumption
dfProduction = dfProduction


merged_df = pd.merge(dfProduction, dfConsumption, on="Zeit")
rangeList = []

n_years = len(merged_df) / 24 / 365 # hourly resolution

# until growth is +200%, in 1% resolution
for i in range(200+1):
    growthFactor = 1+(i)/100

    loss = (
        (merged_df["Netzlast"] - merged_df["OtherRenewables"] - merged_df["PV"] * growthFactor) * -1
    )
    loss = loss.where(loss>0) # loss from pv overproduction can only be positive
    loss = loss.sum()

    rangeList.append((round(growthFactor, 2), loss/1_000_000/n_years)) # loss/1_000_000 converts MWh to TWh

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "8"

print(rangeList)



plt.plot(
    [(i[0] - 1) * 100 for i in rangeList], # convert factor to percent growth
    [i[1] for i in rangeList],
    label="Stromverlust je Wachstum von PV-Nettostromerzeugung pro Jahr",
    color=(0.2, 0.2, 0.75)
)

nBeatsPredictedGrowthFactor = round(171.025252 / 60.8, 2)# (DeStatis 2023 - Bruttostromerzeugung)
predictedLoss = [x[1] for x in rangeList if x[0] == nBeatsPredictedGrowthFactor][0] # access the expected electricity loss
predictedGrowth = (nBeatsPredictedGrowthFactor - 1) * 100

plt.plot([0, predictedGrowth], [predictedLoss, predictedLoss], color=(0,0,0), label="Prognostizierter Stromverlust")
plt.plot([predictedGrowth, predictedGrowth], [0, predictedLoss], color=(0,0,0))

plt.xlabel("Wachstum der PV-Stromerzeugung in %")
plt.ylabel("Stromverluste (TWh)")
plt.legend()

plt.savefig("plots/ElectricityLossPerGrowth.png", dpi=200, pad_inches=0.25)
plt.show()