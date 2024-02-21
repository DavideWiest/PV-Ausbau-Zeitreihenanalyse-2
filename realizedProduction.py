import pandas as pd
from io import StringIO

string = open("dataRaw/Realisierte_Erzeugung_201701010000_202312242359_Stunde.csv", "r", encoding="utf-8").read()
string = string.replace(".", "").replace(",", ".").replace("-", "0")

dfProduction = pd.read_csv(StringIO(string), delimiter=";", header=0)
dfProduction["Zeit"] = dfProduction["Datum"].astype(str)
 
dfProduction = dfProduction.where(dfProduction["Zeit"].str.contains("2022"))

print("Realisierte Erzeugung aus PV 2022 in TWh: " + str(dfProduction["Photovoltaik [MWh] Berechnete Auflösungen"].sum() / 1_000_000))
