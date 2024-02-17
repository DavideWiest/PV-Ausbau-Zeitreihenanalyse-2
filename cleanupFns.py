import re
from io import StringIO
import pandas as pd
import json
import math
from helper import (
    ConvertUSDToEUR,
    dfAdjustForInflation,
    mix_exponent,
    AdjustForInflation
)

def cleanupEinspeiseverguetungBase(s, sepIsWhitespaceToo=False, rowLimit=6):
    for i in range(1,20):
        s = s.replace("*" * i + ")", "")
        s = s.replace("*" * i, "")

    s = re.sub("[\(\[].*?[\)\]]", "", s)
    
    s = s.replace("EEG, ", "EEG,").replace("EEG,", "EEG-")
    s = s.replace(".", "\t").replace(",", ".").replace("\t", ",")
    if sepIsWhitespaceToo:
        s = s.replace(" ", ",")

    slist = s.split("\n")

    for i, line in enumerate(slist):
        slist[i] = ",".join([v.strip() for v in line.split(",")[:rowLimit]])

    return "\n".join(slist)

def cleanupEinspeiseverguetungPart1(s):
    s = cleanupEinspeiseverguetungBase(s)
    df = pd.read_csv(StringIO(s), sep=",", header=None)
    df.columns = ["Jahr", "EV1", "EV2", "EV3", "EV4", "EV5"]

    df = df.loc[:, ["Jahr", "EV2"]]
    df = df.rename({"EV2": "Einspeiseverguetung"}, axis="columns")
    df = dfAdjustForInflation(df, "Einspeiseverguetung")

    return df

def cleanupEinspeiseverguetungPart2(s):
    s = cleanupEinspeiseverguetungBase(s)
    df = pd.read_csv(StringIO(s), sep=",", header=None)
    df.columns = ["Tag", "Monat", "Jahr", "EV1", "EV2", "EV3"]

    df = df.loc[:, ["Jahr", "EV2"]]
    df = df.rename({"EV2": "Einspeiseverguetung"}, axis="columns")

    # Durchschnitt berechnen
    agg_options = {"Einspeiseverguetung": "mean"}
    df = df.groupby(df["Jahr"]).aggregate(agg_options)
    df = df.reset_index()

    df = dfAdjustForInflation(df, "Einspeiseverguetung")

    return df

def cleanupEinspeiseverguetungPart3(s):
    s = cleanupEinspeiseverguetungBase(s, True, 8)
    df = pd.read_csv(StringIO(s), sep=",", header=None)
    df.columns = ["Ab", "Tag", "Leer", "Monat", "Jahr", "EV1", "EV2", "EV3"]

    df = df.drop(df[df["Jahr"] <= 2020].index)

    df = df.loc[:, ["Jahr", "EV2"]]
    df = df.rename({"EV2": "Einspeiseverguetung"}, axis="columns")

    # Durchschnitt berechnen
    agg_options = {"Einspeiseverguetung": "mean"}
    df = df.groupby(df["Jahr"]).aggregate(agg_options)
    df = df.reset_index()

    df = dfAdjustForInflation(df, "Einspeiseverguetung")

    return df


def cleanupAnschaffungskosten(s):
    vals = s.replace("\t", " ").split(" ")
    startYear = 2010
    l = "\n".join(
        ["Jahr,Anschaffungskosten"] +
        [f"{year},{ConvertUSDToEUR(float(v))}" for v, year in zip(vals, range(startYear, startYear+len(vals)))]
    )

    return l

def cleanupElektrizitaetserzeugung(s):
    s = s.replace(",", ".").replace("\t", " ")
    vals = s.split(" ")
    startYear = 2000
    return "\n".join(
        ["Jahr,Erzeugung"] +
        [f"{year}," + v for v, year in zip(vals, range(startYear, startYear+len(vals)))]
    )

def cleanupInvestitionen(s):
    s = s.replace(",", ";").replace(".", "")
    df = pd.read_csv(StringIO(s), sep=";", header=0)

    # drop all columns except
    df = df.loc[:, ["Jahr", "Photovoltaik"]]

    df = df.rename({"Photovoltaik": "Investitionen"}, axis="columns")

    return df

def cleanupLCOE(s):
    s = s.replace(",", ".")
    df = pd.read_csv(StringIO(s), sep="\t", header=None)
    df.columns = ["Jahr", "LCOE"]

    df["LCOE"] = df["LCOE"].apply(ConvertUSDToEUR)

    return df

def cleanupLCOEPrognose(s):
    data = json.loads(s)
    start = data["start"]
    years = data["developmentRange"]
    upper_bound = data["upperBoundDevelopment"]
    lower_bound = data["lowerBoundDevelopment"]
    
    mix_exp = mix_exponent(start, upper_bound, lower_bound, years)

    result = []
    val = start
    result.append(f"{years[0]},{start}")
    for i in range(years[0]+1, years[1]+1):
        val = start * math.exp(mix_exp * (i-years[0]))
        result.append(f"{i},{val}")
    
    return "\n".join(["Jahr,LCOE"] + result)

def cleanupPolysiliconPrices(s):
    df = pd.read_csv(StringIO(s), sep="\t", header=None)
    df.columns = ["Jahr", "PolysilikonPreis"]

    df2 = pd.read_csv("dataRaw/Polysilicon_USD2008_KG.csv", sep=",", header=0)

    df["PolysilikonPreis"] = df["PolysilikonPreis"].apply(ConvertUSDToEUR)
    overlapYear = 2003 # last year in the dataframe for the old values for adjusting them to fit the trend of the newer data

    mul = df2.loc[df["Jahr"] == overlapYear]["PolysilikonPreis"] / df2.loc[df["Jahr"] == overlapYear]["PolysilikonPreis"]

    df2["PolysilikonPreis"] = df2["PolysilikonPreis"].apply(lambda x: x*mul)

    currencyMul = AdjustForInflation(ConvertUSDToEUR(1), 2004, 2023)
    df2["PolysilikonPreis"] = df2["PolysilikonPreis"].apply(lambda x: x*currencyMul)

    df2 = df2.loc[df2["Jahr"] != overlapYear]

    df = pd.concat([df2, df])

    return df

