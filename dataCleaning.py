from cleanupFns import (
    cleanupAnschaffungskosten,
    cleanupEinspeiseverguetungPart1, 
    cleanupEinspeiseverguetungPart2,
    cleanupEinspeiseverguetungPart3,
    cleanupElektrizitaetserzeugung,
    cleanupInvestitionen,
    cleanupLCOEPrognose,
    cleanupLCOE,
    cleanupPolysiliconPrices
)
from helper import mergeCSVFilesWithHeader
import pandas as pd


def cleanupRawData(inputFileName, cleanupFn, outputFileName, saveFn):
    with open("dataRaw/"+inputFileName, "r", encoding="utf-8") as f:
        contents = f.read()

    saveFn(outputFileName, cleanupFn(contents))

def saveStringToFile(outputFileName, output):
    with open("data/"+outputFileName, "w", encoding="utf-8") as f:
        f.write(output)

def saveDfWithHeadingToFile(outputFileName, df: pd.DataFrame):
    df['Jahr'] = df['Jahr'].astype(int)
    df.to_csv("data/"+outputFileName, sep=",", header=True, index=False)


if __name__ == "__main__":
    cleanupRawData("Anschaffungskosten_USD_KW_2010_2023.csv", cleanupAnschaffungskosten, "Anschaffungskosten.csv", saveStringToFile)
    cleanupRawData("BruttoelektrizitätsgewinnungPV.csv", cleanupElektrizitaetserzeugung, "Erzeugung.csv", saveStringToFile)
    cleanupRawData("EinspeiseverguetungAb2000_EURCT_KWH.csv", cleanupEinspeiseverguetungPart1, "EinspeisevergütungBis2008.csv", saveDfWithHeadingToFile)
    cleanupRawData("EinspeiseverguetungAb2009_EURCT_KWH.csv", cleanupEinspeiseverguetungPart2, "EinspeisevergütungAb2009.csv", saveDfWithHeadingToFile)
    cleanupRawData("EinspeiseverguetungAb2021_EURCT_KWH.csv", cleanupEinspeiseverguetungPart3, "EinspeisevergütungAb2021.csv", saveDfWithHeadingToFile)
    cleanupRawData("Investitionen_EUR.csv", cleanupInvestitionen, "Investitionen.csv", saveDfWithHeadingToFile)
    cleanupRawData("LCOE_Pred_EUR_KWH.json", cleanupLCOEPrognose, "LCOEPrognose.csv", saveStringToFile)
    cleanupRawData("LCOE_USD_KWH.csv", cleanupLCOE, "LCOEHistorisch.csv", saveDfWithHeadingToFile)
    cleanupRawData("Polysilicon_USD_KG.csv", cleanupPolysiliconPrices, "PolysilikonPreis.csv", saveDfWithHeadingToFile)

    mergeCSVFilesWithHeader(["data/EinspeisevergütungBis2008.csv", "data/EinspeisevergütungAb2009.csv", "data/EinspeisevergütungAb2021.csv"], "data/Einspeiseverguetung.csv", True)
    mergeCSVFilesWithHeader(["data/LCOEHistorisch.csv", "data/LCOEPrognose.csv"], "data/LCOEGesamt.csv", deleteOld=False, dropDuplicates=True) # dropping duplicate 2022. Row 2022 exists because it is needed in LCOEPrognose.csv
    