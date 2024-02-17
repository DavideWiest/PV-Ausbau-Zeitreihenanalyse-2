import math
import os
import pandas as pd

inflationdf = pd.read_csv("dataRaw/Verbraucherpreisindex.csv", sep=";", header=4)
inflationdf = inflationdf.rename({"Unnamed: 0": "Jahr"}, axis="columns")
inflationdf = inflationdf.set_index("Jahr")

def ConvertUSDToEUR(x):
    return x * 0.91

def AdjustForInflation(x, startYear, endYear=2023):
    currentYear = startYear
    while currentYear < endYear:
        inflation = float(inflationdf["Veränderung zum Vorjahr"][f"{currentYear}".replace(".0", "")].replace(",", ".")) / 100
        x *= 1+inflation
        currentYear += 1
    return x

def mergeCSVFilesWithHeader(fileNameList, resultFileName, deleteOld=False, dropDuplicates=False):
    merged_df = pd.read_csv(fileNameList[0])
    
    for filename in fileNameList[1:]:
        data_to_append = pd.read_csv(filename)
        merged_df = pd.concat([merged_df, data_to_append], ignore_index=True)
    
    if deleteOld:
        for filename in fileNameList:
            os.remove(filename)

    if merged_df["Jahr"].duplicated().any():
        if dropDuplicates:
            merged_df.drop_duplicates(subset="Jahr", keep="first", inplace=True)
        else:
            raise ValueError("Duplicates when merging")
    
    merged_df.to_csv(resultFileName, index=False)

def dfRowAdjustForInflation(row, valColname):
    row[valColname] = AdjustForInflation(row[valColname], row["Jahr"])
    return row

def dfAdjustForInflation(df, valColname):
    df = df.apply(lambda row: dfRowAdjustForInflation(row, valColname), axis=1)
    return df

def calculate_exponent(start, end, years):
    return math.log(end / start) / (years[1] - years[0])

def mix_exponent(start, upper_bound_development, lower_bound_development, years):
    d1 = start - upper_bound_development[0]
    d2 = start - lower_bound_development[0]
    dist = d1 + d2
    return calculate_exponent(upper_bound_development[0], upper_bound_development[1], years) * d1 / dist + \
        + calculate_exponent(lower_bound_development[0], lower_bound_development[1], years) * d2 / dist

def loadDfsAsDict(shortenedFileNames: list[str]):
    return { fn : pd.read_csv(f"data/{fn}.csv", sep=",", header=0) for fn in shortenedFileNames}

def normalizeSeriesValues(series: pd.Series):
    series = series - series.min() # starts at 0
    series = series / series.max() # values from 0 to 1
    return series

def normalizeSeriesOfDictDfs(dfsDict, keyList, seriesKeyList):
    for covar, seriesKey in zip(keyList, seriesKeyList):
        print(dfsDict[covar][seriesKey])
        dfsDict[covar][seriesKey] = normalizeSeriesValues(dfsDict[covar][seriesKey])
        print(dfsDict[covar][seriesKey])
    return dfsDict