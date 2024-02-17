import numpy as np
from darts.utils import timeseries_generation as tg
from model import MeanModel, BeatsModel, XGBoostModel, RegModel, EnsembleBeatsModel
from darts.metrics import rmse
from helper import loadDfsAsDict, normalizeSeriesOfDictDfs
import pandas as pd
from matplotlib import pyplot as plt
import os


past_covariates_full = loadDfsAsDict([
    "Anschaffungskosten",
    "Einspeiseverguetung",
    "Investitionen",
    "LCOEHistorisch",
    "PolysilikonPreis"
])

future_covariates_full = loadDfsAsDict([
    "LCOEGesamt"
])

past_covariates_partial_length = past_covariates_full.copy()
past_covariates_partial_length.pop("Anschaffungskosten")
past_covariates_partial_length.pop("LCOEHistorisch")

future_covariates_partial_length = future_covariates_full.copy()

past_covars_to_normalize = list(past_covariates_full.keys())
past_covariates_full = normalizeSeriesOfDictDfs(
    past_covariates_full, 
    past_covars_to_normalize,
    [covar if not covar.startswith("LCOE") else "LCOE" for covar in past_covars_to_normalize]
)

future_covars_to_normalize = list(future_covariates_full.keys())
future_covariates_full = normalizeSeriesOfDictDfs(
    future_covariates_full, 
    future_covars_to_normalize,
    [covar if not covar.startswith("LCOE") else "LCOE" for covar in future_covars_to_normalize]
)

target = pd.read_csv("data/Erzeugung.csv", sep=",", header=0)

TARGET_COLUMN="Erzeugung"

# Different test train splits so that test length is the same, to make results more accurate
TEST_TRAIN_RATIO=0.8
TEST_TRAIN_RATIO_FULL_LENGTH = 0.85

def selected_past_covars():
    covars = list(past_covariates_full.keys())
    covars.remove("Einspeisevergütung")
    covars.remove("Investitionen")
    return covars

models = [
    MeanModel("Mittelwert-Modell", target, list(past_covariates_partial_length.values()), list(future_covariates_full.values()), TEST_TRAIN_RATIO_FULL_LENGTH),
] + [
    RegModel("Lineare Regression", target, list(past_covariates_partial_length.values()), list(future_covariates_partial_length.values()), TEST_TRAIN_RATIO_FULL_LENGTH),
    XGBoostModel("XGBoost", target, list(past_covariates_partial_length.values()), list(future_covariates_partial_length.values()), TEST_TRAIN_RATIO_FULL_LENGTH),
    EnsembleBeatsModel(20, "N-Beats", target, list(past_covariates_partial_length.values()), list(future_covariates_partial_length.values()), TEST_TRAIN_RATIO_FULL_LENGTH),
    # EnsembleBeatsModel may not be at index 0 - does not have all ModelTemplate properties
]



def measureModel(m, prediction):
    return rmse(m.target_test, prediction)

def getColors():
    return [
        (0.75, 0.2, 0.2),
        (0.45, 0.15, 0.45),
        (0.2, 0.7, 0.2),
        (0.2, 0.2, 0.75),
        (0.25, 0.5, 0.5),
        (0.5, 0.375, 0.375),
        (0.5, 0.375, 0.25),
        (0.125, 0.125, 0.125)
    ]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "8"

if __name__ == "__main__":
    [m.fit_test(4, (0,4)) for m in models]
    [m.fit(10, 5, (0,5), output_chunk_len=10) for m in models]
    testPredictions = [m.predict_test() for m in models]
    predictions = [m.predict() for m in models]
    testDfs = [p.pd_series() for p in testPredictions]
    predictionDfs = [p.pd_series()for p in predictions]

    eval_scores = [(m.name, measureModel(m, testPredictions[i])) for i, m in enumerate(models)]

    colors = getColors()
    colorsSecondary = [(c[0]*1.4, c[1]*1.4, c[2]*1.4) for c in colors]

    colors = [list(c) for c in colors]
    colorsSecondary = [list(c) for c in colorsSecondary]

    for i, s in enumerate(eval_scores):
        print(f"{s[0]}: {s[1]}")

    plt.plot(models[0].target_full.pd_series(), label="Tatsächlich")
    for model, testpred, pred, c in zip(models, testDfs, predictionDfs, colors):
        last_train_val = models[0].target_train.values()[-1][0]
        last_train_index = models[0].target_train.end_time()
        last_full_train_val = models[0].target_full.values()[-1][0]
        last_full_train_index = models[0].target_full.end_time()

        model.info(f"Test predicition: \n{testpred}\n\n")
        model.info(f"Predicition: \n{pred}\n\n")

        testpred = pd.concat([pd.Series([last_train_val], index=[last_train_index]), testpred])
        pred = pd.concat([pd.Series([last_full_train_val], index=[last_full_train_index]), pred])

        plt.plot(testpred, label=model.name, color=c)
        plt.plot(pred, color=c)
    
    plt.xlabel("Jahr")
    plt.ylabel("Bruttoelektrizitätsgewinnung (TWh)")
    plt.legend()

    if not os.path.exists("plots"):
        os.mkdir("plots")

    plt.savefig("plots/Plot_Total.jpg", dpi=200, pad_inches=0.25)
    plt.clf()

    for model, testpred, pred, c in zip(models, testDfs, predictionDfs, colors):
        last_train_val = models[0].target_train.values()[-1][0]
        last_train_index = models[0].target_train.end_time()
        last_full_train_val = models[0].target_full.values()[-1][0]
        last_full_train_index = models[0].target_full.end_time()

        testpred = pd.concat([pd.Series([last_train_val], index=[last_train_index]), testpred])
        pred = pd.concat([pd.Series([last_full_train_val], index=[last_full_train_index]), pred])

        plt.plot(model.target_full.pd_series(), label="Tatsächlich")
        plt.plot(testpred, label=model.name, color=c)
        plt.plot(pred, color=c)
        plt.xlabel("Jahr")
        plt.ylabel("Bruttoelektrizitätsgewinnung (TWh)")
        plt.legend()
        plt.savefig(f"plots/Plot_{model.name}.jpg", bbox_inches="tight", dpi=200)
        plt.clf()