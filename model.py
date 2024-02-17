import numpy as np
from darts.models import NBEATSModel, ARIMA, XGBModel, LinearRegressionModel
from darts.utils import timeseries_generation as tg
import pandas as pd
from sklearn.model_selection import train_test_split
from darts.models.forecasting.baselines import NaiveMean
from darts.utils.callbacks import TFMProgressBar


class ModelTemplate():
    def __init__(self, name, target_df, past_co_dfs, future_co_dfs, train_test_ratio, verbose=True):
        self.name = name
        self.verbose = verbose
        self.train_test_ratio = train_test_ratio
        self.pass_future_covars_predict = False
        self.pass_future_covars_testpredict = False

        self.mergedBeginning = getMergedBeginning([target_df] + past_co_dfs)
        self.mergedEnding = getMergedEnding([target_df] + past_co_dfs)

        target_df = self.sanitize(target_df)
        past_co_dfs = [self.sanitize(df) for df in past_co_dfs]

        if verbose:
            self.info(f"Merged time series start {self.mergedBeginning} End {self.mergedEnding}")

        interlaced_past_covariates_split = [train_test_split(df, test_size=1-self.train_test_ratio, shuffle=False) for df in past_co_dfs]
        past_co_dfs_train, past_co_dfs_test = (
            [s[0] for s in interlaced_past_covariates_split],
            [s[1] for s in interlaced_past_covariates_split]
        )
        target_train, target_test = train_test_split(target_df, test_size=1-self.train_test_ratio, shuffle=False)
        
        if verbose:
            self.info(f"Train length {len(target_train)} Test length {len(target_test)}")

        self.past_covariates_test = past_co_dfs_test
        self.target_test = target_test

        self.target_full = tg.TimeSeries.from_dataframe(
            target_df, 
            "Jahr", 
            value_cols=[c for c in list(target_df.columns) if c != "Jahr"], 
            fill_missing_dates=True
        )
        self.target_train = tg.TimeSeries.from_dataframe(
            target_train, 
            "Jahr", 
            value_cols=[c for c in list(target_df.columns) if c != "Jahr"], 
            fill_missing_dates=True
        )
        self.target_test = tg.TimeSeries.from_dataframe(
            target_test, 
            "Jahr", 
            value_cols=[c for c in list(target_df.columns) if c != "Jahr"], 
            fill_missing_dates=True
        )

        self.input_len = len(target_df.index)
        self.input_len_train = len(self.target_train)
        self.output_len_test = len(self.target_test)

        saniziedDfPastFull = mergeDfs(past_co_dfs)
        sanitizedDfPastTrain = mergeDfs(past_co_dfs_train)
        sanitizedDfPastTest = mergeDfs(past_co_dfs_test)
        sanitizedDfFuture = mergeDfs([df.reset_index(drop=True) for df in future_co_dfs])

        if saniziedDfPastFull is not None:
            self.past_covariates_full = self.dfToTimeseries(saniziedDfPastFull)
        else:
            self.info("Warning: saniziedDfPastFull is None")
        
        if sanitizedDfPastTrain is not None:
            self.past_covariates_train = self.dfToTimeseries(sanitizedDfPastTrain)
        else:
            self.info("Warning: sanitizedDfPastTrain is None")

        if sanitizedDfPastTest is not None:
            self.past_covariates_test = self.dfToTimeseries(sanitizedDfPastTest)
        else:
            self.info("Warning: sanitizedDfPastTest is None")

        if sanitizedDfFuture is not None:
            self.future_covariates_full = self.dfToTimeseries(sanitizedDfFuture)
        else:
            self.info("Warning: sanitizedDfFuture is None")
    
    def dfToTimeseries(self, df):
        return tg.TimeSeries.from_dataframe(
                df, 
                "Jahr", 
                value_cols=[c for c in list(df.columns) if c != "Jahr"], 
                fill_missing_dates=True
            )

    def sanitize(self, df):
        df = timeconstraint(df, self.mergedBeginning, self.mergedEnding)
        df = df.reset_index(drop=True)
        return df
    
    def predict_test(self):
        if self.verbose:
            self.info("predict_test()")
        if self.pass_future_covars_testpredict == True:
            return self.model_test.predict(
                self.output_len_test,
                future_covariates=self.future_covariates_full
            )
        return self.model_test.predict(
            self.output_len_test
        )

    def predict(self):
        if self.verbose:
            self.info("predict()")
        if self.pass_future_covars_predict == True:
            return self.model.predict(
                self.output_len,
                future_covariates=self.future_covariates_full
            )
        return self.model.predict(
                self.output_len,
            )

    def info(self, msg):
        print(f"{self.name}: " + msg)


class MeanModel(ModelTemplate):

    def fit_test(self, *args, **kwargs):
        self.output_len_test = len(self.target_test)

        self.model_test = NaiveMean()
        self.model_test.fit(self.target_train)
    
    def fit(self, output_len, *args, **kwargs):
        self.output_len = output_len

        self.model = NaiveMean()
        self.model.fit(self.target_full)

class BeatsModel(ModelTemplate):
    def fit_test(self, past_lags, future_lag_tuple):
        self.output_len_test = len(self.target_test)

        if self.verbose:
            self.info("fit_test()")

        self.model_test = NBEATSModel(
            input_chunk_length=past_lags,
            output_chunk_length=self.output_len_test, # future_lag_tuple[1], # may cause errors
            generic_architecture=True,
            num_stacks=2,
            num_blocks=1,
            num_layers=2,
            layer_widths=64,
            n_epochs=200,
            model_name=self.name,
            save_checkpoints=True,
            force_reset=True,
            **generate_torch_kwargs(),
        )

        self.model_test.fit(
            self.target_train,
            # future_covariates=self.future_covariates_full,
            past_covariates=self.past_covariates_train,
        )

    
    def fit(self, output_len, past_lags, future_lag_tuple, output_chunk_len=None):
        self.output_len = output_len
        self.output_chunk_len = output_chunk_len or self.output_len

        if self.verbose:
            self.info("fit()")

        self.model = NBEATSModel(
            input_chunk_length=past_lags,
            output_chunk_length=self.output_chunk_len, # future_lag_tuple[1], # may cause errors
            generic_architecture=True,
            num_stacks=2,
            num_blocks=1,
            num_layers=2,
            layer_widths=64,
            n_epochs=200,
            model_name=self.name,
            save_checkpoints=True,
            force_reset=True,
            **generate_torch_kwargs(),
        )

        self.model.fit(
            self.target_full,
            past_covariates=self.past_covariates_full,
        )

class EnsembleBeatsModel():
    def __init__(self, n, name, target_df, past_co_dfs, future_co_dfs, train_test_ratio, verbose=True):
        self.name = name
        self.n = n
        self.models = [
            BeatsModel(name + f" ({i})", target_df, past_co_dfs, future_co_dfs, train_test_ratio, verbose) for i in range(n)
        ]
        self.target_test = self.models[0].target_test
        self.target_full = self.models[0].target_full

    def fit_test(self, past_lags, future_lag_tuple):
        for model in self.models:
            model.fit_test(past_lags, future_lag_tuple)
    
    def fit(self, output_len, past_lags, future_lag_tuple, output_chunk_len=None):
        for model in self.models:
            model.fit(output_len, past_lags, future_lag_tuple, output_chunk_len)

    def predict_test(self):
        predictions = [model.predict_test() for model in self.models]
        return self.mergePredictions(predictions)

    def predict(self):
        predictions = [model.predict() for model in self.models]
        return self.mergePredictions(predictions)
    
    def mergePredictions(self, predictions: list[tg.TimeSeries]):
        prediction = predictions[0].pd_series()
        for nextPred in predictions[1:]:
            prediction = prediction.add(nextPred.pd_series(), fill_value=0)
        prediction = prediction / self.n
        return tg.TimeSeries.from_series(
            prediction, 
            fill_missing_dates=True
        )

    def info(self, msg):
        print(f"{self.name}: " + msg)

class RegModel(ModelTemplate):
    def fit_test(self, past_lags, future_lag_tuple):
        self.output_len_test = len(self.target_test)

        if self.verbose:
            self.info("fit_test()")

        self.model_test = LinearRegressionModel(
            lags=past_lags,
            lags_past_covariates=past_lags,
            # lags_future_covariates=future_lag_tuple,
            output_chunk_length=self.output_len_test
        )

        self.model_test.fit(
            self.target_train,
            # future_covariates=self.future_covariates_full,
            past_covariates=self.past_covariates_train,
        )

    
    def fit(self, output_len, past_lags, future_lag_tuple, output_chunk_len=None):
        self.output_len = output_len
        self.output_chunk_len = output_chunk_len or self.output_len

        if self.verbose:
            self.info("fit()")

        self.model = LinearRegressionModel(
            lags=past_lags,
            lags_past_covariates=past_lags,
            lags_future_covariates=future_lag_tuple,
            output_chunk_length=self.output_chunk_len
        )
        
        self.model.fit(
            self.target_full,
            future_covariates=self.future_covariates_full,
            past_covariates=self.past_covariates_full
        )

class RegModelTargetOnly(ModelTemplate):
    def fit_test(self, past_lags, future_lag_tuple):
        self.output_len_test = len(self.target_test)

        if self.verbose:
            self.info("fit_test()")

        self.model_test = LinearRegressionModel(
            lags=past_lags,
            output_chunk_length=self.output_len_test
        )

        self.model_test.fit(
            self.target_train,
        )

    
    def fit(self, output_len, past_lags, future_lag_tuple, output_chunk_len=None):
        self.output_len = output_len
        self.output_chunk_len = output_chunk_len or self.output_len

        if self.verbose:
            self.info("fit()")

        self.model = LinearRegressionModel(
            lags=past_lags,
            output_chunk_length=self.output_chunk_len
        )
        
        self.model.fit(
            self.target_full,
        )

class XGBoostModel(ModelTemplate):
    def fit_test(self, past_lags, future_lag_tuple):
        self.output_len_test = len(self.target_test)

        if self.verbose:
            self.info("fit_test()")

        self.model_test = XGBModel(
            lags=past_lags,
            lags_past_covariates=past_lags,
            lags_future_covariates=future_lag_tuple,
            output_chunk_length=self.output_len_test
        )

        self.model_test.fit(
            self.target_train,
            future_covariates=self.future_covariates_full,
            past_covariates=self.past_covariates_train,
        )

    
    def fit(self, output_len, past_lags, future_lag_tuple, output_chunk_len=None):
        self.output_len = output_len
        self.output_chunk_len = output_chunk_len or self.output_len

        if self.verbose:
            self.info("fit()")

        self.model = XGBModel(
            lags=past_lags,
            lags_past_covariates=past_lags,
            lags_future_covariates=future_lag_tuple,
            output_chunk_length=self.output_chunk_len
        )

        self.model.fit(
            self.target_full,
            future_covariates=self.future_covariates_full,
            past_covariates=self.past_covariates_full,
        )

class XGBoostModelTargetOnly(ModelTemplate):
    def fit_test(self, past_lags, future_lag_tuple):
        self.output_len_test = len(self.target_test)

        if self.verbose:
            self.info("fit_test()")

        self.model_test = XGBModel(
            lags=past_lags,
            output_chunk_length=self.output_len_test
        )

        self.model_test.fit(
            self.target_train,
        )

    
    def fit(self, output_len, past_lags, future_lag_tuple, output_chunk_len=None):
        self.output_len = output_len
        self.output_chunk_len = output_chunk_len or self.output_len

        if self.verbose:
            self.info("fit()")

        self.model = XGBModel(
            lags=past_lags,
            output_chunk_length=self.output_chunk_len
        )

        self.model.fit(
            self.target_full,
        )


def getMergedBeginning(dfs):
    years_sets = [df["Jahr"][0] for df in dfs]
    common_start = max(years_sets)
    return common_start

def getMergedEnding(dfs):
    years_sets = [df["Jahr"][len(df["Jahr"])-1] for df in dfs]
    common_start = min(years_sets)
    return common_start

def timeconstraint(df, mergedBeginning, mergedEnding):
        return df[(df["Jahr"] >= mergedBeginning) & (df["Jahr"] <= mergedEnding)]

def mergeDfs(dfs):
    if dfs == []:
        return None
    merged_df = dfs[0]
    for i in range(1,len(dfs)):
        merged_df = pd.merge(merged_df, dfs[i], on="Jahr")
    return merged_df

def generate_torch_kwargs():
    # run torch models on CPU, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }