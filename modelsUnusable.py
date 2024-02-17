from darts.models import NLinearModel, NBEATSModel, ARIMA, XGBModel, LinearRegressionModel, DLinearModel
from darts.utils import timeseries_generation as tg
import pandas as pd
from model import ModelTemplate, generate_torch_kwargs


# requires a time series of length at least 22 -> cant be used
class NLinModel(ModelTemplate):
    def fit_test(self, past_lags, future_lag_tuple):
        self.output_len_test = len(self.target_test)

        if self.verbose:
            self.info("fit_test()")

        self.model_test = NLinearModel(
            input_chunk_length=len(self.target_train), # past_lags, # because it is horizon based
            output_chunk_length=self.output_len_test, # future_lag_tuple[1], # may cause errors
            n_epochs=100,
            model_name=self.name,
            save_checkpoints=True,
            force_reset=True,
            **generate_torch_kwargs(),
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

        self.model = NLinearModel(
            input_chunk_length=past_lags,
            output_chunk_length=self.output_chunk_len, # future_lag_tuple[1], # may cause errors
            n_epochs=100,
            model_name=self.name,
            save_checkpoints=True,
            force_reset=True,
            **generate_torch_kwargs(),
        )

        self.model.fit(
            self.target_full,
            future_covariates=self.future_covariates_full,
            past_covariates=self.past_covariates_full,
        )

# requires a time series of length at least 22 -> cant be used
class DLinModel(ModelTemplate):
    def fit_test(self, past_lags, future_lag_tuple):
        self.output_len_test = len(self.target_test)

        if self.verbose:
            self.info("fit_test()")

        self.model_test = DLinearModel(
            input_chunk_length=len(self.target_train), # past_lags, # not past_lags, because it is horizon based
            output_chunk_length=self.output_len_test,
            n_epochs=100,
            model_name=self.name,
            save_checkpoints=True,
            force_reset=True,
            **generate_torch_kwargs(),
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

        self.model = DLinearModel(
            input_chunk_length=past_lags,
            output_chunk_length=self.output_chunk_len,
            n_epochs=100,
            model_name=self.name,
            save_checkpoints=True,
            force_reset=True,
            **generate_torch_kwargs(),
        )

        self.model.fit(
            self.target_full,
            future_covariates=self.future_covariates_full,
            past_covariates=self.past_covariates_full,
        )

# requires LCOE to be known for the time steps of the target series -> can hardly be used because target series (and learning data) would have to be shortened immensely
class ARIMAModel(ModelTemplate):
    def fit_test(self, past_lags, future_lag_tuple):
        self.output_len_test = len(self.target_test)
        self.pass_future_covars_testpredict = True

        if self.verbose:
            self.info("fit_test()")

        self.model_test = ARIMA(
            p=past_lags,
            d=1,
            trend="t"
        )

        self.model_test.fit(
            self.target_train,
            future_covariates=self.future_covariates_full,
        )

    
    def fit(self, output_len, past_lags, future_lag_tuple, output_chunk_len=None):
        self.output_len = output_len
        self.output_chunk_len = output_chunk_len or self.output_len
        self.pass_future_covars_predict = True

        if self.verbose:
            self.info("fit()")

        self.model = ARIMA(
            p=past_lags,
            d=1,
            trend="t"
        )
        
        self.model.fit(
            self.target_full,
            future_covariates=self.future_covariates_full,
        )