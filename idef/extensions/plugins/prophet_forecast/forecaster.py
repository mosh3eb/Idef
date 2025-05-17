"""
Time Series Forecasting Plugin using Facebook Prophet.
"""

from typing import Any, Dict
import pandas as pd
from prophet import Prophet

from ...plugins import AnalysisMethodPlugin, PluginMetadata

class ProphetForecaster(AnalysisMethodPlugin):
    """Prophet forecasting plugin."""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self._model = None
        self._params = {}
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the forecaster."""
        try:
            # Store parameters
            self._params = {
                'periods': kwargs.get('periods', 30),
                'freq': kwargs.get('freq', 'D'),
                'seasonality_mode': kwargs.get('seasonality_mode', 'additive'),
                'changepoint_prior_scale': kwargs.get('changepoint_prior_scale', 0.05)
            }
            return True
        except Exception as e:
            return False
    
    def cleanup(self):
        """Clean up resources."""
        self._model = None
        self._params = {}
    
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform time series forecasting."""
        # Validate input data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        # Get column names
        ds_col = kwargs.get('date_col', 'ds')
        y_col = kwargs.get('value_col', 'y')
        
        if ds_col not in data.columns or y_col not in data.columns:
            raise ValueError(f"Data must contain '{ds_col}' and '{y_col}' columns")
        
        # Prepare data for Prophet
        df = data.rename(columns={ds_col: 'ds', y_col: 'y'})
        
        # Create and fit model
        self._model = Prophet(
            seasonality_mode=self._params['seasonality_mode'],
            changepoint_prior_scale=self._params['changepoint_prior_scale']
        )
        
        # Add additional regressors if specified
        regressors = kwargs.get('regressors', [])
        for regressor in regressors:
            if regressor in data.columns:
                self._model.add_regressor(regressor)
        
        self._model.fit(df)
        
        # Create future dataframe
        future = self._model.make_future_dataframe(
            periods=self._params['periods'],
            freq=self._params['freq']
        )
        
        # Add regressor values to future if needed
        for regressor in regressors:
            if regressor in data.columns:
                # Simple forward fill for demo purposes
                future[regressor] = data[regressor].iloc[-1]
        
        # Make forecast
        forecast = self._model.predict(future)
        
        # Prepare results
        results = {
            'forecast': forecast,
            'model': self._model,
            'metrics': {
                'mse': ((df['y'] - forecast['yhat'][:len(df)])** 2).mean(),
                'rmse': ((df['y'] - forecast['yhat'][:len(df)])** 2).mean() ** 0.5
            },
            'components': {
                'trend': forecast['trend'],
                'seasonal': forecast['seasonal'] if 'seasonal' in forecast else None,
                'weekly': forecast['weekly'] if 'weekly' in forecast else None,
                'yearly': forecast['yearly'] if 'yearly' in forecast else None
            }
        }
        
        return results
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the forecasting parameters."""
        return {
            'periods': {
                'type': 'integer',
                'description': 'Number of periods to forecast',
                'default': 30
            },
            'freq': {
                'type': 'string',
                'description': 'Frequency of the time series',
                'default': 'D'
            },
            'seasonality_mode': {
                'type': 'string',
                'description': 'Seasonality mode',
                'options': ['additive', 'multiplicative'],
                'default': 'additive'
            },
            'changepoint_prior_scale': {
                'type': 'float',
                'description': 'Flexibility of the trend',
                'default': 0.05
            },
            'regressors': {
                'type': 'list',
                'description': 'Additional regressors to include',
                'default': []
            }
        } 