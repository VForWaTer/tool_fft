import os
from datetime import datetime as dt
import pandas as pd

import lib as fftlib

from json2args import get_parameter

# parse parameters
kwargs = get_parameter()

# check if a toolname was set in env
toolname = os.environ.get('TOOL_RUN', 'fft').lower()

# switch the tool
if toolname == 'fft':
    # get the data
    df_or_filename = kwargs['data']
    if isinstance(df_or_filename, (pd.DataFrame, pd.Series)):
        df = df_or_filename
    elif isinstance(df_or_filename, str):
        df = fftlib.load(kwargs['data'])
    else:
        raise AttributeError("The data parameter could not be understood.")

    # apply the discrete fast-fourier transformation
    fft_df = fftlib.calculate_fft(df)

    # save as CSV
    fft_df.to_csv('/out/discrete_fft.csv', index=None)

    # create the plotly plots
    fig = fftlib.plotly_fft(fft_df=fft_df, timestep=kwargs.get('timestep', 1))
    fig.write_html('/out/fft_plot.html')
    fig.write_json('/out/fft_plot.plotly.json')

    # create the pdf plot
    fig = fftlib.matplotlib_fft(fft_df=fft_df, timestep=kwargs.get('timestep', 1))
    fig.savefig('/out/fft_plot.pdf', dpi=200)
    fig.savefig('/out/fft_plot.png')


# In any other case, it was not clear which tool to run
else:
    raise AttributeError(f"[{dt.now().isocalendar()}] Either no TOOL_RUN environment variable available, or '{toolname}' is not valid.\n")
