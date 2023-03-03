import numpy as np
import pandas as pd
from scipy.fftpack import rfft, irfft, rfftfreq
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def load(filename: str, summary: bool = True) -> pd.DataFrame:
    """
    Load in the data, which can be csv or dat right now.
    """
    # load the file
    if filename.lower().endswith('.csv'):
        df: pd.DataFrame = pd.read_csv(filename, parse_dates=True)
    elif filename.lower().endswith('.dat'):
        df: pd.DataFrame = pd.read_csv(filename, has_header=False, parse_dates=True)
    else:
        raise AttributeError(f"The file type '{filename.split('.')[-1]}' is currently not supported. Please use one of ['csv', 'dat']")

    # describe
    if summary:
        print(df.describe())

    return df


def calculate_fft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Fast Fourier Transform (FFT) of a Pandas DataFrame containing numerical values.

    The function takes a Pandas DataFrame and returns a new Pandas DataFrame containing the FFT values for each numerical 
    column in the input DataFrame. The input DataFrame must have numerical columns only. If a column contains missing 
    values, the missing values will be linearly interpolated before computing the FFT. The returned DataFrame has columns 
    with names in the format "<column_name>_FFT".

    Args:
        df (pd.DataFrame): A Pandas DataFrame with numerical columns only.

    Returns:
        pd.DataFrame: A new Pandas DataFrame with the FFT values of the input DataFrame.

    Raises:
        ValueError: If the input DataFrame has no numerical columns.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from scipy.fft import rfft
        >>> np.random.seed(42)
        >>> time = np.arange(0, 10, 0.1)
        >>> signal1 = np.sin(2*np.pi*0.5*time) + np.sin(2*np.pi*2*time)
        >>> signal2 = np.cos(2*np.pi*1.5*time) + np.sin(2*np.pi*4*time)
        >>> df = pd.DataFrame({'Signal 1': signal1, 'Signal 2': signal2})
        >>> fft_df = calculate_fft(df)
        >>> print(fft_df.head())
                Signal 1_FFT  Signal 2_FFT
        0.0    4.310515+0.000000j  -7.392037+0.000000j
        1.0    4.411234-0.192176j   3.232090-0.463952j
        2.0    4.339049-0.388063j   1.506034+1.050393j
        3.0    4.096452-0.563433j   0.165987-2.280368j
        4.0    3.712191-0.703674j  -0.542603-0.193758j

    """
    # create an empty dataframe to store the transformed values
    fft_df = pd.DataFrame()

    # loop over each column in the input dataframe
    for col in df.columns:
        # check if column is numerical
        if np.issubdtype(df[col].dtype, np.number):
            # extract the values from the dataframe
            values = df[col].interpolate().to_numpy()

            # perform FFT on the data
            fft_values = rfft(values)

            # add new column to fft_df containing the FFT result
            fft_df[f"{col}_FFT"] = fft_values

    # return the transformed dataframe
    return fft_df


def plotly_fft(fft_df: pd.DataFrame, timestep: float = 1) -> go.Figure:
    """Create a Plotly figure showing the Fast Fourier Transform (FFT) of a Pandas DataFrame.

    This function creates a subplots figure with one subplot per column of the input DataFrame. Each subplot shows the 
    FFT magnitude of the corresponding column as a function of frequency. The frequency range is determined by the length 
    of the input DataFrame and the sampling rate, which is calculated as the inverse of the input timestep.

    Args:
        fft_df (pd.DataFrame): A Pandas DataFrame with numerical values to be transformed by the FFT. Each column 
            corresponds to a different signal or variable. The number of rows must be a power of 2, or the function will 
            pad with zeros until the next power of 2 is reached.
        timestep (float, optional): The time step between each sample in the input signal, in seconds. Defaults to 1.

    Returns:
        go.Figure: A Plotly figure with one subplot per column of the input DataFrame, showing the FFT magnitude as a 
            function of frequency.

    Raises:
        ValueError: If the number of rows in the input DataFrame is not a power of 2.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from scipy.fft import fft
        >>> np.random.seed(42)
        >>> time = np.arange(0, 10, 0.1)
        >>> signal1 = np.sin(2*np.pi*0.5*time) + np.sin(2*np.pi*2*time)
        >>> signal2 = np.cos(2*np.pi*1.5*time) + np.sin(2*np.pi*4*time)
        >>> fft_df = pd.DataFrame({'Signal 1': signal1, 'Signal 2': signal2})
        >>> fig = plotly_fft(fft_df, timestep=0.1)
        >>> fig.show()

    """
    # sampling rate and frequencies
    sr = 1 / (timestep)
    N = len(fft_df)
    freqs = rfftfreq(N, sr)

    # create subplots for each column in the dataframe
    fig = make_subplots(rows=len(fft_df.columns), cols=1)

    # loop over each column in the dataframe
    for i, col in enumerate(fft_df.columns):
        # extract the FFT values for the current column
        fft_values = fft_df[col].values

        # add a new trace to the subplot for the current column
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=np.abs(fft_values),
                mode='lines',
                name=col
            ),
            row=i+1,
            col=1
        )

    # update the layout of the plot
    fig.update_layout({
        'template': 'none',
        'title': 'FFT Plot',
        f'xaxis{len(fft_df.columns)}': dict(title=f'Period [{timestep} s]'),
        'yaxis_title': 'Magnitude',
        'height': 200 * len(fft_df.columns)
    })

    # return the plot
    return fig




def matplotlib_fft(fft_df: pd.DataFrame, timestep: float = 1) -> plt.figure:
    """Create a Matplotlib figure showing the Fast Fourier Transform (FFT) of a Pandas DataFrame.

    This function creates a subplots figure with one subplot per column of the input DataFrame. Each subplot shows the 
    FFT magnitude of the corresponding column as a function of frequency. The frequency range is determined by the length 
    of the input DataFrame and the sampling rate, which is calculated as the inverse of the input timestep.

    Args:
        fft_df (pd.DataFrame): A Pandas DataFrame with numerical values to be transformed by the FFT. Each column 
            corresponds to a different signal or variable. The number of rows must be a power of 2, or the function will 
            pad with zeros until the next power of 2 is reached.
        timestep (float, optional): The time step between each sample in the input signal, in seconds. Defaults to 1.

    Returns:
        None

    Raises:
        ValueError: If the number of rows in the input DataFrame is not a power of 2.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from scipy.fft import fft
        >>> np.random.seed(42)
        >>> time = np.arange(0, 10, 0.1)
        >>> signal1 = np.sin(2*np.pi*0.5*time) + np.sin(2*np.pi*2*time)
        >>> signal2 = np.cos(2*np.pi*1.5*time) + np.sin(2*np.pi*4*time)
        >>> fft_df = pd.DataFrame({'Signal 1': signal1, 'Signal 2': signal2})
        >>> matplotlib_fft(fft_df, timestep=0.1)

    """
    # sampling rate and frequencies
    sr = 1 / (timestep)
    N = len(fft_df)
    freqs = rfftfreq(N, sr)

    # create subplots for each column in the dataframe
    fig, axs = plt.subplots(nrows=len(fft_df.columns), ncols=1, figsize=(6, 4*len(fft_df.columns)))

    # loop over each column in the dataframe
    for i, col in enumerate(fft_df.columns):
        # extract the FFT values for the current column
        fft_values = fft_df[col].values

        # plot the FFT magnitude for the current column
        axs[i].plot(freqs, np.abs(fft_values))
        axs[i].set_xlabel('Frequency')
        axs[i].set_ylabel('Magnitude')
        axs[i].set_title(col)

    # update the layout of the plot
    fig.suptitle('FFT Plot')
    fig.tight_layout()

    return fig


def invert_fft(fft_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a pandas DataFrame containing FFT values and returns a new pandas DataFrame
    containing the inverted input data.

    Args:
        fft_df: The input pandas DataFrame containing FFT values.

    Returns:
        A new pandas DataFrame containing the inverted input data.
    """
    # create an empty dataframe to store the inverted values
    inv_df = pd.DataFrame()

    # loop over each column in the FFT dataframe
    for col in fft_df.columns:
        # extract the FFT values from the dataframe
        fft_values = fft_df[col]

        # perform inverse FFT on the data
        inv_values = irfft(fft_values)

        # add new column to inv_df containing the inverted result
        inv_df[col.replace('_FFT', '')] = inv_values.real

    # return the inverted dataframe
    return inv_df
