import pandas as pd
import matplotlib.pyplot as plt

# : and -> are "type annotations"; don't worry about it now
def read_rheometer_data(filepath: str) -> pd.DataFrame:
    dataframe = pd.read_csv(filepath, delimiter="\t", header = [0, 1], skiprows=119, engine = "c")
    return dataframe

def plot_frequency_moduli(dataframe: pd.DataFrame, figsize = (7, 5)) -> plt.Figure:
    freq = dataframe[('Angular frequency', 'rad/s')].to_numpy()
    storage = dataframe[('Storage modulus', 'Pa')].to_numpy()
    loss = dataframe[('Loss modulus', 'Pa')].to_numpy()

    fig, ax = plt.subplots(1, 1, figsize = figsize)
    ax2 = ax.twinx()
    
    axes = [ax, ax2]
    names = ['Storage Modulus [Pa]', 'Loss Modulus [Pa]']
    moduli = [storage, loss]
    colors = ["blue", "red"]
    
    for a, name, G, c in zip(axes, names, moduli, colors):
        a.set_yscale("symlog", base = 10) 
        a.set_ylabel(name)
        a.plot(freq, G, "o-", color = c, label = name)
        a.tick_params(axis='y', colors=c)
        a.yaxis.label.set_color(c)

    ax.set_xscale("log", base = 10)
    ax.set_xlabel('Angular_Frequency[rad/s]')
    ax.set_title('Modulus & Angular_Frequency_Graph')
    ax.grid(ls = "--", color = "lightgray")

    return fig