import matplotlib


def configure_matplotlib_defaults(fontsize: int = 12) -> None:
    matplotlib.rc("font", size=fontsize)
    matplotlib.rc("axes", grid=True)
    matplotlib.rc("axes.spines", top=False, right=False)
    matplotlib.rc("grid", linestyle="--", color="lightgray")
    matplotlib.rc("savefig", facecolor="white")
