import marimo

__generated_with = "0.10.12"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _(mo):
    mo.md(f"# Frequency of seeing curves")
    return


@app.cell
def _(mo):
    mo.md(f"### Created by Juan Angueyra (biologyofvision@umd.edu), Jan 2025")
    return


@app.cell
def _(mo):
    mo.md(
        f"> Modeled after [Hecht, Schlaer and Pirenne (1942)](https://doi.org/10.1085/jgp.25.6.819)"
    )
    return


@app.cell
def _(mo):
    mo.md(
        f"> Includes rod data from [Baylor, Nunn, Schnapf (1984)](https://doi.org/10.1113/jphysiol.1984.sp015518)"
    )
    return


@app.cell
def _(mo, radio):
    mo.hstack([radio, mo.md(f"Observer: {radio.value}")])
    return


@app.cell
def _(mo):
    n = mo.ui.slider(
        0, 20, value=4, step=1, label="n", show_value=True, full_width=True
    )

    mo.md(
        f"""
        ### Threshold of vision (in photons).

        {n}
        """
    )
    return (n,)


@app.cell
def _(mo):
    alpha = mo.ui.slider(
        0.1,
        300,
        value=10,
        step=0.1,
        label="α",
        show_value=True,
        full_width=True,
    )

    mo.md(
        f"""
        ### Correction factor α for a.
        {alpha}
        """
    )
    return (alpha,)


@app.cell
def _(
    FOS,
    a,
    allP,
    alpha,
    formatPlot,
    functools,
    mo,
    n,
    pC2,
    plt,
    radio,
    ticker,
):
    @functools.cache
    def _plotFOS(n, alpha):
        P = allP[n, :]

        fH, axH = plt.subplots(figsize=(8, 6))
        pH = axH.plot(a * alpha, P, "-", ms=5, label="FOS", color=pC2["p1"])
        pH = axH.plot(
            FOS["a"].values,
            FOS["pSeen"].values,
            "x",
            ms=8,
            label="pSeen",
            color="#000000",
        )
        axH.annotate(
            f"$n={n}$",
            xy=(0.1, 0.1),
            xytext=(0.11, 0.11),
            fontsize="24",
            color=pC2["p1"],
        )
        axH.annotate(
            f"{radio.value}",
            xy=(0.1, 0.1),
            xytext=(0.11, 0.97),
            fontsize="24",
            color="#000000",
        )
        axH.set_xscale("log")
        axH.set_xticks([10, 100, 1000])
        axH.xaxis.set_major_formatter(ticker.ScalarFormatter())

        formatPlot(ax=axH)

        # axH.set_ylim([-0.02, 1.05])
        # # axH.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axH.set_xlabel("$α×a$")
        axH.set_ylabel("$P_{seen} = n_{seen} / n_{total}$")
        plt.tight_layout()
        return plt.gca()


    mo.md(
        f"""
        {mo.as_html(_plotFOS(n.value, alpha.value))}
        """
    )
    return


@app.cell
def _(PoissonP, np, pd, radio):
    logScale = True
    if logScale:
        a = np.logspace(-1, 1.6, num=100)
    else:
        a = np.arange(0, 21, 0.5)

    ns = np.arange(0, 20, 1)
    allP = np.empty([np.shape(ns)[0], np.shape(a)[0]])

    for n_now in ns:
        i = 0
        for f in a:
            pmf = PoissonP(f, np.arange(0, n_now, 1))
            allP[n_now, i] = 1 - np.sum(pmf)
            i = i + 1
            
            

    if radio.value == "Hecht":
        FOS = pd.read_csv(str(mo.notebook_location() / "public" / "BoV" / "HSP_FOS_SH.csv"))
    elif radio.value == "Schlaer":
        FOS = pd.read_csv(str(mo.notebook_location() / "public" / "BoV" / "HSP_FOS_SS.csv")")
    elif radio.value == "Pirenne":
        FOS = pd.read_csv(str(mo.notebook_location() / "public" / "BoV" / "HSP_FOS_MHP.csv"))
    elif radio.value == "Rods":
        FOS = pd.read_csv(str(mo.notebook_location() / "public" / "BoV" / "Rod_FOSCombo.csv"))
    else:
        FOS = pd.read_csv(str(mo.notebook_location() / "public" / "BoV" / "HSP_FOS_SH.csv"))
    return FOS, a, allP, f, i, logScale, n_now, ns, pmf


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager
    import matplotlib.ticker as ticker
    import math
    import functools
    from scipy.stats import poisson

    # default parameters for plotting
    plt.style.use("default")
    params = {
        "ytick.color": "k",
        "xtick.color": "k",
        "axes.labelcolor": "k",
        "axes.edgecolor": "k",
        "axes.linewidth": 3,
        "xtick.major.width": 3,
        "ytick.major.width": 3,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "text.color": "k",
    }
    plt.rcParams.update(params)
    baseColor = "#000000"  # black

    pC2 = {
        "p0": "#001160",
        "p1": "#185462",
        "p2": "#116395",
        "p3": "#A6C9DA",
        "p4": "#E1B79F",
        "p5": "#B75925",
        "p6": "#590007",
        "p7": "#34000B",  # invented; not colormap
        "p8": "#190007",  # invented; not colormap
    }


    def formatFigureMain(figH, axH, plotH):
        fontTicks = font_manager.FontProperties(size=20)
        fontLabels = font_manager.FontProperties(size=24)
        fontTitle = font_manager.FontProperties(size=22)
        axH.set_xscale("linear")
        axH.spines["top"].set_visible(False)
        axH.spines["right"].set_visible(False)

        for label in axH.get_xticklabels() + axH.get_yticklabels():
            label.set_fontproperties(fontTicks)
        axH.set_xlabel(axH.get_xlabel(), fontproperties=fontTicks)
        axH.set_ylabel(axH.get_ylabel(), fontproperties=fontTicks)
        return fontLabels


    def formatPlot(ax=None):
        if not ax:
            ax = plt.gca()
        # [fontTicks, fontLabels, fontTitle] = defaultFonts(ax=ax)
        fontTicks = font_manager.FontProperties(size=20)
        fontLabels = font_manager.FontProperties(size=24)
        fontTitle = font_manager.FontProperties(size=22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(fontTicks)
        ax.set_xlabel(ax.get_xlabel(), fontproperties=fontTicks)
        ax.set_ylabel(ax.get_ylabel(), fontproperties=fontTicks)
        ax.ticklabel_format(style="plain")


    def PoissonP(mu, ks):
        p = np.empty(np.shape(ks))
        i = 0
        for k in ks:
            # p[i] = (np.power(mu, k)) / (np.exp(mu) * math.factorial(k))
            p[i] = (mu**k) / (np.exp(mu) * math.factorial(k))
            i = i + 1
        return p


    def sumPoissonP(mu, ks):
        p = PoissonP(mu, ks)
        return np.round(np.sum(p), 6)


    observers = ["Hecht", "Schlaer", "Pirenne", "Rods"]
    radio = mo.ui.radio(options=observers, value="Schlaer")
    return (
        PoissonP,
        baseColor,
        font_manager,
        formatFigureMain,
        formatPlot,
        functools,
        math,
        mo,
        np,
        observers,
        pC2,
        params,
        pd,
        plt,
        poisson,
        radio,
        sumPoissonP,
        ticker,
    )


if __name__ == "__main__":
    app.run()
