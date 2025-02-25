import marimo

__generated_with = "0.10.12"
app = marimo.App(
    width="medium",
    app_title="Poisson",
    css_file="minini.css",
    html_head_file="head.html",
    auto_download=["html"],
)


@app.cell
def _(mo):
    mo.md(f"# Poisson distribution simulator")
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
    mu = mo.ui.slider(
        0, 25, value=7, step=0.1, label="a", show_value=True, full_width=True
    )

    mo.md(
        f"""
        ### Mean of Poisson distribution (in photons absorbed).
        {mu}
        """
    )
    return (mu,)


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
def _(PoissonP, baseColor, formatFigureMain, functools, mo, mu, n, plt):
    @functools.cache
    def _plot(mu, n):

        x = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
        ]

        fH, axH = plt.subplots(figsize=(10, 6))
        # pH = axH.plot(x, poisson.pmf(x, mu), 'o', ms=10, label='poisson pmf')
        pH = axH.vlines(n - 0.25, 0, 1, colors="#FF0000", lw=3, alpha=0.7)
        pH = axH.vlines(x, 0, PoissonP(mu, x), colors=baseColor, lw=5, alpha=0.8)
        pH = axH.plot(x, PoissonP(mu, x), "o", ms=10, label="poisson pmf")
        plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.1)
        formatFigureMain(fH, axH, pH)
        # axH.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axH.set_xlabel("$k$")
        axH.set_ylabel("Probability")
        axH.set_title("$a = {0}$".format(mu), fontsize=24)
        axH.set_xticks(x)
        # axH.set_xticks(np.arange(0, 20, 2))
        if mu == 0:
            plt.ylim(-0.02, 1.02)
        else:
            plt.ylim(-0.02, 0.4)
        plt.tight_layout()
        return plt.gca()


    mo.md(
        f"""
        {mo.as_html(_plot(mu.value, n.value))}
        """
    )
    return


@app.cell
def _(mo, mu, n, np, sumPoissonP):
    mo.md(
        f"## Sum of poisson probabilities below n: P(k<{n.value})={np.round(sumPoissonP(mu.value,np.arange(0,n.value,1)),6)}"
    )
    return


@app.cell
def _(mo, mu, n, np, sumPoissonP):
    mo.md(
        f"## Sum of poisson probabilities above n: P(k≥{n.value})={np.round(1-sumPoissonP(mu.value,np.arange(0,n.value,1)),6)}"
    )
    return


@app.cell
def _(mo):
    checkbox = mo.ui.checkbox(label="Log scale")
    return (checkbox,)


@app.cell
def _(checkbox, mo):
    mo.hstack([checkbox, mo.md(f"Log scale?: {checkbox.value}")])
    return


@app.cell
def _(
    PoissonP,
    baseColor,
    checkbox,
    formatPlot,
    functools,
    mo,
    mu,
    n,
    np,
    pC2,
    plt,
    ticker,
):
    @functools.cache
    def _plotCurve(mu, n, logScale):

        if logScale:
            a = np.logspace(-1, 1.6, num=100)
        else:
            a = np.arange(0, 21, 2)

        x = np.arange(0, n, 1)

        P = np.empty(np.shape(a))
        i = 0
        for f in a:
            pmf = PoissonP(f, x)
            P[i] = 1 - np.sum(pmf)
            i = i + 1

        Pk = 1 - np.sum(PoissonP(mu, np.arange(0, n, 1)))

        fH, axH = plt.subplots(figsize=(8, 6))
        pH = axH.plot(a, P, "o", ms=10, label="poisson pmf", color=pC2["p6"])
        pH = axH.plot(mu, Pk, "o", ms=10, label="poisson pmf", color=baseColor)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.1)
        formatPlot(ax=axH)
        axH.annotate(
            f"$a={mu}$",
            xy=(mu, Pk),
            xytext=(10, 0.2),
            fontsize="24",
            arrowprops=dict(facecolor="white", shrink=0.1),
        )
        if logScale:
            axH.annotate(
                f"$n={n}$",
                xy=(0.1, 0.1),
                xytext=(18, 0),
                fontsize="24",
                color=pC2["p6"],
            )
            axH.set_xscale("log")
            axH.set_xticks([0.1, 1, 10])
            axH.xaxis.set_major_formatter(ticker.ScalarFormatter())
        else:
            axH.annotate(
                f"$n={n}$",
                xy=(0, 0),
                xytext=(18, 0),
                fontsize="24",
                color=pC2["p6"],
            )
            axH.set_xscale("linear")
            axH.set_xticks([0, 5, 10, 15, 20])
            axH.xaxis.set_major_formatter(ticker.ScalarFormatter())

        axH.set_ylim([-0.02, 1.05])
        # # axH.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axH.set_xlabel("$a$")
        axH.set_ylabel("P($k ≥ n$)")
        plt.tight_layout()
        return plt.gca()


    mo.md(
        f"""
        {mo.as_html(_plotCurve(mu.value,n.value, checkbox.value))}
        """
    )
    return


@app.cell
def _():
    # ks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # df = pd.DataFrame({"k": ks, "mu": PoissonP(mu.value, k)})
    # mo.plain(mo.ui.dataframe(df, page_size=30))
    return


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
        pC2,
        params,
        pd,
        plt,
        poisson,
        sumPoissonP,
        ticker,
    )


if __name__ == "__main__":
    app.run()
