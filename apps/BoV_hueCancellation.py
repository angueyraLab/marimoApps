import marimo

__generated_with = "0.11.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _(mo):
    mo.md("""# Hue cancellation Demo""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Test Color""")
    return


@app.cell
def _(mo):
    R = mo.ui.slider(
        0, 255, value=125, step=1, label="R", show_value=True, full_width=True
    )
    G = mo.ui.slider(
        0, 255, value=125, step=1, label="G", show_value=True, full_width=True
    )
    B = mo.ui.slider(
        0, 255, value=0, step=1, label="B", show_value=True, full_width=True
    )

    mo.md(
        f"""
        ### Mean of Poisson distribution (in photons absorbed).
        {R}
        {G}
        {B}
        """
    )
    return B, G, R


@app.cell
def _(mo):
    mo.md(r"""### Cancel Hue""")
    return


@app.cell
def _(mo):
    cancelHue = mo.ui.dropdown(
        options=["Blue", "Yellow", "Red", "Green"],
        value="Yellow",
        label="Choose cancel hue",
    )
    alpha = mo.ui.slider(
        0, 100, value=0, step=1, label="alpha", show_value=True, full_width=True
    )


    mo.md(
        f"""
        ### Mean of Poisson distribution (in photons absorbed).
        {cancelHue}
        {alpha}
        """
    )
    return alpha, cancelHue


@app.cell
def _():
    return


@app.cell
def _(B, G, R, alpha, cancelHue, draw, functools, mo):
    @functools.cache
    def _getTestColor(r, g, b):
        return "rgb({0},{1},{2}".format(r, g, b)


    @functools.cache
    def _getCancelHue(cH):
        if cH == "Yellow":
            cHV = "rgb(255,255,0)"
        elif cH == "Blue":
            cHV = "rgb(0,0,255)"
        elif cH == "Green":
            cHV = "rgb(255,0,0)"
        elif cH == "Red":
            cHV = "rgb(0,255,0)"
        else:
            cHV = "rgb(0,0,0)"
        return cHV


    @functools.cache
    def _plotTest(r, g, b, cH, a):
        # Create drawing
        d = draw.Drawing(
            2, 2, origin="center", context=draw.Context(invert_y=True)
        )
        d.set_render_size(500)
        d.append(draw.Circle(0, 0, 1, fill=_getTestColor(r, g, b), fill_opacity=1))
        d.append(draw.Circle(0, 0, 1, fill=_getCancelHue(cH), fill_opacity=a))
        group = draw.Group()
        d.append(group)

        return d


    mo.md(
        f"""
        {mo.as_html(_plotTest(R.value,G.value,B.value, cancelHue.value, alpha.value/100))}
        """
    )
    return


@app.cell
def _(cancelHue):
    cancelHue.value
    return


@app.cell
def _():
    return


@app.cell
def _():
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
    import drawsvg as draw

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
    return (
        baseColor,
        draw,
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
        ticker,
    )


if __name__ == "__main__":
    app.run()
