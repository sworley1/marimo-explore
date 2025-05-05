import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl
    import altair as alt
    import helper 
    alt.themes.enable('black_marks')
    return alt, helper, pl


@app.cell
def _(pl):
    d = pl.read_parquet('./data/01-raw/2024_pbp.parquet')
    d.head()
    return (d,)


@app.cell
def _(d, pl):
    posteam_epa = d.filter(
        pl.col("posteam").is_not_null()
    ).group_by(['posteam']).agg( pl.col("epa").mean().alias('off_epa') )

    defteam_epa = d.filter(
        pl.col("defteam").is_not_null()
    ).group_by(['defteam']).agg( pl.col("epa").mean().alias('def_epa') )

    for_plotting = posteam_epa.join(
        defteam_epa
        ,right_on=['defteam']
        ,left_on=['posteam']
    )

    return defteam_epa, for_plotting, posteam_epa


@app.cell
def _(alt, for_plotting):
    alt.Chart(for_plotting).mark_circle(size=100, opacity=1).encode(
        alt.X("off_epa"),
        alt.Y("def_epa"),
        color=alt.Color('posteam:N')
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
