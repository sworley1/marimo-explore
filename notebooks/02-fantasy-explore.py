import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    mo.md(
        """# Ideas  
        * Use clustering to pick best fantasy predictors   
        * Who had the biggest movement year over year
        * How long does a player typically remain in the top 25 players? Where's the dropoff? How does this vary by position?
        * 
        """
    )
    return alt, mo, pl


@app.cell
def _(pl):
    import nfl_data_py as nfl
    years = [2024]
    rePullBool = False
    if rePullBool == True:
        seasonal_data = pl.from_pandas(
            nfl.import_seasonal_data(years)
        )
        seasonal_data.write_parquet('./data/01-raw/2024_seasonal.parquet')
    else:
        seasonal_data = pl.read_parquet('./data/01-raw/2024_seasonal.parquet')

    return nfl, rePullBool, seasonal_data, years


@app.cell
def _(seasonal_data):
    seasonal_data
    return


@app.cell
def _(pl):
    roster = pl.read_parquet('./data/01-raw/roster.parquet')
    roster
    return (roster,)


@app.cell
def _(pl, roster, seasonal_data):
    merge = (
        seasonal_data
        .join(
            roster.select(['player_id','player_name','team','position','headshot_url'])
            ,on='player_id'
            ,how='left'
        )
        .with_columns(
            pl.col("fantasy_points_ppr").rank(descending=True).alias("fantasy_ranks")
        )
    )

    merge.select(['player_name','fantasy_points_ppr','position','fantasy_ranks'])
    return (merge,)


@app.cell
def _(alt, merge, pl):
    base = alt.Chart(
        merge.filter(pl.col("fantasy_ranks") <= 30)
    )
    bars = base.mark_bar().encode(
        alt.X("fantasy_ranks:O").title("Rank").axis(labelAngle=0),
        alt.Y("fantasy_points_ppr").title("Season Fantasy Points"),
        alt.Color("position:N"),
        tooltip=['fantasy_ranks','fantasy_points_ppr','position','player_name']
    )
    images = bars.mark_image(width=40, height=40,baseline='bottom').encode(
        url='headshot_url'
        #y=alt.Y('sum(fantasy_points_ppr)')
        #,x=alt.X("fantasy_ranks:O")
        ,tooltip=['fantasy_points_ppr','player_name','fantasy_ranks']
    )
    (bars+images).properties(
        title='Fantasy Point Leaders for 2024 Season'
    )
    return bars, base, images


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
