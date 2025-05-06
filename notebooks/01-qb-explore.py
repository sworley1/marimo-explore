import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Understanding the NFL QB Landscape
        And playing with creating a [Marimo](https://marimo.io/) app 

        Data from `nfl_data_py` - [link](https://github.com/nflverse/nfl_data_py)

        To start, we'll load play by play data from `nfl_data_py` for the 2024 NFL season. I'm not super interested in including all players who through a pass in the 2024 season so we'll use the count of throwing plays to identify a cutoff of players to look at.   

        Fortunately `marimo` makes this easy with the built in interactivity with altair to a dataframe.
        """
    )
    return


@app.cell
def _():
    import polars as pl
    import altair as alt
    import nfl_data_py as nfl
    import marimo as mo
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    None

    return PCA, alt, mo, nfl, pl, plt


@app.cell
def _(nfl, pl):
    years = [2024]
    reloadBool = True

    if reloadBool == True:
        print("Reloading the 2024 pbp data")
        d24 = nfl.import_pbp_data(years, downcast=False, cache=False, alt_path=None)
        d24 = pl.from_pandas(d24)
        #d24.write_parquet('./data/01-raw/2024_pbp.parquet')
    else:
        d24 = pl.read_parquet('./data/01-raw/2024_pbp.parquet')

    return d24, reloadBool, years


@app.cell
def _(alt):
    @alt.theme.register("black_marks", enable=True)
    def black_marks() -> alt.theme.ThemeConfig:
        markColor = '#2559A7'
        backgroundColor='#FAFAFA'

        categoryColors = [
            markColor,
            '#DB5461 ',
            '#6CAE75',
            '#9A049F',
            '#F4B886',
            '#D98324',
            #'#4E0110'
            '#1D6600',
            '#A82431',
        ]
        cColors = ["a82431","f5a6e6","8d80c7","2559a7","ff934f","6cae75",'49848E','60B2E5']
        cColors = ['#'+c for c in cColors]

        return {
            "config": {
                "view": {"continuousWidth": 300, "continuousHeight": 300},
                "mark": {"color": markColor, "fill": markColor },
                "legend":{'orient':'bottom'},
                'group':{'fill':backgroundColor},
                'background':backgroundColor,
                'title':{
                    'anchor':'start',
                    'fontSize':20,
                    'font':"Roboto",
                    'subtitleFontSize':13
                },
                "axis":{
                    'titleFont':"Asap Condensed",
                    'titleFontSize':14,
                    'labelFontSize':12,
                    'labelFont':'Asap Condensed'

                },
                'text':{
                    'font':"Roboto",
                    'fontSize':14
                },
                'range':{'category':cColors}

            }
        }
    alt.theme.enable('black_marks')
    None
    return (black_marks,)


@app.cell
def _(d24, pl, roster):

    qb_epa_pp = (
        d24
        .filter(
            (pl.col("qb_scramble")==1 ) | (pl.col("play_type")=='pass')
        )
        #.select(['desc','passer_id','rusher_id'])
        .group_by(
            ['passer_id']
        ).agg(
            pl.len().alias("numb_plays")
            ,pl.col('epa').alias("all_epa_list")
            ,pl.col("epa").mean().alias('mean_epa')
            ,pl.col("epa").median().alias('median_epa')
            ,pl.col("epa").std().alias("std_epa")
            ,( pl.col("epa").filter(pl.col("epa")>0).count() / pl.len() ).alias("percent_positive_plays")
            ,( pl.col("epa").filter(pl.col("epa")<0).count() / pl.len() ).alias("percent_negative_plays")

        )
        .join(
            roster.select(['player_id','team','player_name'])
            ,left_on=['passer_id']
            ,right_on=['player_id']
            ,coalesce=True
            ,how='left'
        )
        .sort(by=['mean_epa'],descending=True)
        .filter(pl.col("numb_plays")>30)

    )
    #qb_epa_pp

    return (qb_epa_pp,)


@app.cell
def _(alt, mo, qb_epa_pp):
    playCountHistAlt = alt.Chart(qb_epa_pp).mark_bar().encode(
        alt.X("numb_plays").title('Number of Plays')#.bin(True)
        ,alt.Y("count()").title("Number of Players with Count")
    ).properties(
        title='Number of Plays a QB threw or ran (uses passer_id)'
    )
    playCountHistUi= mo.ui.altair_chart(playCountHistAlt)
    return playCountHistAlt, playCountHistUi


@app.cell
def _(mo, playCountHistUi):
    mo.vstack([
        playCountHistUi,
        mo.ui.table( playCountHistUi.value )
    ])
    return


@app.cell
def _(mo, pl, qb_epa_pp):
    numb_of_plays_thresh = 250
    qb_epa_pp2 = (
        qb_epa_pp
        .filter(pl.col("numb_plays")>=numb_of_plays_thresh)
    )
    mo.md(
        f"Threshold for filtering to look at QBs is {numb_of_plays_thresh} plays"
    )
    return numb_of_plays_thresh, qb_epa_pp2


@app.cell
def _(nfl, pl, reloadBool, years):

    if reloadBool== True:
        roster = nfl.import_seasonal_rosters(years )
        #roster.to_parquet('./data/01-raw/roster.parquet')
        roster = pl.from_pandas(roster)
    else:
        roster = pl.read_parquet('./data/01-raw/roster.parquet')

    return (roster,)


@app.cell
def _(mo):
    mo.md(
        """
        Cool, now that we have a threshold, we can get on with analysis. We'll start with the most popular metric EPA.  

        ## EPA Exploration
        """
    )
    return


@app.cell
def _(nfl):
    team_desc = nfl.import_team_desc()
    #team_desc
    return (team_desc,)


@app.cell
def _(alt, numb_of_plays_thresh, qb_epa_pp2):
    # Lets look at the mean_epa (is it normally distributed?)
    league_epa_hist_base = alt.Chart(qb_epa_pp2)
    leage_epa_hist_bars = league_epa_hist_base.mark_bar().encode(
        alt.X("mean_epa").bin(True).title("Mean QB EPA/play"),
        alt.Y("count()")
    )
    leage_epa_rule = league_epa_hist_base.mark_rule(color='darkorange', size=5).encode(
        alt.X("mean(mean_epa)").title('')
    )

    leage_epa_text = leage_epa_rule.mark_text(dx=20, color='darkorange', dy=-135).encode(
        alt.Text('mean(mean_epa)', format='.2f'),
        #y=alt.datum(10)
    )
    league_epa_hist = (leage_epa_hist_bars + leage_epa_rule + leage_epa_text ).properties(
        title={'text':'Distribution of mean QB EPA per play', 'subtitle':[
            f'QBs with more than {numb_of_plays_thresh} plays',
            'Mean Avg EPA shown in dark orange'
        ]}
    )

    league_epa_hist
    return (
        leage_epa_hist_bars,
        leage_epa_rule,
        leage_epa_text,
        league_epa_hist,
        league_epa_hist_base,
    )


@app.cell
def _(mo, qb_epa_pp2):
    # What if we take the actual plays rather than the mean
    qb_epa_pp2_explode = (
        qb_epa_pp2
        .explode('all_epa_list')
        .rename({'all_epa_list':'epa'})
    )
    mo.md(
        'Looking instead at all plays rather than averaging them first'
    )
    return (qb_epa_pp2_explode,)


@app.cell
def _(alt, numb_of_plays_thresh, qb_epa_pp2_explode):
    median_epa = qb_epa_pp2_explode['epa'].median()
    mean_epa = qb_epa_pp2_explode['epa'].mean()

    epa_chart2_base = alt.Chart(qb_epa_pp2_explode)
    epa_chart2_bars = epa_chart2_base.mark_bar().encode(
        alt.X("epa").bin(maxbins=40)
        ,alt.Y("count()")
    ).properties(
        title={'text':'QB EPA/play', 'subtitle':[f'QBs with more than {numb_of_plays_thresh} plays', f'Mean : {mean_epa:.3f} (darkorange line)', f'Median : {median_epa:.3f}']}
    )
    epa_chart2_rule = epa_chart2_base.mark_rule(color='darkorange',size=5).encode(
        alt.X("mean(epa)")
    )
    epa_chart2_median = epa_chart2_base.mark_rule(color='purple',size=5).encode(
        alt.X("median(epa)")
    )
    epa_chart2 = epa_chart2_bars + epa_chart2_rule + epa_chart2_median
    epa_chart2
    return (
        epa_chart2,
        epa_chart2_bars,
        epa_chart2_base,
        epa_chart2_median,
        epa_chart2_rule,
        mean_epa,
        median_epa,
    )


@app.cell
def _(min_epa, min_epa_desc, mo):
    mo.md(f"""What play was the lowest EPA ({min_epa:.4f})?  
      >{min_epa_desc}  

      ...oof. Sorry Raiders fans
    """)
    return


@app.cell
def _(qb_epa_pp2_explode):
    min_epa = qb_epa_pp2_explode['epa'].min()

    return (min_epa,)


@app.cell
def _(d24, min_epa, pl):
    min_epa_desc = d24.filter(pl.col("epa")==min_epa)['desc'][0]
    return (min_epa_desc,)


@app.cell
def _(mo):
    mo.md(
        """
        Now that we have a baseline understanding of the distribution of EPA/play, let's look at individual QB performance.   
        Looking across:     

        * Mean EPA/play (affected by outliers)
        * Median EPA/play (resistant to outliers)
        * Standard deviation of EPA/play (how varied each QB is)
        """
    )
    return


@app.cell
def _(mo):
    rank_measure = mo.ui.dropdown(
        options=['Median EPA/play','Mean EPA/play','Standard Dev. EPA/play']
        ,label='Rank Measure: '
        ,value='Median EPA/play'
    )
    rank_measure
    return (rank_measure,)


@app.cell
def _(alt, qb_epa_pp2, rank_measure):
    if rank_measure.value == 'Median EPA/play':
        rank_measure_d = 'median_epa'
    elif rank_measure.value == 'Standard Dev. EPA/play':
        rank_measure_d = 'std_epa'
    else:
        rank_measure_d = 'mean_epa'

    qb_rank_base = alt.Chart(qb_epa_pp2)
    qb_rank_bars = qb_rank_base.mark_bar().encode(
        alt.X(rank_measure_d).title(rank_measure.value)
        ,alt.Y('player_name:N',sort='-x').title(None)
    )
    qb_rank_bars
    return qb_rank_bars, qb_rank_base, rank_measure_d


@app.cell
def _(mo):
    mo.md(r"""Quite a difference in the ranking when comparing mean EPA/play and median EPA/play""")
    return


@app.cell
def _(mo, qb_epa_pp2_explode):
    player_name_dropdown = mo.ui.dropdown(
        options=qb_epa_pp2_explode['player_name'].unique().sort().to_list()
        ,label='Select a player to look at their distribution of EPA/play'
        ,value='Josh Allen'
    )
    player_name_dropdown
    return (player_name_dropdown,)


@app.cell
def _(pl, qb_epa_pp2):
    qb_epa_pp3 = (
        qb_epa_pp2
        .with_columns(
            pl.col("median_epa").rank(descending=True).alias("Median EPA Rank")
            ,pl.col("mean_epa").rank(descending=True).alias("Mean EPA Rank")
        )
        .with_columns(
            (pl.col("Mean EPA Rank")-pl.col("Median EPA Rank")).abs().alias('delta in rank')
        )
        .sort(by='delta in rank',descending=True)
    )
    #qb_epa_pp3#.head()
    return (qb_epa_pp3,)


@app.cell
def _(alt, pl, player_name_dropdown, qb_epa_pp2_explode):
    josh_allen_chart_base = alt.Chart(
        qb_epa_pp2_explode.filter(pl.col("player_name")==player_name_dropdown.value)
    )
    josh_allen_hist = josh_allen_chart_base.mark_bar().encode(
        alt.X("epa").bin(maxbins=40)
        ,alt.Y("count()")
    )
    josh_allen_mean = josh_allen_chart_base.mark_rule(color='darkorange',strokeDash=[6,2], size=5).encode(
        x=alt.X('mean(epa)')
    )
    josh_allen_median = josh_allen_chart_base.mark_rule(color='purple',strokeDash=[6,2],size=5).encode(
        x=alt.X('median(epa)')
    )
    josh_allen_chart = josh_allen_hist + josh_allen_mean + josh_allen_median
    josh_allen_chart.properties(
        title={
            'text':f'{player_name_dropdown.value} EPA distribution' 
            ,'subtitle':['Mean EPA/play shown in orange','Median EPA/play shown in purple']
              }
    )
    return (
        josh_allen_chart,
        josh_allen_chart_base,
        josh_allen_hist,
        josh_allen_mean,
        josh_allen_median,
    )


@app.cell
def _(mo):
    mo.md(r"""EPA gives a measure of if a play helped or hurt a respective team. Lets translate that into a measure of how often a QB contributed positively using `number of EPA plays / number of QB centric plays` where the number of QB centric plays is a pass play or QB run.""")
    return


@app.cell
def _(alt, qb_epa_pp2):
    bars_percent_plays_base = alt.Chart(qb_epa_pp2)
    bars_percent_plays = bars_percent_plays_base.mark_bar().encode(
        alt.X('percent_positive_plays:Q').title("Percentage of plays with EPA > 0").axis(format='%')
        ,alt.Y("player_name",sort='-x').title(None)
    ).properties(
        title={'text':"Ranking Quarterbacks by Percentage of plays with positive EPA"}
    )

    bars_percent_plays_text = bars_percent_plays.mark_text(dx=5, align='left').encode(
        alt.Text("percent_positive_plays",format='.1%')
    )



    bars_percent = bars_percent_plays + bars_percent_plays_text

    mean_percent_circles = bars_percent_plays_base.mark_circle(color='darkorange',opacity=1, size=50).encode(
        alt.X("mean_epa")
        ,alt.Y("player_name").sort(field='percent_positive_plays')
        #,alt.Sort('percent_positive_plays')
        ,tooltip=['player_name']
    ) 

    bars_percent #+ mean_percent_circles
    return (
        bars_percent,
        bars_percent_plays,
        bars_percent_plays_base,
        bars_percent_plays_text,
        mean_percent_circles,
    )


@app.cell
def _():
    return


@app.cell
def _(d24, numb_of_plays_thresh, pl):
    for_pca = (
        d24
        .filter(
            (pl.col("qb_scramble")==1 ) | (pl.col("play_type")=='pass')
        )
        .filter(
            pl.col("qb_kneel")==0
        )
        .filter(pl.col("qb_spike")==0)
        .with_columns(
            pl.when(pl.col("passer_player_id").is_not_null()).then(pl.col("passer_player_id"))
            .when(pl.col("rusher_player_id").is_not_null()).then(pl.col("rusher_player_id"))
            .otherwise(pl.lit(None)).alias("qb_id")
        )
        .filter(pl.col("qb_id").is_not_null())
        .group_by(['qb_id'])
        .agg(
            pl.len().alias("numb_plays")
            ,pl.col("passer_player_id").filter(pl.col("passer_player_id")==pl.col("qb_id")).count().alias('pass_attempt_count')
            ,pl.col('incomplete_pass').sum().alias("numb_incomplete_pass")
            ,pl.col("interception").sum().alias("numb_interception")
            ,pl.col("complete_pass").sum().alias("numb_complete_pass")
            ,pl.col("sack").sum().alias("numb_sacks")
            ,pl.col("receiving_yards").sum().alias("sum_recieving_yards")
            ,pl.col("rushing_yards").sum().alias("sum_rushing_yards")
            ,pl.col("air_yards").sum().alias('sum_air_yards')
            ,pl.col("qb_hit").sum().alias('numb_times_hit')
            ,pl.col("pass_touchdown").sum().alias('numb_pass_td')
            # Columns from above
            ,pl.col("epa").mean().alias('mean_epa')
            ,pl.col("epa").median().alias('median_epa')
            ,pl.col("epa").std().alias("std_epa")
            ,( pl.col("epa").filter(pl.col("epa")>0).count() / pl.len() ).alias("percent_positive_plays")
            ,( pl.col("epa").filter(pl.col("epa")<0).count() / pl.len() ).alias("percent_negative_plays")

        )
        .with_columns(
            (pl.col("numb_incomplete_pass") / pl.col("pass_attempt_count") ).alias('incomplete_pass_percent'),
            (pl.col("numb_interception") / pl.col("pass_attempt_count") ).alias("interception_percentage"),
            (pl.col("pass_attempt_count") / pl.col("numb_plays") ).alias("pass_percent"),
            (pl.col("numb_sacks") / pl.col("numb_plays")).alias("sack_percent"),
        )
        .filter(pl.col("numb_plays")>=numb_of_plays_thresh)
    #    .join(
    #        roster
    #        ,left_on=['qb_id']
    #        ,right_on=['player_id']
    #    )
        .drop('numb_plays')
        .sort(by='pass_attempt_count',descending=True)
    )
    cols_to_keep = [z for z in for_pca.columns if 'percent' in z or 'epa' in z] + ['qb_id']
    for_pca = for_pca.select(cols_to_keep)
    return cols_to_keep, for_pca


@app.cell
def _(mo):
    mo.md(
        r"""
        ## PCA Grouping

        Now that we have identified a few trends, let's look at grouping the QBs together to see how they relate and differ from each other. We'll use Principal Component Analysis to start.
        """
    )
    return


@app.cell
def _(mo):
    nComponentsUI = mo.ui.number(
        label='Select the number of components for PCA',
        value=2,
    )
    nComponentsUI
    return (nComponentsUI,)


@app.cell
def _(PCA, for_pca, mo, nComponentsUI):
    n_components = nComponentsUI.value
    pca2 = PCA(n_components=n_components)
    pca_out2 = pca2.fit_transform(for_pca.drop('qb_id'))

    mo.md(f"""Variance Explained using <u>**{n_components:,.0f}**</u> Components: {sum(pca2.explained_variance_ratio_):.4f}""")

    mo.stat(
        value=f"{sum(pca2.explained_variance_ratio_):.15f}"
        ,label='Total Variance Explained'
        ,caption=f'up {sum(pca2.explained_variance_ratio_)-sum(pca2.explained_variance_ratio_[:-1]):.3f} from {sum(pca2.explained_variance_ratio_[:-1]):.3f}   \nusing {n_components-1} Components'
        ,direction='increase'
    )
    return n_components, pca2, pca_out2


@app.cell
def _(mo):
    mo.md(r"""#### Interpretation of PCA Components:""")
    return


@app.cell
def _(mo, pca2, pl):
    test = pl.DataFrame(pca2.components_)
    test.columns = pca2.feature_names_in_
    pca_descr_df=test.transpose(include_header=True, column_names=pca2.get_feature_names_out(), header_name='feature')
    mo.accordion(
       {'Full Feature Dataframe':pca_descr_df} 
    )
    return pca_descr_df, test


@app.cell
def _():
    # ¯\_(ツ)_/¯
    import os
    import requests

    def download_image(image_url:str, image_name:str ,folder='assets'):
        # check of folder exsists
        cwd = os.getcwd()
        fpath = os.path.join(cwd, folder)
        if not os.path.exists(fpath):
            os.mkdir(fpath)

        img_data = requests.get(image_url).content
        image_format = 'png' #image_url.split('.')[-1]

        image_path =fpath+'/'+image_name+'.'+image_format 
        with open(image_path, 'wb') as handler:
            handler.write(img_data)

        return image_path

    # Columns (PCA0 , PCA1, PCA2)
    # Rows
    # - Top variable
    # - Top 2 variable
    # - Top 3 variable

    # Top 2 QBs
    # Bottom 2 QBs

    # Build DF
    numb_players_to_look = 2
    descrSummaryList2 = []
    columns = []

    '''
    for p2 in pca2.get_feature_names_out():
        t2 = pca_descr_df2.sort(by=f'{p2}_abs',descending=True)
        roster2 = pca_result_df.sort(by=f'PC{p2[-1]}', descending=True)

        top_qbs = roster2['player_name'][:numb_players_to_look]
        top_qbs_image = roster2['headshot_url'][:numb_players_to_look]

        bottom_qbs = roster2.sort(by=f'PC{p2[-1]}', descending=False)
        bottom_qbs_image = bottom_qbs['headshot_url'][:numb_players_to_look]
        bottom_qbs = bottom_qbs['player_name'][:numb_players_to_look] 

        image_fps = []
        for i2, image1 in enumerate( top_qbs_image) :
            #image_fp = download_image( image1, top_qbs[i2] )
            #image_fps.append(image_fp)

        bottomg_images = []
        for i2, image1 in enumerate( bottom_qbs_image ):
            image_fp = download_image(image1, bottom_qbs[i2] )
            bottomg_images.append(image_fp)

        fnames = t2['feature'][:numb_of_cols_to_look_at].to_list()
        vals2 = t2[f'{p}_abs'][:numb_of_cols_to_look_at].to_list()
        raw_vals2 = t2[p][:numb_of_cols_to_look_at].to_list()

        col1 = [p2] + [p2+'_1'] + fnames + [','.join(image_fps)] + [','.join(bottomg_images)]
        #col2 = [p2] + [p2+'_1'] + raw_vals2 + [image_fps[1]]

        columns.append(col1)
        #columns.append(col2)

    '''
    None
    return (
        columns,
        descrSummaryList2,
        download_image,
        numb_players_to_look,
        os,
        requests,
    )


@app.cell
def _(mo, pca2, pca_descr_df, pl):
    pca_descr_df2 = (
        pca_descr_df    
        .with_columns(
            [
                pl.col(x).abs().alias(x+"_abs") for x in pca2.get_feature_names_out() 
            ]
        )
    )
    numb_of_cols_to_look_at = 3
    descrSummaryList = []
    markDownList = []
    nl = '\n'
    for p in pca2.get_feature_names_out():
        t = pca_descr_df2.sort(by=f'{p}_abs',descending=True)
        names = t['feature'][:numb_of_cols_to_look_at]
        vals = t[f'{p}_abs'][:numb_of_cols_to_look_at]
        raw_vals = t[p][:numb_of_cols_to_look_at]

        text = f"""### {p.upper()}    
            { f'   {nl}'.join( [f'{"+" if v_raw > 0 else "-"} {n} ({v_raw:.3f})' for v,n,v_raw in zip(vals,names,raw_vals)] ) }
            """

        markDownList.append(text)

    mo.hstack(
        [mo.md(m) for m in markDownList]
    )
    return (
        descrSummaryList,
        markDownList,
        names,
        nl,
        numb_of_cols_to_look_at,
        p,
        pca_descr_df2,
        raw_vals,
        t,
        text,
        vals,
    )


@app.cell
def _(alt, for_pca, n_components, pca_out2, pl, roster, x_axis_ui, y_axis_ui):
    player_id_series = for_pca['qb_id']
    pca_result_df = pl.DataFrame(pca_out2,schema=[f'PC{i}' for i in range(0,n_components)]).insert_column(0, player_id_series).join(
        roster,
        left_on=['qb_id'],
        right_on=['player_id']
    )

    alt.Chart(pca_result_df).mark_image(width=50,height=50).encode(
        alt.X(x_axis_ui.value),
        alt.Y(y_axis_ui.value),
        url='headshot_url',
        tooltip=['player_name']
    ).interactive().properties(
        title='NLF QB Landscape shown with PCA'
    )
    return pca_result_df, player_id_series


@app.cell
def _(mo, n_components):
    x_axis_ui = mo.ui.dropdown(
        label='Select the X-axis for chart'
        ,options=[f'PC{i}' for i in range(0,n_components)]
        ,value=[f'PC{i}' for i in range(0,n_components)][0]
        ,full_width=False
    )
    y_axis_ui = mo.ui.dropdown(
        label='Select the X-axis for chart'
        ,options=[f'PC{i}' for i in range(0,n_components)]
        ,value=[f'PC{i}' for i in range(0,n_components)][-1]
        ,full_width=False
    )
    return x_axis_ui, y_axis_ui


@app.cell
def _(mo, x_axis_ui, y_axis_ui):
    mo.hstack([x_axis_ui,y_axis_ui])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Clustering
        Last, we will use clustering to group our QBs together. We will use an agglomerative/hierarchical clustering to see how QBs relate and differ to each other. The hierarchical approach will allow us to see not only which QBs are most similar, but also how much they differ across groups using the dendrogram.
        """
    )
    return


@app.cell
def _(for_pca, mo):
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram
    import numpy as np
    # Hierarchical Clustering
    # distance_threshold=0 ensures we compute the full tree
    #model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)


    model = AgglomerativeClustering(
        linkage='ward',
        compute_full_tree=True,
        n_clusters=None,
        distance_threshold=0
    )

    model = model.fit(for_pca.drop(['qb_id']))

    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)


    mo.hstack(
        [
            mo.stat(model.n_clusters_ ,label='Number of Clusters')
        ]
    )

    return AgglomerativeClustering, dendrogram, model, np, plot_dendrogram


@app.cell
def _(model, np):

    n_samples = len(model.labels_)
    counts = np.zeros(model.children_.shape[0])
    for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

    linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
    ).astype(float)
    return (
        child_idx,
        counts,
        current_count,
        i,
        linkage_matrix,
        merge,
        n_samples,
    )


@app.cell
def _():
    import scipy.cluster.hierarchy as h
    return (h,)


@app.cell
def _(mo):
    use_pca_checkbox = mo.ui.checkbox(
        label='Use PCA transformation in clustering?'
    )
    use_pca_checkbox
    return (use_pca_checkbox,)


@app.cell
def _(for_pca, h, pca_out2, plt, roster, use_pca_checkbox):

    labelsDendrogram = (

        for_pca
        .join(
            roster
            ,left_on=['qb_id']
            ,right_on=['player_id']
            ,how='left'
        )
    )['player_name'].to_list()

    # Calculate the distance between each sample
    if use_pca_checkbox.value == False:
        #print('Running wihtout PCA')
        Z = h.linkage(for_pca.drop('qb_id'), 'ward')
    else:
        #print("Running on PCA Analysis")
        Z = h.linkage(pca_out2, 'ward')

    # Plot with Custom leaves
    fig = plt.figure()
    plt.title('Clustering Analysis on QBs')
    h.dendrogram(Z, leaf_rotation=0, leaf_font_size=8, labels=labelsDendrogram, orientation='right')

    # Show the graph
    fig
    return Z, fig, labelsDendrogram


@app.cell
def _():
    '''
    from great_tables import GT, style, google_font

    for_gt_table = pl.DataFrame(columns)
    for_gt_table = for_gt_table.insert_column(0,
        pl.Series(values=["PCA#",'TMP','1st Descriptor','2nd Descriptor','3rd Descriptor','Top QBs','Bottom QBs'],name='row'
    )
                                                       )
    for_gt_table2 = for_gt_table.transpose(column_names='row')

    if False:
        x= (
            GT(for_gt_table,rowname_col='row')
            .tab_header('PCA Components Interpretation')
            .fmt_image(rows=['Top QBs','Bottom QBs'],encode=True)
            .opt_table_font(font=google_font('Roboto Condensed'))
            .save('test.png')
        )
        mo.image('test.png')
    '''
    None
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
