import altair as alt

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

