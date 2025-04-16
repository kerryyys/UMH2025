from dash import dcc, html
from data_loader import generate_dummy_data

df = generate_dummy_data()

layout = html.Div([
    html.H1("📊 Simple Crypto Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),

    html.Label("Choose a Cryptocurrency:"),
    dcc.Dropdown(
        id='crypto-dropdown',
        options=[{'label': coin, 'value': coin} for coin in ['Bitcoin', 'Ethereum']],
        value='Bitcoin'
    ),

    html.Br(),

    html.Label("Select Date Range:"),
    dcc.DatePickerRange(
        id='date-range',
        start_date=df['Date'].min(),
        end_date=df['Date'].max(),
        display_format='YYYY-MM-DD'
    ),        

    html.Hr(),

    dcc.Tabs(
        id="tabs",
        value='tab-1',
        className="tab-container", # link to custom CSS
        children=[
            dcc.Tab(label='Overview', value='tab-1', className='dash-tabs'),
            dcc.Tab(label='Predictions', value='tab-2'),
            dcc.Tab(label='Backtest Results', value='tab-3'),
        ]
    ), 
    html.Div(id='tabs-content')
])