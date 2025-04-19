from dash import dcc, html
from data_loader import generate_dummy_data
# from data_loader import fetch_real_time_data

# Load dummy data
df = generate_dummy_data()
# df = fetch_real_time_data()

layout = html.Div([
    html.H1(
        "📊 Crypto Dashboard",
        style={
            'textAlign': 'center',
            'marginBottom': '20px',
            'marginTop': '15px',
            'color': 'black'
        }
    ),

    html.Label("Select a Cryptocurrency:"),
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

    html.Br(),

    html.Hr(style={'marginTop': '20px', 'marginBottom': '20px'}),

    dcc.Tabs(
        id="tabs",
        value='tab-1',
        className="tab-container",
        children=[
            dcc.Tab(label='Overview', value='tab-1', className='dash-tabs'),
            dcc.Tab(label='Predictions', value='tab-2', className='dash-tabs'),
            dcc.Tab(label='Backtest Results', value='tab-3', className='dash-tabs'),
            dcc.Tab(label='Forward Test Results', value='tab-4', className='dash-tabs'),
        ]
    ),

    html.Div(id='tabs-content')
], style={'padding': '50px'})
