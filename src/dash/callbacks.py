import base64
from dash import Input, Output, html, dcc
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import generate_dummy_data, load_model

# Load dummy data and model
df = generate_dummy_data()
model = load_model("./models/HMM_Model.pkl")

def register_callbacks(app):

    @app.callback(
        Output('tabs-content', 'children'),
        Input('tabs', 'value')
    )
    def render_tab_content(tab):
        if tab == 'tab-1':
            return html.Div([
                dcc.Graph(id='price-graph'),
                html.Div(id='result-box', style={'marginTop': '20px', 'fontWeight': 'bold'})
            ])

        elif tab == 'tab-2':
            return html.Div([
                html.H3(
                    'Predict Market Regime Based on Selection',
                    className='tab-title',
                    style={'textAlign': 'center', 'fontSize': '30px', 'fontWeight': 'bold'}
                ),
                html.P(
                    "Prediction is based on selected cryptocurrency and date range from the Overview tab.",
                    className='tab-text',
                    style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '40px'}
                ),
                html.Div(
                    id='prediction-output',
                    className='tab-container',
                    style={
                        'padding': '30px',
                        'borderRadius': '12px',
                        'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.08)',
                        'backgroundColor': '#ffffff',
                        'maxWidth': '750px',
                        'margin': 'auto',
                        'marginBottom': '40px',
                        'border': '1px solid #e0e0e0'
                    }
                )
            ])

        elif tab == 'tab-3':  # Backtest results tab
            try:
                # Load pre-generated results
                strategy_df = pd.read_csv("./data/crypto_strategy_output.csv", parse_dates=["start_time"])

                returns = strategy_df["Net_Return"].dropna()
                sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)
                cum = (1 + strategy_df["Net_Return"]).cumprod()
                peak = cum.cummax()
                drawdown = (cum - peak) / peak
                max_dd = drawdown.min()
                trade_freq = (strategy_df["Action"].shift() != strategy_df["Action"]).mean()

                image_path = "./results/performance_visualization.png"
                with open(image_path, 'rb') as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode()

                return html.Div([
                    html.H3(
                        'Strategy Backtest Summary',
                        className='tab-title',
                        style={'textAlign': 'center', 'fontSize': '30px', 'fontWeight': 'bold'}
                    ),
                    html.P(
                        'Backtest results are based on HMM model predictions.',
                        className='tab-text',
                        style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '40px'}
                    ),
                    # Row of metrics
                    html.Div([
                        html.Div([ 
                            html.P("📈 Sharpe Ratio", style={'fontWeight': 'bold', 'fontSize': '20px'}),
                            html.H4(f"{sharpe:.2f}", style={'color': 'black', 'fontSize': '35px'})
                        ], style={'width': '20%', 'textAlign': 'center'}),

                        html.Div([ 
                            html.P("📉 Max Drawdown", style={'fontWeight': 'bold', 'fontSize': '20px'}),
                            html.H4(f"{max_dd:.2%}", style={'color': 'black', 'fontSize': '35px'})
                        ], style={'width': '20%', 'textAlign': 'center'}),

                        html.Div([ 
                            html.P("🔁 Trade Frequency", style={'fontWeight': 'bold', 'fontSize': '20px'}),
                            html.H4(f"{trade_freq:.2%}", style={'color': 'black', 'fontSize': '35px'})
                        ], style={'width': '20%', 'textAlign': 'center'})
                    ], style={
                        'display': 'flex',
                        'justifyContent': 'space-around',
                        'columnGap': '10px',
                        'marginTop': '15px',
                        'marginBottom': '20px'
                    }),

                    # Graph below metrics
                    html.Div([ 
                        html.P("Performance Graph", style={'fontWeight': 'bold', 'fontSize': '20px'}),
                        html.Img(
                            src=f'data:image/png;base64,{encoded_image}',
                            style={"width": "100%", "maxWidth": "700px", "border": "1px solid #ccc"}
                        )
                    ], style={'display': 'flex-col', 'justifyContent': 'center'})
                ], className='tab-container')

            except Exception as e:
                return html.Div([ 
                    html.H3("❗ Error Loading Backtest Results", className='tab-title'),
                    html.P(str(e), className='tab-text')
                ], className='tab-container')

        elif tab == 'tab-4':  # Forward testing results tab
            try:
                # Forward test results processing
                forward_test_df = pd.read_csv("./data/forward_test_results.csv", parse_dates=["timestamp"])

                # Calculate forward test metrics like cumulative returns and other performance measures
                forward_cum_returns = (1 + forward_test_df["Net_Return"]).cumprod()
                forward_peak = forward_cum_returns.cummax()
                forward_drawdown = (forward_cum_returns - forward_peak) / forward_peak
                forward_max_dd = forward_drawdown.min()

                image_path = "./results/forward_test_visualization.png"
                with open(image_path, 'rb') as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode()

                return html.Div([
                    html.H3(
                        'Forward Test Summary',
                        className='tab-title',
                        style={'textAlign': 'center', 'fontSize': '30px', 'fontWeight': 'bold'}
                    ),
                    html.P(
                        'Forward test results based on live market predictions.',
                        className='tab-text',
                        style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '40px'}
                    ),
                    # Row of metrics
                    html.Div([
                        html.Div([ 
                            html.P("📈 Cumulative Return", style={'fontWeight': 'bold', 'fontSize': '20px'}),
                            html.H4(f"{forward_cum_returns[-1]:.2f}", style={'color': 'black', 'fontSize': '35px'})
                        ], style={'width': '20%', 'textAlign': 'center'}),

                        html.Div([ 
                            html.P("📉 Max Drawdown", style={'fontWeight': 'bold', 'fontSize': '20px'}),
                            html.H4(f"{forward_max_dd:.2%}", style={'color': 'black', 'fontSize': '35px'})
                        ], style={'width': '20%', 'textAlign': 'center'})
                    ], style={
                        'display': 'flex',
                        'justifyContent': 'space-around',
                        'columnGap': '10px',
                        'marginTop': '15px',
                        'marginBottom': '20px'
                    }),

                    # Graph below metrics
                    html.Div([ 
                        html.Img(
                            src=f'data:image/png;base64,{encoded_image}',
                            style={"width": "100%", "maxWidth": "700px", "border": "1px solid #ccc"}
                        )
                    ], style={'display': 'flex', 'justifyContent': 'center'})
                ], className='tab-container')

            except Exception as e:
                return html.Div([ 
                    html.H3("❗ Error Loading Forward Test Results", className='tab-title'),
                    html.P(str(e), className='tab-text')
                ], className='tab-container')

    @app.callback(
        Output('price-graph', 'figure'),
        Input('crypto-dropdown', 'value'),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    )
    def update_graph(crypto, start_date, end_date):
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        fig = px.line(filtered_df, x='Date', y=crypto, title=f"{crypto} Daily Price Trend Over Selected Period")
        fig.update_layout(
            title_font_size=20,
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title=f"{crypto} Price (USD)",
            template="plotly_dark"
        )
        return fig

    @app.callback(
        Output('result-box', 'children'),
        Input('crypto-dropdown', 'value')
    )
    def update_output(selected_value):
        return f"You chose {selected_value}"

    @app.callback(
        Output('prediction-output', 'children'),
        Input('crypto-dropdown', 'value'),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    )
    def predict_regime(crypto, start_date, end_date):
        np.random.seed(0)
        n = 30

        dummy_features = pd.DataFrame({
            "active_addresses": np.random.rand(n),
            "exchange_inflow": np.random.rand(n),
            "exchange_outflow": np.random.rand(n),
            "exchange_whale_ratio": np.random.rand(n),
            "transaction_count": np.random.rand(n),
            "reserve_usd": np.random.rand(n),
            "SSR_v": np.random.rand(n),
            "funding_rate": np.random.rand(n),
            "open_interest": np.random.rand(n),
        })

        try:
            if len(dummy_features) < 30:
                return html.Div([ 
                    html.P("❗ Not enough data for prediction.", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': '#d9534f'})
                ])

            scaler = StandardScaler()
            scaled_input = scaler.fit_transform(dummy_features)

            current_state = model.predict(scaled_input)[-1]
            next_state = np.argmax(model.transmat_[current_state])
            confidence = model.transmat_[current_state][next_state]

            regime_map = {0: "Bear", 1: "Neutral", 2: "Bull"}
            regime_color = {0: "red", 1: "yellow", 2: "green"}

            return html.Div([
                html.H4("🧠 Regime Prediction Summary", style={
                    'fontWeight': 'bold', 'fontSize': '24px', 'marginBottom': '25px', 'textAlign': 'center', 'color': '#333'}),

                # Table with nicer styling
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Feature", style={
                                'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '18px', 'color': '#6c757d', 'backgroundColor': '#f8f9fa', 'padding': '10px'}),
                            html.Th("Value", style={
                                'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '18px', 'color': '#6c757d', 'backgroundColor': '#f8f9fa', 'padding': '10px'})
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td("📍 Current Market Regime", style={'fontWeight': 'bold', 'textAlign': 'center', 'padding': '10px'}),
                            html.Td([
                                html.Span("●", style={'color': regime_color.get(current_state), 'fontSize': '24px', 'marginRight': '8px'}),
                                html.Span(regime_map.get(current_state, "Unknown"), style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#495057'})
                            ], style={'textAlign': 'center'})
                        ], style={'borderBottom': '1px solid #ddd'}),
                        html.Tr([
                            html.Td("🔮 Predicted Next Regime", style={'fontWeight': 'bold', 'textAlign': 'center', 'padding': '10px'}),
                            html.Td([
                                html.Span("●", style={'color': regime_color.get(next_state), 'fontSize': '24px', 'marginRight': '8px'}),
                                html.Span(regime_map.get(next_state, "Unknown"), style={'fontSize': '22px', 'fontWeight': 'bold', 'color': '#495057'})
                            ], style={'textAlign': 'center'})
                        ], style={'borderBottom': '1px solid #ddd'}),
                        html.Tr([
                            html.Td("🔒 Prediction Confidence", style={'fontWeight': 'bold', 'textAlign': 'center', 'padding': '10px'}),
                            html.Td(f"{confidence:.2f}", style={'textAlign': 'center', 'fontSize': '22px', 'fontWeight': 'bold', 'color': '#495057'})
                        ], style={'borderBottom': '1px solid #ddd'})
                    ])
                ], style={'width': '90%', 'margin': 'auto', 'borderCollapse': 'collapse'})
            ])
        except Exception as e:
            return html.Div([ 
                html.P(f"❗ Error during prediction: {str(e)}", style={'color': 'red'})
            ])
