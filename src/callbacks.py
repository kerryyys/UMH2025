import base64
from dash import Input, Output, html, dcc
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import generate_dummy_data, load_model

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
                    className='tab-title'
                ),
                html.P(
                    "Prediction is based on selected cryptocurrency and date range from the Overview tab.",
                    className='tab-text'
                ),
                html.Div(
                    id='prediction-output',
                    className='tab-container'
                )
            ])
        elif tab == 'tab-3':
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
                
                evaluation_text = [
                    html.P(f"📈 Sharpe Ratio: {sharpe:.2f}", className='tab-text'),
                    html.P(f"📉 Max Drawdown: {max_dd:.2%}", className='tab-text'),
                    html.P(f"🔁 Trade Frequency: {trade_freq:.2%}", className='tab-text'),
                ]

                image_path = "./results/performance_visualization.png"
                with open(image_path, 'rb') as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode()

                return html.Div([
                    html.H3(
                        '📊 Strategy Backtest Summary',
                        className='tab-title'
                    ),
                    html.P(
                        'Backtest results are based on HMM model predictions.',
                        className='tab-text'
                    ),
                    html.Div([  # This div holds the evaluation text and image
                        html.Div(
                            evaluation_text, style={'display': 'flex', 'flexDirection': 'column', 'width': '30%'}
                        ),
                        html.Div([
                            html.Img(
                                src=f'data:image/png;base64,{encoded_image}',
                                style={"width": "100%", "maxWidth": "600px", "border": "1px solid #ccc"}
                            )
                        ], style={'flex': 1, 'marginLeft': '20px'}
                        )  # Graph takes more space
                    ], 
                        style={'display': 'flex', 'alignItems': 'flex-start', 'justifyContent': 'space-between'},
                        className='tab-container'
                    ),
                ])
                
            except Exception as e:
                return html.Div([
                    html.H3("❗ Error Loading Backtest Results", className='tab-title'),
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
        fig = px.line(filtered_df, x='Date', y=crypto,
                    title=f"{crypto} Daily Price Trend Over Selected Period")
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
        
        # Might need to adjust this part to fetch real data for the last 30? days
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
                return "❗ Not enough data for prediction."

            scaler = StandardScaler()
            scaled_input = scaler.fit_transform(dummy_features)

            current_state = model.predict(scaled_input)[-1]
            next_state = np.argmax(model.transmat_[current_state])
            confidence = model.transmat_[current_state][next_state]

            regime_map = {0: "Bear", 1: "Neutral", 2: "Bull"}

            return html.Div([
                html.H4(f"📍 Current State: {regime_map.get(current_state, 'Unknown')}"),
                html.H4(f"🔮 Predicted Next Regime: {regime_map.get(next_state, 'Unknown')}"),
                html.H4(f"✅ Confidence: {confidence:.2%}")
            ])
        except Exception as e:
            return f"⚠️ Error during prediction: {str(e)}"
