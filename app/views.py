from urllib import request
from pathlib import Path
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import RequestContext

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter

import pandas as pd
import numpy as np
import json

import yfinance as yf
import datetime as dt
import qrcode
from functools import lru_cache

from .models import Project

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm


DEFAULT_ACTIVE_STOCKS = ['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM']
APP_LOGIN_USERNAME = 'sreenivasulu'
APP_LOGIN_PASSWORD = 'Sreenu@181825'
DATA_DIR = Path(__file__).resolve().parent / 'Data'


def _data_file(filename):
    return str(DATA_DIR / filename)


def _current_username(request):
    return request.session.get('username', 'Guest')


def _require_login(request):
    if request.session.get('is_logged_in'):
        return None
    return redirect('/login/')


def login_view(request):
    if request.session.get('is_logged_in'):
        return redirect('/')

    error_message = None

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        if not username:
            error_message = 'Please enter a username.'
        elif username.lower() != APP_LOGIN_USERNAME:
            error_message = 'Invalid username.'
        elif password != APP_LOGIN_PASSWORD:
            error_message = 'Invalid password.'
        else:
            request.session['is_logged_in'] = True
            request.session['username'] = username
            request.session.modified = True
            return redirect('/')

    return render(request, 'login.html', {
        'error_message': error_message,
    })


def logout_view(request):
    request.session.flush()
    return redirect('/login/')


def _safe_float(value, default=0.0):
    """Convert strings like '$123.45' to float safely."""
    try:
        return float(str(value).replace('$', '').replace(',', '').strip())
    except Exception:
        return default


def _download_from_stooq(ticker):
    """Fallback source using Stooq daily candles."""
    try:
        stooq_symbol = '{}.us'.format(ticker.lower())
        url = 'https://stooq.com/q/d/l/?s={}&i=d'.format(stooq_symbol)
        data = pd.read_csv(url)

        required = {'Date', 'Open', 'High', 'Low', 'Close'}
        if data.empty or not required.issubset(set(data.columns)):
            return pd.DataFrame()

        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date']).set_index('Date').sort_index()

        if 'Volume' not in data.columns:
            data['Volume'] = 0
        data['Adj Close'] = data['Close']

        return data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].dropna(how='all')
    except Exception:
        return pd.DataFrame()


def _build_recent_rows_from_csv(tickers):
    """Guarantee non-empty recent stocks table using static CSV values."""
    rows = []
    try:
        ticker_df = pd.read_csv(_data_file('Tickers.csv'))
        ticker_df.columns = [
            'Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change',
            'Market_Cap', 'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry'
        ]
        ticker_df['Symbol'] = ticker_df['Symbol'].astype(str).str.upper()

        for ticker in tickers:
            match = ticker_df[ticker_df['Symbol'] == ticker]
            if match.empty:
                price = 0.0
                volume = 0
            else:
                item = match.iloc[0]
                price = _safe_float(item.get('Last_Sale', 0.0), default=0.0)
                volume = int(_safe_float(item.get('Volume', 0), default=0.0))

            rows.append({
                'Ticker': ticker,
                'Open': round(price, 2),
                'High': round(price, 2),
                'Low': round(price, 2),
                'Close': round(price, 2),
                'Adj_Close': round(price, 2),
                'Volume': volume,
            })
    except Exception:
        for ticker in tickers:
            rows.append({
                'Ticker': ticker,
                'Open': 0.0,
                'High': 0.0,
                'Low': 0.0,
                'Close': 0.0,
                'Adj_Close': 0.0,
                'Volume': 0,
            })

    return rows


def _resolve_ticker_value(raw_value):
    """Resolve user input to a symbol when the user enters company name text."""
    cleaned = str(raw_value).strip()
    candidate = cleaned.upper()

    # Fast path for common symbol-like inputs.
    if cleaned and " " not in cleaned and len(cleaned) <= 8:
        return candidate

    try:
        ticker_df = pd.read_csv(_data_file('Tickers.csv'))
        ticker_df.columns = [
            'Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change',
            'Market_Cap', 'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry'
        ]
        ticker_df['Symbol'] = ticker_df['Symbol'].astype(str).str.upper()
        ticker_df['Name'] = ticker_df['Name'].astype(str)

        # Exact symbol match.
        if candidate in set(ticker_df['Symbol']):
            return candidate

        # Exact name match, then contains match.
        exact_match = ticker_df[ticker_df['Name'].str.lower() == cleaned.lower()]
        if not exact_match.empty:
            return exact_match.iloc[0]['Symbol']

        contains_match = ticker_df[ticker_df['Name'].str.lower().str.contains(cleaned.lower(), na=False)]
        if not contains_match.empty:
            return contains_match.iloc[0]['Symbol']
    except Exception:
        pass

    return candidate


@lru_cache(maxsize=1)
def _get_valid_tickers():
    """Load valid ticker symbols from CSV once and cache in memory."""
    try:
        ticker_df = pd.read_csv(_data_file('Tickers.csv'))
        ticker_df.columns = [
            'Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change',
            'Market_Cap', 'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry'
        ]
        return set(ticker_df['Symbol'].astype(str).str.upper())
    except Exception:
        return set()


def _download_with_fallback(ticker, attempts):
    """Try multiple period/interval combinations and return first non-empty DataFrame.

    Uses yf.Ticker().history() as the primary path (yfinance 1.x returns flat
    columns there) and yf.download() as a secondary path with MultiIndex handling.
    """
    t = yf.Ticker(ticker)

    for period, interval in attempts:
        # Primary: .history() — flat columns, no MultiIndex headache.
        try:
            data = t.history(
                period=period,
                interval=interval,
                auto_adjust=True,
                actions=False,
            )
            if data is not None and not data.empty:
                data = data.dropna(how='all')
                if not data.empty:
                    return data
        except Exception:
            pass

        # Secondary: yf.download() with MultiIndex handling for yfinance 1.x.
        try:
            data = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    data = data.xs(ticker, axis=1, level=-1, drop_level=True)
                except Exception:
                    data.columns = [col[0] if isinstance(col, tuple) else col
                                    for col in data.columns]
            data = data.dropna(how='all')
            if not data.empty:
                return data
        except Exception:
            pass

    stooq_data = _download_from_stooq(ticker)
    if not stooq_data.empty:
        return stooq_data

    return pd.DataFrame()


def _get_active_stocks_from_session(request):
    """Return user-selected active stocks from session with validation and fallback."""
    stored = request.session.get('active_stocks', DEFAULT_ACTIVE_STOCKS)
    if not isinstance(stored, list):
        stored = DEFAULT_ACTIVE_STOCKS

    valid_tickers = _get_valid_tickers()
    cleaned = []
    for raw in stored:
        symbol = str(raw).strip().upper()
        if not symbol:
            continue
        if valid_tickers and symbol not in valid_tickers:
            continue
        if symbol not in cleaned:
            cleaned.append(symbol)
        if len(cleaned) >= 10:
            break

    if cleaned:
        return cleaned

    return DEFAULT_ACTIVE_STOCKS




# The Home page when Server loads up
def index(request):
    auth_redirect = _require_login(request)
    if auth_redirect:
        return auth_redirect

    watchlist = _get_active_stocks_from_session(request)

    # ================================================= Left Card Plot =========================================================
    # Use local CSV snapshot to avoid long page loads when external APIs are slow.
    ticker_df = pd.read_csv(_data_file('Tickers.csv'))
    ticker_df.columns = [
        'Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change',
        'Market_Cap', 'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry'
    ]
    ticker_df['Symbol'] = ticker_df['Symbol'].astype(str).str.upper()

    fig_left = go.Figure()
    end_date = dt.datetime.today()
    dates = [end_date - dt.timedelta(days=i) for i in reversed(range(7))]

    for ticker in watchlist:
        row = ticker_df[ticker_df['Symbol'] == ticker]
        if row.empty:
            continue

        item = row.iloc[0]
        last_sale = _safe_float(item.get('Last_Sale', 0.0), default=0.0)
        net_change = _safe_float(item.get('Net_Change', 0.0), default=0.0)

        # Generate a compact synthetic trend around the latest known snapshot.
        start = max(last_sale - net_change * 3, 0.01)
        series = [round(start + (net_change * i), 2) for i in range(7)]

        fig_left.add_trace(go.Scatter(x=dates, y=series, name=ticker))

    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')


    # ================================================ To show recent stocks ==============================================
    recent_tickers = watchlist
    recent_stocks = _build_recent_rows_from_csv(recent_tickers)

    # ========================================== Page Render section =====================================================

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks
    })

def search(request):
    auth_redirect = _require_login(request)
    if auth_redirect:
        return auth_redirect

    return render(request, 'search.html', {})

def ticker(request):
    auth_redirect = _require_login(request)
    if auth_redirect:
        return auth_redirect

    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv(_data_file('new_tickers.csv')) 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'ticker.html', {
        'ticker_list': ticker_list,
        'active_stocks': _get_active_stocks_from_session(request),
        'saved_count': request.GET.get('saved'),
        'invalid_count': request.GET.get('invalid'),
    })


def set_active_stocks(request):
    """Update active stocks from ticker page selection."""
    auth_redirect = _require_login(request)
    if auth_redirect:
        return auth_redirect

    if request.method != 'POST':
        return redirect('/ticker/')

    selected = request.POST.getlist('active_tickers')
    manual_input = request.POST.get('manual_tickers', '')
    manual_symbols = [s.strip().upper() for s in manual_input.split(',') if s.strip()]
    all_symbols = selected + manual_symbols

    valid_tickers = _get_valid_tickers()
    active = []
    invalid_count = 0

    for raw in all_symbols:
        symbol = str(raw).strip().upper()
        if not symbol:
            continue

        if valid_tickers and symbol not in valid_tickers:
            invalid_count += 1
            continue

        if symbol not in active:
            active.append(symbol)

        if len(active) >= 10:
            break

    if active:
        request.session['active_stocks'] = active
        request.session.modified = True
        return redirect('/ticker/?saved={}&invalid={}'.format(len(active), invalid_count))

    return redirect('/ticker/?saved=0&invalid={}'.format(invalid_count))


# The Predict Function to implement Machine Learning as well as Plotting
def predict(request, ticker_value, number_of_days):
    auth_redirect = _require_login(request)
    if auth_redirect:
        return auth_redirect

    # ticker_value = request.POST.get('ticker')
    ticker_value = _resolve_ticker_value(ticker_value)

    valid_tickers = _get_valid_tickers()
    if valid_tickers and ticker_value not in valid_tickers:
        return render(request, 'Invalid_Ticker.html', {})

    try:
        # number_of_days = request.POST.get('days')
        number_of_days = int(number_of_days)
    except:
        return render(request, 'Invalid_Days_Format.html', {})

    if number_of_days < 0:
        return render(request, 'Negative_Days.html', {})

    if number_of_days > 365:
        return render(request, 'Overflow_days.html', {})

    df = _download_with_fallback(
        ticker_value,
        [
            ('5d', '15m'),
            ('1mo', '1h'),
            ('3mo', '1d'),
        ],
    )
    if df.empty or not {'Open', 'High', 'Low', 'Close'}.issubset(set(df.columns)):
        return render(request, 'API_Down.html', {})

    print("Downloaded ticker = {} successfully".format(ticker_value))
    

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))
    fig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Stock Price (USD per Shares)')
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
        )
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')



    # ========================================== Machine Learning ==========================================


    df_ml = _download_with_fallback(
        ticker_value,
        [
            ('1y', '1d'),
            ('6mo', '1d'),
            ('3mo', '1d'),
        ],
    )

    if df_ml.empty:
        return render(request, 'API_Down.html', {})

    # Fetching ticker values from Yahoo Finance API 
    if 'Adj Close' in df_ml.columns:
        df_ml = df_ml[['Adj Close']]
    elif 'Close' in df_ml.columns:
        df_ml = df_ml[['Close']].rename(columns={'Close': 'Adj Close'})
    else:
        return render(request, 'API_Down.html', {})
    forecast_out = int(number_of_days)

    if forecast_out <= 0 or len(df_ml) <= forecast_out:
        return render(request, 'API_Down.html', {})

    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)
    # Splitting data for Test and Train
    X = np.array(df_ml.drop(['Prediction'], axis=1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_ml['Prediction'])
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
    # Applying Linear Regression
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    # Prediction Score
    confidence = clf.score(X_test, y_test)
    # Predicting for 'n' days stock data
    forecast_prediction = clf.predict(X_forecast)
    forecast = forecast_prediction.tolist()


    # ========================================== Plotting predicted data ======================================


    pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])
    
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    # ========================================== Display Ticker Info ==========================================

    ticker = pd.read_csv(_data_file('Tickers.csv'))
    to_search = ticker_value
    ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
                    'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
    for i in range(0,ticker.shape[0]):
        if ticker.Symbol[i] == to_search:
            Symbol = ticker.Symbol[i]
            Name = ticker.Name[i]
            Last_Sale = ticker.Last_Sale[i]
            Net_Change = ticker.Net_Change[i]
            Percent_Change = ticker.Percent_Change[i]
            Market_Cap = ticker.Market_Cap[i]
            Country = ticker.Country[i]
            IPO_Year = ticker.IPO_Year[i]
            Volume = ticker.Volume[i]
            Sector = ticker.Sector[i]
            Industry = ticker.Industry[i]
            break

    # ========================================== Page Render section ==========================================
    

    return render(request, "result.html", context={ 'plot_div': plot_div, 
                                                    'confidence' : confidence,
                                                    'forecast': forecast,
                                                    'ticker_value':ticker_value,
                                                    'number_of_days':number_of_days,
                                                    'plot_div_pred':plot_div_pred,
                                                    'Symbol':Symbol,
                                                    'Name':Name,
                                                    'Last_Sale':Last_Sale,
                                                    'Net_Change':Net_Change,
                                                    'Percent_Change':Percent_Change,
                                                    'Market_Cap':Market_Cap,
                                                    'Country':Country,
                                                    'IPO_Year':IPO_Year,
                                                    'Volume':Volume,
                                                    'Sector':Sector,
                                                    'Industry':Industry,
                                                    })


def users(request):
    auth_redirect = _require_login(request)
    if auth_redirect:
        return auth_redirect

    team = [
        {'name': 'N Sreenivasulu', 'initials': 'NS', 'color': '#eb1616', 'bg': 'rgba(235,22,22,0.15)', 'border': 'rgba(235,22,22,0.3)'},
        {'name': 'P Jaswanth',     'initials': 'PJ', 'color': '#4f8ef7', 'bg': 'rgba(79,142,247,0.15)', 'border': 'rgba(79,142,247,0.3)'},
        {'name': 'B Karthik',      'initials': 'BK', 'color': '#00d084', 'bg': 'rgba(0,208,132,0.15)', 'border': 'rgba(0,208,132,0.3)'},
        {'name': 'B Simhadri',     'initials': 'BS', 'color': '#f7a74f', 'bg': 'rgba(247,167,79,0.15)', 'border': 'rgba(247,167,79,0.3)'},
    ]

    return render(request, 'users.html', {
        'team': team,
        'current_username': _current_username(request),
    })
