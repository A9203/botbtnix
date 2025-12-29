import requests
import pandas as pd
import talib
import time
import streamlit as st
import threading
import hashlib
import uuid
import json
from streamlit_option_menu import option_menu
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go


# ========================
# CONFIGURACIÓN DEL BOT
# ========================
TIMEFRAME = '15m'
EMAS = [5, 14, 21, 34, 50, 100]
ATR_PERIOD = 21
ATR_MULTIPLIER = 1.2
VOLUME_MULTIPLIER = 1.5
RISK_PER_TRADE = 0.01  # 1%
PARTIAL_CLOSE = 0.4   # 40%
TRAILING_BTC_ETH = 0.0035  # 0.35%
TRAILING_ALT = 0.006      # 0.6%
MAX_TRADES = 2
BASE_URL = 'https://fapi.bitunix.com'


# Variables globales para credenciales
api_key = None
secret = None


# ========================
# FUNCIONES API BITUNIX
# ========================
def public_request(method, path, params=None):
    url = BASE_URL + path
    if params:
        query_str = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
        url += '?' + query_str
    response = requests.request(method, url)
    return response.json()


def private_request(method, path, params=None, data=None):
    query_str = ''
    if method == 'GET' and params:
        query_str = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
    body_str = json.dumps(data, separators=(',', ':')) if data else ''
    nonce = uuid.uuid4().hex
    timestamp = str(int(time.time() * 1000))
    digest_input = nonce + timestamp + api_key + query_str + body_str
    digest = hashlib.sha256(digest_input.encode()).hexdigest()
    sign_input = digest + secret
    sign = hashlib.sha256(sign_input.encode()).hexdigest()


    headers = {
        'api-key': api_key,
        'nonce': nonce,
        'timestamp': timestamp,
        'sign': sign,
        'Content-Type': 'application/json',
        'language': 'en-US'
    }
    url = BASE_URL + path
    if query_str:
        url += '?' + query_str
    response = requests.request(method, url, headers=headers, json=data)
    return response.json()


# ========================
# DATOS E INDICADORES
# ========================
def fetch_data(symbol, limit=300):
    path = '/api/v1/futures/market/kline'
    params = {'symbol': symbol, 'interval': TIMEFRAME, 'limit': limit}
    resp = public_request('GET', path, params)
    if resp.get('code') != 0:
        raise Exception(resp.get('msg'))
    data = resp['data']
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'})
    df['volume'] = df['quoteVol'].astype(float)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
    return df


def calculate_indicators(df):
    for p in EMAS:
        df[f'ema_{p}'] = talib.EMA(df['close'], timeperiod=p)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_PERIOD)
    df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
    df['ema21_slope'] = df['ema_21'] - df['ema_21'].shift(1)
    return df


def is_trending(df):
    r = df.iloc[-1]
    if (r['ema_5'] > r['ema_14'] > r['ema_21'] > r['ema_34'] > r['ema_50'] > r['ema_100'] and
        r['close'] > r['ema_21'] and r['close'] > r['ema_50'] and r['ema21_slope'] > 0):
        return 'up'
    elif (r['ema_5'] < r['ema_14'] < r['ema_21'] < r['ema_34'] < r['ema_50'] < r['ema_100'] and
          r['close'] < r['ema_21'] and r['close'] < r['ema_50'] and r['ema21_slope'] < 0):
        return 'down'
    return None


def has_momentum(df):
    last3 = df.iloc[-4:-1]
    last = df.iloc[-1]
    break_high = last['close'] > last3['high'].max()
    break_low = last['close'] < last3['low'].min()
    vol_ok = last['volume'] >= VOLUME_MULTIPLIER * last['volume_sma']
    body_ok = abs(last['close'] - last['open']) >= 0.7 * (last['high'] - last['low'])
    atr_avg = df['atr'].iloc[-51:-1].mean()
    atr_ok = last['atr'] > atr_avg
    if break_high and vol_ok and body_ok and atr_ok:
        return 'up'
    if break_low and vol_ok and body_ok and atr_ok:
        return 'down'
    return None


# ========================
# BACKTEST HISTÓRICO
# ========================
def fetch_historical_data(symbol, start_date, end_date):
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date).timestamp() * 1000)
    all_data = []
    current = end_ms
    while current > start_ms:
        params = {'symbol': symbol, 'interval': TIMEFRAME, 'limit': 1000, 'endTime': current}
        resp = public_request('GET', '/api/v1/futures/market/kline', params)
        if resp.get('code') != 0 or not resp['data']:
            break
        batch = resp['data']
        all_data = batch + all_data
        current = batch[0]['time'] - 1
    df = pd.DataFrame(all_data)
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    df['volume'] = df['quoteVol'].astype(float)
    df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'})
    df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    return df


def backtest_strategy(df, initial_balance, symbol):
    df = calculate_indicators(df.copy())
    balance = initial_balance
    trades = []
    in_position = False
    side = None
    entry = 0
    qty = 0
    sl = 0
    tp1_hit = False


    for i in range(max(EMAS + [ATR_PERIOD, 50]), len(df)):
        sub_df = df.iloc[:i+1]
        row = df.iloc[i]
        trend = is_trending(sub_df)
        momentum = has_momentum(sub_df)


        if not in_position and trend and momentum and trend == momentum:
            atr = row['atr']
            risk = balance * RISK_PER_TRADE
            side = 'BUY' if trend == 'up' else 'SELL'
            entry = row['close']
            sl = entry - ATR_MULTIPLIER * atr if side == 'BUY' else entry + ATR_MULTIPLIER * atr
            qty = risk / abs(entry - sl)
            in_position = True
            tp1_hit = False


        if in_position:
            price = row['close']
            if (side == 'BUY' and price <= sl) or (side == 'SELL' and price >= sl):
                pl = (price - entry) * qty if side == 'BUY' else (entry - price) * qty
                balance += pl
                trades.append({'time': row['timestamp'], 'side': side, 'entry': entry, 'exit': price, 'pl': pl, 'reason': 'SL'})
                in_position = False
                continue


            tp1 = entry + ATR_MULTIPLIER * atr if side == 'BUY' else entry - ATR_MULTIPLIER * atr
            if not tp1_hit and ((side == 'BUY' and price >= tp1) or (side == 'SELL' and price <= tp1)):
                partial_qty = qty * PARTIAL_CLOSE
                pl_partial = (price - entry) * partial_qty if side == 'BUY' else (entry - price) * partial_qty
                balance += pl_partial
                qty -= partial_qty
                tp1_hit = True
                sl = entry


            if tp1_hit:
                trailing = TRAILING_BTC_ETH if 'BTC' in symbol or 'ETH' in symbol else TRAILING_ALT
                new_sl = price * (1 - trailing) if side == 'BUY' else price * (1 + trailing)
                if (side == 'BUY' and new_sl > sl) or (side == 'SELL' and new_sl < sl):
                    sl = new_sl


            if (side == 'BUY' and row['ema_5'] < row['ema_21']) or (side == 'SELL' and row['ema_5'] > row['ema_21']):
                pl = (price - entry) * qty if side == 'BUY' else (entry - price) * qty
                balance += pl
                trades.append({'time': row['timestamp'], 'side': side, 'entry': entry, 'exit': price, 'pl': pl, 'reason': 'EMA Cross'})
                in_position = False


    return pd.DataFrame(trades), balance


# ========================
# EMAIL ALERTS
# ========================
def send_email(subject, body, to_email, smtp_server, smtp_port, smtp_user, smtp_pass):
    if not to_email:
        return
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        st.warning(f"Email error: {e}")


# ========================
# OPERACIONES EN VIVO
# ========================
def get_current_price(symbol):
    resp = public_request('GET', '/api/v1/futures/market/tickers', {'symbols': symbol})
    if resp.get('code') == 0 and resp['data']:
        return float(resp['data'][0]['lastPrice'])
    raise Exception("Price error")


def get_balance():
    resp = private_request('GET', '/api/v1/futures/account', {'marginCoin': 'USDT'})
    if resp.get('code') == 0 and resp['data']:
        return float(resp['data'][0]['available'])
    return 0.0


def get_positions(symbol):
    resp = private_request('GET', '/api/v1/futures/position/get_pending_positions', {'symbol': symbol})
    if resp.get('code') == 0:
        return resp['data']
    return []


def place_order(symbol, qty, side, order_type, trade_side, reduce_only=False, sl_price=None):
    data = {
        'symbol': symbol,
        'qty': str(qty),
        'side': side,
        'tradeSide': trade_side,
        'orderType': order_type,
        'reduceOnly': reduce_only
    }
    if sl_price:
        data['slPrice'] = str(sl_price)
        data['slStopType'] = 'LAST_PRICE'
        data['slOrderType'] = 'MARKET'
    resp = private_request('POST', '/api/v1/futures/trade/place_order', data=data)
    if resp.get('code') != 0:
        raise Exception(resp.get('msg'))


def modify_sl(position_id, sl_price):
    data = {'positionId': position_id, 'slPrice': str(sl_price), 'slStopType': 'LAST_PRICE', 'slOrderType': 'MARKET'}
    private_request('POST', '/api/v1/futures/tpsl/position/modify_order', data=data)


def close_position(symbol, pos, price, reason, log_container, email_params):
    side_close = 'SELL' if pos['side'] == 'BUY' else 'BUY'
    place_order(symbol, pos['qty'], side_close, 'MARKET', 'CLOSE', reduce_only=True)
    pl = (price - pos['entry']) * pos['qty'] if pos['side'] == 'BUY' else (pos['entry'] - price) * pos['qty']
    trade = {**pos, 'exit': price, 'pl': pl, 'reason': reason, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    st.session_state.trades.append(trade)
    log_container.write(f"[{trade['timestamp']}] CLOSED {symbol} {pos['side']} P/L: {pl:.2f} ({reason})")
    send_email("Trade Closed", f"{symbol} {pos['side']} closed at {price}\nP/L: {pl:.2f}\nReason: {reason}", **email_params)


def bot_loop(symbol, leverage, log_c, status_c, bal_c, pos_c, email_params):
    global api_key, secret
    private_request('POST', '/api/v1/futures/account/change_leverage', data={'marginCoin': 'USDT', 'symbol': symbol, 'leverage': leverage})
    positions = {}


    while st.session_state.bot_running:
        try:
            df = fetch_data(symbol)
            df = calculate_indicators(df)
            trend = is_trending(df)
            momentum = has_momentum(df)


            api_pos = get_positions(symbol)
            if api_pos:
                p = api_pos[0]
                positions[symbol] = {
                    'position_id': p['positionId'],
                    'entry': float(p['avgOpenPrice']),
                    'sl': positions.get(symbol, {}).get('sl', 0),
                    'side': 'BUY' if p['side'] == 'LONG' else 'SELL',
                    'qty': float(p['qty']),
                    'tp1_hit': positions.get(symbol, {}).get('tp1_hit', False)
                }
            else:
                positions.pop(symbol, None)


            if trend and momentum and trend == momentum and len(positions) < MAX_TRADES and symbol not in positions:
                price = get_current_price(symbol)
                atr = df.iloc[-1]['atr']
                balance = get_balance()
                risk = balance * RISK_PER_TRADE
                side = 'BUY' if trend == 'up' else 'SELL'
                sl = price - ATR_MULTIPLIER * atr if side == 'BUY' else price + ATR_MULTIPLIER * atr
                qty = risk / abs(price - sl)
                place_order(symbol, qty, side, 'MARKET', 'OPEN', sl_price=sl)
                log_c.write(f"[{datetime.now().strftime('%H:%M:%S')}] OPENED {side} {symbol} @ {price}")
                send_email("New Position", f"Opened {side} {symbol} at {price}\nSL: {sl}", **email_params)
                time.sleep(5)
                positions[symbol] = {'entry': price, 'sl': sl, 'side': side, 'qty': qty, 'tp1_hit': False}


            if symbol in positions:
                pos = positions[symbol]
                price = get_current_price(symbol)
                atr = df.iloc[-1]['atr']


                if (pos['side'] == 'BUY' and price <= pos['sl']) or (pos['side'] == 'SELL' and price >= pos['sl']):
                    close_position(symbol, pos, price, 'SL Hit', log_c, email_params)
                    del positions[symbol]
                    continue


                tp1 = pos['entry'] + ATR_MULTIPLIER * atr if pos['side'] == 'BUY' else pos['entry'] - ATR_MULTIPLIER * atr
                if not pos['tp1_hit'] and ((pos['side'] == 'BUY' and price >= tp1) or (pos['side'] == 'SELL' and price <= tp1)):
                    close_qty = pos['qty'] * PARTIAL_CLOSE
                    close_side = 'SELL' if pos['side'] == 'BUY' else 'BUY'
                    place_order(symbol, close_qty, close_side, 'MARKET', 'CLOSE', reduce_only=True)
                    pos['qty'] -= close_qty
                    pos['tp1_hit'] = True
                    modify_sl(pos['position_id'], pos['entry'])
                    pos['sl'] = pos['entry']
                    log_c.write(f"TP1 Hit - Partial close & SL to BE")
                    send_email("TP1 Hit", f"Partial close {symbol} {pos['side']}", **email_params)


                if pos['tp1_hit']:
                    trailing = TRAILING_BTC_ETH if 'BTC' in symbol or 'ETH' in symbol else TRAILING_ALT
                    new_sl = price * (1 - trailing) if pos['side'] == 'BUY' else price * (1 + trailing)
                    if (pos['side'] == 'BUY' and new_sl > pos['sl']) or (pos['side'] == 'SELL' and new_sl < pos['sl']):
                        modify_sl(pos['position_id'], new_sl)
                        pos['sl'] = new_sl


                if (pos['side'] == 'BUY' and df.iloc[-1]['ema_5'] < df.iloc[-1]['ema_21']) or (pos['side'] == 'SELL' and df.iloc[-1]['ema_5'] > df.iloc[-1]['ema_21']):
                    close_position(symbol, pos, price, 'EMA Cross', log_c, email_params)
                    del positions[symbol]


            # Dashboard update
            status_c.metric("Status", "Running" if st.session_state.bot_running else "Stopped")
            bal_c.metric("Available Balance", f"{get_balance():.2f} USDT")
            pos_text = "\n".join([f"{s}: {p['side']} @ {p['entry']:.4f} (Qty: {p['qty']})" for s,p in positions.items()]) or "None"
            pos_c.text("Open Positions\n" + pos_text)


            time.sleep(60)
        except Exception as e:
            log_c.error(f"Error: {e}")
            time.sleep(60)


# ========================
# INTERFAZ STREAMLIT
# ========================
st.set_page_config(page_title="Bitunix EMA Ribbon Bot", layout="wide", page_icon="🤖")


st.markdown("""
<style>
    .main {background-color: #0e1117; color: #fafafa;}
    .stButton>button {background-color: #1f77b4; color: white; border-radius: 8px;}
    .metric-card {background-color: #262730; padding: 15px; border-radius: 10px;}
    h1, h2, h3 {color: #1f77b4;}
</style>
""", unsafe_allow_html=True)


st.title("🤖 Bitunix Futures EMA Ribbon Trading Bot")
st.markdown("**Estrategia profesional con EMA 5-14-21-34-50-100 + Volumen + ATR(21)** – Timeframe 15m")


with st.sidebar:
    st.header("Navegación")
    selected = option_menu(
        "Menú Principal",
        ["Dashboard", "Configuración", "Backtest", "Rendimiento", "Logs"],
        icons=['house', 'gear', 'clock-history', 'bar-chart', 'journal'],
        menu_icon="cast",
        default_index=0
    )


if selected == "Dashboard":
    col1, col2, col3 = st.columns(3)
    status_c = col1.empty()
    bal_c = col2.empty()
    pos_c = col3.empty()


    if st.session_state.get('bot_running', False):
        st.success("Bot en ejecución")
    else:
        st.warning("Bot detenido")


elif selected == "Configuración":
    st.header("Configuración del Bot")
    with st.expander("Credenciales API Bitunix", expanded=True):
        global api_key, secret
        api_key = st.text_input("API Key", type="password")
        secret = st.text_input("API Secret", type="password")


    with st.expander("Parámetros de Trading"):
        symbol = st.selectbox("Par", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"])
        leverage = st.slider("Apalancamiento", 1, 125, 10)
        st.session_state.initial_balance = st.number_input("Balance inicial (para métricas)", value=10000.0)


    with st.expander("Alertas por Email"):
        to_email = st.text_input("Email destinatario")
        smtp_server = st.text_input("Servidor SMTP", "smtp.gmail.com")
        smtp_port = st.number_input("Puerto SMTP", value=587)
        smtp_user = st.text_input("Usuario SMTP")
        smtp_pass = st.text_input("Contraseña SMTP", type="password")


    col1, col2 = st.columns(2)
    if col1.button("Iniciar Bot", type="primary") and not st.session_state.get('bot_running', False):
        if api_key and secret:
            st.session_state.bot_running = True
            log_c = st.container()
            status_c, bal_c, pos_c = st.columns(3)
            email_params = {'to_email': to_email, 'smtp_server': smtp_server, 'smtp_port': smtp_port,
                       'smtp_user': smtp_user, 'smtp_pass': smtp_pass}
            threading.Thread(target=bot_loop, args=(symbol, leverage, log_c, status_c, bal_c, pos_c, email_params), daemon=True).start()
            st.success("Bot iniciado correctamente")
        else:
            st.error("Introduce credenciales API")


    if col2.button("Detener Bot") and st.session_state.get('bot_running', False):
        st.session_state.bot_running = False
        st.success("Bot detenido")


elif selected == "Backtest":
    st.header("Backtesting Histórico")
    col1, col2 = st.columns(2)
    symbol_bt = col1.selectbox("Par", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    start_bt = col1.date_input("Fecha inicio", datetime(2024, 1, 1))
    end_bt = col2.date_input("Fecha fin", datetime.now())
    init_bal_bt = col2.number_input("Balance inicial", value=10000.0)


    if st.button("Ejecutar Backtest"):
        with st.spinner("Descargando datos y ejecutando backtest..."):
            df_hist = fetch_historical_data(symbol_bt, str(start_bt), str(end_bt))
            if df_hist.empty:
                st.error("No se pudieron obtener datos históricos")
            else:
                trades_bt, final_bal = backtest_strategy(df_hist, init_bal_bt, symbol_bt)
                ret = (final_bal - init_bal_bt) / init_bal_bt * 100
                st.success(f"Backtest completado | Balance final: {final_bal:.2f} USDT (+{ret:.2f}%)")
                st.dataframe(trades_bt)


                equity = [init_bal_bt]
                for p in trades_bt['pl']:
                    equity.append(equity[-1] + p)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(equity))), y=equity, mode='lines', name='Equity'))
                fig.update_layout(title="Curva de Capital", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)


elif selected == "Rend