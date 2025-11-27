from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import base64
from io import BytesIO

app = FastAPI(title="TA Engine with Charts")

# ----------- Request Model ----------- #
class TARequest(BaseModel):
    symbol: str
    interval: str = "1d"
    lookback: int = 365

# ----------- Indicator Functions ----------- #
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

# ----------- Chart Generator ----------- #
def generate_chart(df):
    df = df.copy()
    df.set_index("Date", inplace=True)

    # Add overlays
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)

    # Create additional panels
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])

    # Plot style
    style = mpf.make_mpf_style(base_mpf_style="yahoo")

    # Buffer for image
    buf = BytesIO()

    mpf.plot(
        df,
        type="candle",
        style=style,
        volume=True,
        mav=(20, 50, 200),
        addplot=[
            mpf.make_addplot(df["RSI"], panel=1, color="purple", ylabel="RSI"),
            mpf.make_addplot(df["MACD"], panel=2, color="blue"),
            mpf.make_addplot(df["MACD_signal"], panel=2, color="red"),
            mpf.make_addplot(df["MACD_hist"], type="bar", panel=2, color="dimgray"),
        ],
        figsize=(14, 10),
        tight_layout=True,
        savefig=dict(fname=buf, dpi=100, pad_inches=0.2),
    )

    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64

# ----------- FastAPI Endpoint ----------- #
@app.post("/analyze")
def analyze(req: TARequest):
    try:
        df = yf.download(req.symbol, period=f"{req.lookback}d", interval=req.interval)
        df = df.reset_index()
        df = df.rename(columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        })
        if df.empty:
            raise HTTPException(400, "No data for symbol.")
    except Exception as e:
        raise HTTPException(500, str(e))

    # Generate the chart image
    chart_image = generate_chart(df)

    # Latest indicators
    ema20 = ema(df["Close"], 20).iloc[-1]
    ema50 = ema(df["Close"], 50).iloc[-1]
    ema200 = ema(df["Close"], 200).iloc[-1]
    rsi_val = rsi(df["Close"]).iloc[-1]
    macd_line, macd_signal, macd_hist = macd(df["Close"])

    return {
        "symbol": req.symbol,
        "latest_close": float(df["Close"].iloc[-1]),
        "ema20": float(ema20),
        "ema50": float(ema50),
        "ema200": float(ema200),
        "rsi": float(rsi_val),
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(macd_signal.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "chart_image_base64": chart_image
    }

@app.get("/")
def home():
    return {"message": "TA Engine with chart generation is running"}
# main.py
"""
FastAPI app to render candlestick charts with EMA20/50/200, RSI, and MACD.
Default data provider: yfinance (no API keys).
Implements in-memory TTL cache to reduce rate limits.
Produces PNG images and also supports JSON outputs.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from cachetools import TTLCache, cached
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import io
import logging
from datetime import datetime

app = FastAPI(
    title="Stock Charting API",
    description="Generate candlestick charts with EMA20/50/200, RSI, and MACD.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-chart-service")

DATA_CACHE = TTLCache(maxsize=512, ttl=300)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14).mean()
    roll_down = down.ewm(span=14).mean()
    rs = roll_up / roll_down
    df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))
    
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    
    return df

@cached(DATA_CACHE, key=lambda t, i, s, e, p: f"{t}|{i}|{p}|{s}|{e}")
def fetch_ohlc(ticker: str, interval: str = "1d", start=None, end=None, period=None):
    try:
        tk = yf.Ticker(ticker)
        if start and end:
            df = tk.history(interval=interval, start=start, end=end, auto_adjust=False)
        elif period:
            df = tk.history(interval=interval, period=period, auto_adjust=False)
        else:
            df = tk.history(interval=interval, period="1mo", auto_adjust=False)
        
        if df.empty:
            return None
        
        df.columns = [c.capitalize() for c in df.columns]
        if "Close" not in df.columns and "Adj close" in df.columns:
            df["Close"] = df["Adj close"]
        
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = compute_indicators(df)
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

def render_chart_png(df, ticker, width=1200, height=600, dpi=100, show_rsi=True, show_macd=True):
    if df.empty:
        raise ValueError("Empty OHLC data")
    
    apds = [
        mpf.make_addplot(df["EMA20"], width=1.0),
        mpf.make_addplot(df["EMA50"], width=1.0),
        mpf.make_addplot(df["EMA200"], width=1.0),
    ]
    
    panels = []
    panel_count = 1
    if show_rsi:
        panels.append(mpf.make_addplot(df["RSI14"], panel=1, ylabel="RSI"))
        panel_count += 1
    if show_macd:
        macd_panel = 1 if show_rsi else 1
        panels.append(mpf.make_addplot(df["MACD"], panel=macd_panel, ylabel="MACD"))
        panels.append(mpf.make_addplot(df["MACD_SIGNAL"], panel=macd_panel))
        panels.append(mpf.make_addplot(df["MACD_HIST"], type="bar", panel=macd_panel))
    
    all_addplots = apds + panels
    
    fig_width = max(6, width / dpi)
    fig_height = max(4, height / dpi)
    
    buf = io.BytesIO()
    mpf.plot(
        df,
        type="candle",
        style="charles",
        mav=(20, 50, 200),
        volume=True,
        addplot=all_addplots,
        figscale=1,
        figsize=(fig_width, fig_height),
        savefig=dict(fname=buf, dpi=dpi, bbox_inches="tight"),
        warn_too_much_data=10000
    )
    buf.seek(0)
    return buf.read()

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/chart/{ticker}", response_class=StreamingResponse, tags=["charts"])
def get_chart(
    ticker: str = Query(..., description="Ticker symbol, e.g. AAPL or TCS.NS"),
    interval: str = Query("1d", description="Data interval (yfinance format): 1d, 1wk, 1mo, 60m, 30m, 15m, 5m"),
    period: str = Query(None, description="Period shorthand (e.g. 1mo, 6mo, 1y) if start/end not provided"),
    start: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end: str = Query(None, description="End date (YYYY-MM-DD)"),
    width: int = Query(1200, ge=600, le=2400),
    height: int = Query(600, ge=300, le=1200),
    dpi: int = Query(100, ge=50, le=200),
    show_rsi: bool = Query(True),
    show_macd: bool = Query(True),
):
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")
    
    try:
        df = fetch_ohlc(ticker=ticker, interval=interval, start=start, end=end, period=period)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data available for ticker/period")
        
        img_bytes = render_chart_png(df, ticker, width=width, height=height, dpi=dpi, show_rsi=show_rsi, show_macd=show_macd)
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart render failed: {e}")
        raise HTTPException(status_code=500, detail="Chart rendering error")

@app.get("/data/{ticker}", tags=["data"])
def get_data(
    ticker: str = Query(..., description="Ticker symbol"),
    interval: str = Query("1d"),
    period: str = Query(None),
    start: str = Query(None),
    end: str = Query(None),
    limit: int = Query(500, ge=1, le=5000)
):
    ticker = ticker.strip().upper()
    try:
        df = fetch_ohlc(ticker=ticker, interval=interval, start=start, end=end, period=period)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data available for ticker/period")
        
        out = df.tail(limit).reset_index()
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        data = out.to_dict(orient="records")
        return {"ticker": ticker, "rows": len(data), "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        raise HTTPException(status_code=502, detail="Data provider error")

@app.get("/", tags=["root"])
def root():
    return {
        "service": "Stock Charting API",
        "version": "1.0.0",
        "endpoints": ["/chart/{ticker}", "/data/{ticker}", "/health"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
