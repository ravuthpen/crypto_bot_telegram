# 🚀 Crypto Bot Telegram

An AI-powered Binance Futures trading assistant that analyzes the market, generates trading signals using technical indicators and AI, and sends alerts directly to Telegram.

> This bot provides trading signals only. It does NOT place trades automatically unless auto-trading is explicitly enabled.

---

# Features

- Real-time Binance Futures market data
- AI-powered trade analysis
- Technical indicators
  - RSI
  - EMA
  - SMA
  - MACD
  - Bollinger Bands
  - ATR
  - Volume Analysis
- Long / Short recommendations
- Risk Management
  - Stop Loss
  - Take Profit
  - Risk/Reward Ratio
- Telegram notifications
- Trade logging
- Docker support
- Scheduled market scanning

---

# Technologies

- Python 3.12+
- Binance Futures API
- OpenAI API
- Telegram Bot API
- pandas
- numpy
- ta
- APScheduler
- Docker
- MySQL / SQLite

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/crypto_bot_telegram.git

cd crypto_bot_telegram
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file.

```env
BINANCE_API_KEY=
BINANCE_SECRET_KEY=

OPENAI_API_KEY=

TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

DATABASE_URL=mysql+pymysql://user:password@localhost/trading

SYMBOL=BTCUSDT

INTERVAL=15m

LEVERAGE=10

RISK_PERCENT=2
```

---

# Trading Strategy

The bot evaluates multiple indicators before generating a signal.

## Long Conditions

- RSI oversold
- Price above EMA50
- MACD bullish crossover
- Increasing volume
- AI confidence > 80%

## Short Conditions

- RSI overbought
- Price below EMA50
- MACD bearish crossover
- Strong selling volume
- AI confidence > 80%

---

# Telegram Signal Example

```
LONG SIGNAL

Symbol: BTCUSDT

Entry:
107,250

Stop Loss:
106,700

Take Profit 1:
108,000

Take Profit 2:
108,600

Risk Reward:
1 : 2.8

Confidence:
91%

Reason

• RSI Oversold
• EMA Bullish
• MACD Cross
• High Volume
• AI confirms trend

Time:
2026-06-29 10:30 UTC
```

---

# Risk Management

- Maximum 2% account risk per trade
- Always use Stop Loss
- Minimum Risk/Reward ratio of 1:2
- Skip low-confidence trades
- Avoid trading during extreme volatility unless confirmed

---

# Roadmap

- [ ] Auto Trading
- [ ] Multi-symbol scanning
- [ ] Position management
- [ ] News sentiment analysis
- [ ] Whale alert integration
- [ ] Trading dashboard
- [ ] Backtesting engine
- [ ] Web dashboard
- [ ] Performance analytics
- [ ] Portfolio tracking

---

# Disclaimer

Trading cryptocurrencies involves substantial risk. This software is intended for educational and research purposes only. Always perform your own analysis before making trading decisions. The authors are not responsible for any financial losses.

---

# License

License
This project is licensed under the MIT License.

Copyright (c) 2026 Ravuth Pen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

# Author

Ravuth Pen

AI Futures Trader

Python • Binance Futures • Telegram