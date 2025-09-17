import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime, timezone, timedelta

# Local timezone (India Standard Time, UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

symbol = "btcusdt"
url = f"wss://stream.binance.com:9443/ws/{symbol}@bookTicker"
url2 = f"wss://stream.binance.com:9443/ws/{symbol}@depth"
url3 = f"wss://stream.binance.com:9443/ws/{symbol}@aggTrade"

data = []
seq = 0

async def collect_orderbook():
    global seq
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            msg = json.loads(msg)
            seq += 1
            now = datetime.now(IST) 

            data.append({
                "seq": seq,
                "time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "bid_price": float(msg["b"]),
                "bid_size": float(msg["B"]),
                "ask_price": float(msg["a"]),
                "ask_size": float(msg["A"])
            })

            if seq % 100 == 0:
                df = pd.DataFrame(data)
                df.to_csv("orderbook.csv", index=False)
                print(f"Saved {len(data)} rows (last time {now})")

asyncio.run(collect_orderbook())
