from __future__ import annotations
import asyncio
from dotenv import load_dotenv
from rich.console import Console
from trading_bot import TradingBot
from config import BotConfig
load_dotenv()
console = Console()

def main():
    bot = TradingBot(BotConfig())
    asyncio.run(bot.run())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
