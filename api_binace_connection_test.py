import asyncio
from binance import AsyncClient

async def main():
    print("Connecting...")
    client = await AsyncClient.create()
    print("Connected!")
    print(await client.ping())
    await client.close_connection()

asyncio.run(main())