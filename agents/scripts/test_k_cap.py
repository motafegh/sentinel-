# agents/scripts/test_k_cap.py
"""Simple test for k cap - run this separately"""

import asyncio
import json
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

async def test_k_cap():
    url = "http://localhost:8011/sse"
    
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Test with k=99
            print("Testing search with k=99...")
            result = await session.call_tool(
                "search",
                {"query": "access control vulnerability", "k": 99}
            )
            
            print(f"Result content length: {len(result.content)}")
            if result.content:
                raw = result.content[0].text
                print(f"Raw response (first 200 chars): {raw[:200]}")
                
                try:
                    data = json.loads(raw)
                    print(f"Success! k_returned: {data.get('k_returned')}")
                    print(f"k_requested: {data.get('k_requested')}")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Raw response: {raw}")
            else:
                print("No content in response!")

if __name__ == "__main__":
    asyncio.run(test_k_cap())