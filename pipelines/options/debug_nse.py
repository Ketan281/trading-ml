import requests
import json

headers = {
    "User-Agent"     : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept"         : "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer"        : "https://www.nseindia.com/option-chain",
    "X-Requested-With": "XMLHttpRequest"
}

session = requests.Session()

# Step 1 — Hit homepage first
print("  Step 1 — Getting cookies...")
session.get(
    "https://www.nseindia.com",
    headers=headers,
    timeout=15
)

# Step 2 — Hit option chain page
print("  Step 2 — Getting option chain page...")
session.get(
    "https://www.nseindia.com/option-chain",
    headers=headers,
    timeout=15
)

# Step 3 — Fetch actual data
print("  Step 3 — Fetching NIFTY data...")
url      = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
response = session.get(url, headers=headers, timeout=15)

print(f"  Status Code : {response.status_code}")
print(f"  Response Keys: {list(response.json().keys())}")

# Show full structure
data = response.json()
print(f"\n  Full structure:")
print(json.dumps(
    {k: type(v).__name__ for k, v in data.items()},
    indent=2
))

# Show first level deeper
for key in data:
    val = data[key]
    if isinstance(val, dict):
        print(f"\n  '{key}' contains: {list(val.keys())}")
    elif isinstance(val, list):
        print(f"\n  '{key}' is list of {len(val)} items")
        if val:
            print(f"  First item keys: {list(val[0].keys()) if isinstance(val[0], dict) else val[0]}")