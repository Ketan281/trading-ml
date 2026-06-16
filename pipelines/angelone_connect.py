import os
import sys
import pyotp
import logging
from dotenv        import load_dotenv
from SmartApi      import SmartConnect
from datetime      import datetime

# ── Load Credentials ──────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

API_KEY     = os.getenv("ANGELONE_API_KEY")
API_SECRET  = os.getenv("ANGELONE_API_SECRET")
CLIENT_ID   = os.getenv("ANGELONE_CLIENT_ID")
PASSWORD    = os.getenv("ANGELONE_PASSWORD")
TOTP_SECRET = os.getenv("ANGELONE_TOTP_SECRET")

# ── Logging ───────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("AngelOne")

# ── Connection Class ──────────────────────────────────
class AngelOneConnection:
    def __init__(self):
        self.smart_api   = None
        self.auth_token  = None
        self.feed_token  = None
        self.refresh_token = None
        self.connected   = False

    # ── Login ─────────────────────────────────────────
    def login(self):
        try:
            log.info("  Connecting to AngelOne Smart API...")

            # Validate credentials
            if not all([API_KEY, CLIENT_ID,
                        PASSWORD, TOTP_SECRET]):
                raise ValueError(
                    "Missing credentials in .env file"
                )

            # Generate TOTP
            totp     = pyotp.TOTP(TOTP_SECRET)
            otp      = totp.now()
            log.info(f"  Generated OTP : {otp}")

            # Initialize Smart API
            self.smart_api = SmartConnect(api_key=API_KEY)

            # Login
            data = self.smart_api.generateSession(
                CLIENT_ID, PASSWORD, otp
            )

            if data["status"] == False:
                raise Exception(
                    f"Login failed: {data['message']}"
                )

            # Store tokens
            self.auth_token    = data["data"]["jwtToken"]
            self.refresh_token = data["data"]["refreshToken"]
            self.feed_token    = self.smart_api.getfeedToken()
            self.connected     = True

            log.info("  ✅ Login successful!")
            log.info(f"  Client     : {CLIENT_ID}")
            log.info(
                f"  Auth Token : "
                f"{self.auth_token[:20]}..."
            )
            return True

        except Exception as e:
            log.error(f"  ❌ Login failed: {e}")
            self.connected = False
            return False

    # ── Get Profile ───────────────────────────────────
    def get_profile(self):
        if not self.connected:
            log.error("  Not connected. Call login() first.")
            return None
        try:
            profile = self.smart_api.getProfile(
                self.refresh_token
            )
            if profile["status"]:
                data = profile["data"]
                log.info(f"  Name    : {data['name']}")
                log.info(f"  Email   : {data['email']}")
                log.info(f"  Broker  : {data['broker']}")
                return data
        except Exception as e:
            log.error(f"  ❌ Profile fetch failed: {e}")
        return None

    # ── Logout ────────────────────────────────────────
    def logout(self):
        try:
            if self.smart_api and self.connected:
                self.smart_api.terminateSession(CLIENT_ID)
                self.connected = False
                log.info("  ✅ Logged out successfully")
        except Exception as e:
            log.error(f"  ❌ Logout failed: {e}")

    # ── Get Funds ─────────────────────────────────────
    def get_funds(self):
        if not self.connected:
            return None
        try:
            funds = self.smart_api.rmsLimit()
            if funds["status"]:
                data = funds["data"]
                log.info("  💰 Fund Details:")
                log.info(
                    f"     Available Cash : "
                    f"₹{data.get('availablecash', 'N/A')}"
                )
                log.info(
                    f"     Used Margin    : "
                    f"₹{data.get('utiliseddebits', 'N/A')}"
                )
                log.info(
                    f"     Net Value      : "
                    f"₹{data.get('net', 'N/A')}"
                )
                return data
        except Exception as e:
            log.error(f"  ❌ Funds fetch failed: {e}")
        return None

    # ── Get LTP (Last Traded Price) ───────────────────
    def get_ltp(self, exchange, symbol, token):
        if not self.connected:
            return None
        try:
            data = self.smart_api.ltpData(
                exchange, symbol, token
            )
            if data["status"]:
                ltp = data["data"]["ltp"]
                log.info(f"  {symbol} LTP : ₹{ltp}")
                return ltp
        except Exception as e:
            log.error(f"  ❌ LTP fetch failed: {e}")
        return None

# ── Singleton Connection ──────────────────────────────
_connection = None

def get_connection():
    global _connection
    if _connection is None or not _connection.connected:
        _connection = AngelOneConnection()
        _connection.login()
    return _connection

# ── Main Test ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Trading AI — AngelOne Connection Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Test connection
    conn = AngelOneConnection()

    if conn.login():
        print("\n  Testing profile fetch...")
        conn.get_profile()

        print("\n  Testing funds fetch...")
        conn.get_funds()

        print("\n  Testing LTP fetch (NIFTY)...")
        conn.get_ltp("NSE", "Nifty 50", "99926000")

        print("\n  Testing LTP fetch (BANKNIFTY)...")
        conn.get_ltp("NSE", "Nifty Bank", "99926009")

        conn.logout()

        print("\n" + "=" * 55)
        print("  ✅ AngelOne connection working perfectly!")
        print("  Ready to fetch options + futures data")
        print("=" * 55)
    else:
        print("\n  ❌ Connection failed.")
        print("  Check your .env credentials and try again")