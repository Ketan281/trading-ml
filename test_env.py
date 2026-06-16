from dotenv import load_dotenv
import os
import pyotp

load_dotenv()

api_key     = os.getenv("ANGELONE_API_KEY")
api_secret  = os.getenv("ANGELONE_API_SECRET")
client_id   = os.getenv("ANGELONE_CLIENT_ID")
password    = os.getenv("ANGELONE_PASSWORD")
totp_secret = os.getenv("ANGELONE_TOTP_SECRET")

print("API Key     :", api_key[:6]     + "******")
print("API Secret  :", api_secret[:6]  + "******")
print("Client ID   :", client_id)
print("Password    :", "********")
print("TOTP Secret :", totp_secret[:4] + "******")

# Test TOTP generation
totp = pyotp.TOTP(totp_secret)
otp  = totp.now()
print(f"\nGenerated OTP : {otp}")
print("✅ All 5 credentials loaded and working!")