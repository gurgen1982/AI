import http.client
import json

conn = http.client.HTTPSConnection("api.coinbase.com")
payload = ''
headers = {
  'Content-Type': 'application/json'
}
conn.request("GET", "/api/v3/brokerage/products/BTC/candles?start=2015-01-01&end=2023-09-01&granularity=ONE_MINUTE", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))