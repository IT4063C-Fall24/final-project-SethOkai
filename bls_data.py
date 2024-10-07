import requests

api_key = '166fbd6bbd2d4951af0431d257b62d42'

url = f'https://api.bls.gov/publicAPI/v2/timeseries/data/'

headers = {'Content-type': 'application/json'}
data = {
    "seriesid": ["LNS14027662"],
    "startyear": "2010",
    "endyear": "2024",
    "registrationkey": api_key
}

response = requests.post(url, json=data, headers=headers)
json_data = response.json()
print(json_data)
