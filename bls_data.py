import requests
import csv

api_key = '166fbd6bbd2d4951af0431d257b62d42'
url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

headers = {'Content-type': 'application/json'}
data = {
    "seriesid": ["LNS14027662"],
    "startyear": "2010",
    "endyear": "2024",
    "registrationkey": api_key
}

response = requests.post(url, json=data, headers=headers)
json_data = response.json()
series_data = json_data['Results']['series'][0]['data']
csv_file_path = 'bls_unemployment_data.csv'

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Year', 'Period', 'Value', 'Footnote'])

    for item in series_data:
        writer.writerow([item['year'], item['period'], item['value'], item.get('footnotes', '')])

print(f"Data downloaded and saved to {csv_file_path}")
