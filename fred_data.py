import requests
import csv

url = 'https://api.stlouisfed.org/fred/series/observations'
api_key = '36d75930c2466915834a50cacb962b30'
series_id = 'UNRATE'

params = {
    'series_id': series_id,
    'api_key': api_key,
    'file_type': 'json',
    'observation_start': '2020-01-01',
    'observation_end': '2024-12-31'
}

response = requests.get(url, params=params)
json_data = response.json()
observations = json_data['observations']
csv_file_path = 'fred_unemployment_data.csv'

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Date', 'Unemployment Rate'])

    for observation in observations:
        writer.writerow([observation['date'], observation['value']])

print(f"Data downloaded and saved to {csv_file_path}")
