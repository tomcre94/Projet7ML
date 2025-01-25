import requests

url = 'https://projet7-deeplearning-e9hafvfabugpe0c3.francecentral-01.azurewebsites.net/predict'
payload = {'text': 'Je suis très content aujourd\'hui!'}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=payload, headers=headers)

print(response.json())