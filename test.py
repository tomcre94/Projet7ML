import requests

url = 'http://localhost:5000/predict'
payload = {'text': 'Je suis très content aujourd\'hui!'}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=payload, headers=headers)

print(response.json())
