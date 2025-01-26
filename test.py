import requests

url = 'https://projet7ml-c5h3ase6h6ayamhv.francecentral-01.azurewebsites.net/predict'
payload = {'text': 'Je suis très content aujourd\'hui!'}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=payload, headers=headers)

# Imprimer le statut de la réponse et le contenu brut
print(f'Status Code: {response.status_code}')
print(f'Response Content: {response.text}')

# Essayer de décoder la réponse en JSON
try:
    print(response.json())
except requests.exceptions.JSONDecodeError:
    print("La réponse n'est pas au format JSON.")