import requests
import json

# specifying the API endpoint
API_ENDPOINT = ""

# your API key here
API_KEY = ""

# Prepare data
payload = {
    'userconfig': {},
}

# Make POST request
headers = {'x-api-key': API_KEY, 
        'Content-Type': 'application/json'}

response = requests.post(url=API_ENDPOINT, 
                        data=json.dumps(payload),
                        headers=headers)

print("Response JSON: %s" % response.text)


