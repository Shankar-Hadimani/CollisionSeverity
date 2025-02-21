import requests
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_prediction():
    url = 'http://127.0.0.1:5000/predict'  # Adjust the URL based on your Flask app's configuration
    files = {'file': open('data/accident_vehicle_predict.csv', 'rb')}
    
    logging.info("Starting prediction request...")
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        logging.info("Prediction successful.")
        print("Prediction Results:")
        print(json.dumps(response.json(), indent=4))  # Prettify JSON output
    else:
        logging.error(f"Prediction failed with status code {response.status_code}")
        print("Error:", response.text)

if __name__ == '__main__':
    test_prediction()