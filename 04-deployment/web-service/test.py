import requests
import predict


ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

features = predict.prepare_features(ride)
pred = predict.predict(features)
print(pred)

url = 'http://localhost:9696/predict'
try:
    response = requests.post(url, json=ride)
    print(response.json())
except requests.exceptions.ConnectionError as e:
    print("Connection refused, make sure the app is online")
    # python predict.py to init the app with flask server 
    # gunicorn --bind=0.0.0.0:9696 predict:app for gunicorn server
