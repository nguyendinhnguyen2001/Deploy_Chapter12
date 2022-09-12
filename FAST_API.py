from fastapi import FastAPI
import pickle
app=FastAPI()
filename='diabetes.sav'
model = pickle.load(open(filename, 'rb'))

@app.post("/diabetes/v1/predict")
def predict(BMI:int,Age:int,Glucose:int):
    features_list = [BMI,Age,Glucose]
    prediction = model.predict([features_list])
    confidence = model.predict_proba([features_list])
    response = {}
    response['prediction'] = int(prediction[0])
    response['confidence'] = str(round(np.amax(confidence[0]) * 100 ,2))
    return response