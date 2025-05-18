# recommendation_service/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model và data
print("Đang load model...")
model = tf.keras.models.load_model('models/ncf_model.h5')
print("Đã load model xong")

print("Đang load data...")
df = pd.read_csv('data/ncf/review.csv')
print("Đã load data xong")

# Load mappings và product_ids
print("Đang load mappings...")
with open('models/mappings.pickle', 'rb') as f:
    mappings = pickle.load(f)
    user2user_encoded = mappings['user2user_encoded']
    product2product_encoded = mappings['product2product_encoded']
    user_encoded2user = mappings['user_encoded2user']
    product_encoded2product = mappings['product_encoded2product']

with open('models/product_ids.pickle', 'rb') as f:
    product_ids = pickle.load(f)
print("Đã load mappings xong")

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: str, top_k: int = 10):
    if user_id not in user2user_encoded:
        return {"error": "User not found"}

    user_idx = user2user_encoded[user_id]
    products_bought = df[df['user_id'] == user_id]['product_id'].values
    products_not_bought = [pid for pid in product_ids if pid not in products_bought]

    product_idx = [product2product_encoded[x] for x in products_not_bought]
    user_array = np.full(len(product_idx), user_idx)

    ratings = model.predict([user_array, np.array(product_idx)], verbose=0).flatten()
    top_k_indices = ratings.argsort()[-top_k:][::-1]
    recommended_product_ids = [products_not_bought[i] for i in top_k_indices]

    return {
        "recommendations": [
            {"product_id": pid, "predicted_rating": float(ratings[i])}
            for i, pid in enumerate(recommended_product_ids)
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)