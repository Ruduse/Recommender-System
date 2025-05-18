import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.optimizers import Adam
import pickle
import os

# === Load & tiền xử lý dữ liệu ===
df = pd.read_csv('data/ncf/review.csv')
df = df.dropna(subset=["user_id", "product_id", "rating"])

# Chuyển đổi ID thành số nguyên (bắt buộc với embedding)
user_ids = df['user_id'].unique().tolist()
product_ids = df['product_id'].unique().tolist()

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
product2product_encoded = {x: i for i, x in enumerate(product_ids)}

user_encoded2user = {i: x for x, i in user2user_encoded.items()}
product_encoded2product = {i: x for x, i in product2product_encoded.items()}

df['user_encoded'] = df['user_id'].map(user2user_encoded)
df['product_encoded'] = df['product_id'].map(product2product_encoded)
df['rating'] = df['rating'].astype(float)

# === Tạo tập train/test ===
X = df[['user_encoded', 'product_encoded']].values
y = df['rating'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Xây dựng mô hình NCF đơn giản ===
num_users = len(user_ids)
num_products = len(product_ids)
embedding_size = 50

user_input = Input(shape=(1,))
user_embedding = Embedding(num_users, embedding_size)(user_input)
user_vec = Flatten()(user_embedding)

product_input = Input(shape=(1,))
product_embedding = Embedding(num_products, embedding_size)(product_input)
product_vec = Flatten()(product_embedding)

concat = Concatenate()([user_vec, product_vec])
dense = Dense(128, activation='relu')(concat)
dense = Dense(64, activation='relu')(dense)
output = Dense(1, activation='linear')(dense)

model = Model([user_input, product_input], output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.summary()

# === Huấn luyện mô hình ===
model.fit([X_train[:, 0], X_train[:, 1]], y_train,
          batch_size=64,
          epochs=10,
          validation_data=([X_test[:, 0], X_test[:, 1]], y_test))

def recommend_products(user_id, top_k=10):
    if user_id not in user2user_encoded:
        print("User not found")
        return []

    user_idx = user2user_encoded[user_id]
    products_bought = df[df['user_id'] == user_id]['product_id'].values
    products_not_bought = [pid for pid in product_ids if pid not in products_bought]

    product_idx = [product2product_encoded[x] for x in products_not_bought]
    user_array = np.full(len(product_idx), user_idx)

    ratings = model.predict([user_array, np.array(product_idx)], verbose=0).flatten()
    top_k_indices = ratings.argsort()[-top_k:][::-1]
    recommended_product_ids = [products_not_bought[i] for i in top_k_indices]

    return pd.DataFrame({
        "product_id": recommended_product_ids,
        "predicted_rating": ratings[top_k_indices]
    })

# Tạo thư mục models nếu chưa tồn tại
os.makedirs('models', exist_ok=True)

# Lưu model
model.save('models/ncf_model.h5')

# Lưu mappings
mappings = {
    'user2user_encoded': user2user_encoded,
    'product2product_encoded': product2product_encoded,
    'user_encoded2user': user_encoded2user,
    'product_encoded2product': product_encoded2product
}

with open('models/mappings.pickle', 'wb') as f:
    pickle.dump(mappings, f)

# Lưu danh sách product_ids
with open('models/product_ids.pickle', 'wb') as f:
    pickle.dump(product_ids, f)

print("Đã lưu model và mappings thành công!")

# Test recommend
recommend_df = recommend_products("U01003", top_k=5)
print("\nRecommendations for user U01003:")
print(recommend_df) 