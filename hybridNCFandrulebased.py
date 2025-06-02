# Cài đặt thư viện nếu chưa có
# !pip install pymongo pandas scikit-learn tensorflow

from pymongo import MongoClient
import pandas as pd
import numpy as np
# Kết nối MongoDB
try:
    client = MongoClient('mongodb://localhost:27017/')  # Hoặc MongoDB Atlas connection string
    client.admin.command('ping')
    print("MongoDB connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")

db = client["sendo_database"]

# Trích xuất dữ liệu từ MongoDB
reviews = list(db["reviews"].find({}, {
    "userId": 1,
    "productId": 1,
    "rating": 1,
}))
products = list(db["products"].find({}, {
    "_id": 1,
    "category": 1,
}))

# Chuyển về DataFrame
df_reviews = pd.DataFrame(reviews)
df_products = pd.DataFrame(products)

# Chuyển đổi kiểu dữ liệu
df_reviews['user_id'] = df_reviews['userId'].apply(str)
df_reviews['product_id'] = df_reviews['productId'].apply(str)

df_products['product_id'] = df_products['_id'].apply(str)
df_products['category_id'] = df_products['category'].apply(lambda x: x.get('$oid') if isinstance(x, dict) else str(x))

# Merge reviews với product category
df = pd.merge(df_reviews, df_products[['product_id', 'category_id']], on='product_id', how='left')

# Kiểm tra
print("Shape của DataFrame:", df.shape)
print("\n5 dòng đầu tiên:")
print(df.head())

# Encode ID
from sklearn.preprocessing import LabelEncoder

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
category_encoder = LabelEncoder()

df['user_enc'] = user_encoder.fit_transform(df['user_id'])
df['item_enc'] = item_encoder.fit_transform(df['product_id'])
df['category_enc'] = category_encoder.fit_transform(df['category_id'].fillna('unknown'))

print("\nThông tin encode:")
print(df[['user_id', 'user_enc', 'product_id', 'item_enc', 'category_id', 'category_enc']].head(10))

# Phân tích dữ liệu
print("\nSố lượng người dùng:", df['user_id'].nunique())
print("Số lượng sản phẩm:", df['product_id'].nunique())

# Gợi ý sản phẩm phổ biến (Rule-Based)
popular_products = df.groupby('product_id').size().sort_values(ascending=False).head(10)
print("\nTop sản phẩm phổ biến:\n", popular_products)

# Chuẩn bị dữ liệu cho NCF
from sklearn.model_selection import train_test_split

X = df[['user_enc', 'item_enc', 'category_enc']].values
y = df['rating'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình NCF
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_ncf_model(num_users, num_items, num_categories, embed_size=64):
    user_input = keras.Input(shape=(1,), name='user_input')
    item_input = keras.Input(shape=(1,), name='item_input')
    category_input = keras.Input(shape=(1,), name='category_input')

    # GMF
    user_gmf = layers.Embedding(num_users, embed_size)(user_input)
    item_gmf = layers.Embedding(num_items, embed_size)(item_input)
    gmf = layers.Multiply()([layers.Flatten()(user_gmf), layers.Flatten()(item_gmf)])

    # MLP
    user_mlp = layers.Embedding(num_users, embed_size)(user_input)
    item_mlp = layers.Embedding(num_items, embed_size)(item_input)
    category_mlp = layers.Embedding(num_categories, embed_size)(category_input)
    mlp = layers.Concatenate()([layers.Flatten()(user_mlp), layers.Flatten()(item_mlp), layers.Flatten()(category_mlp)])
    mlp = layers.Dense(128, activation='relu')(mlp)
    mlp = layers.Dropout(0.2)(mlp)
    mlp = layers.Dense(64, activation='relu')(mlp)
    mlp = layers.Dropout(0.2)(mlp)

    # Kết hợp
    combined = layers.Concatenate()([gmf, mlp])
    output = layers.Dense(1, activation='linear')(combined)

    model = keras.Model(inputs=[user_input, item_input, category_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Huấn luyện
num_users = df['user_enc'].nunique()
num_items = df['item_enc'].nunique()
num_categories = df['category_enc'].nunique()

model = build_ncf_model(num_users, num_items, num_categories)
model.fit([X_train[:, 0], X_train[:, 1], X_train[:, 2]], y_train,
          validation_data=([X_test[:, 0], X_test[:, 1], X_test[:, 2]], y_test),
          epochs=10, batch_size=32)

# Đánh giá mô hình
loss, mae = model.evaluate([X_test[:, 0], X_test[:, 1], X_test[:, 2]], y_test)
print(f"\nTest MSE: {loss:.4f}, Test MAE: {mae:.4f}")


# Lưu model
model.save('model.h5')
# Hàm gợi ý

def recommend(user_id_str, top_k=10):
    # Gợi ý phổ biến nếu người dùng mới
    popular_list = df.groupby('product_id').size().sort_values(ascending=False).head(top_k).index.tolist()

    if user_id_str not in df['user_id'].values:
        return popular_list

    uid = user_encoder.transform([user_id_str])[0]
    all_items = df['item_enc'].unique()

    user_ids = np.array([uid] * len(all_items))              # Convert to np.array
    item_ids = np.array(all_items)

    # Lấy category tương ứng từng item
    category_map = df.drop_duplicates('item_enc')[['item_enc', 'category_enc']].set_index('item_enc')
    categories = category_map.loc[all_items, 'category_enc'].values
    categories = np.array(categories)                        # Ensure it's np.array

    # Dự đoán
    preds = model.predict([user_ids, item_ids, categories], verbose=0)
    top_indices = sorted(zip(all_items, preds.flatten()), key=lambda x: x[1], reverse=True)[:top_k]
    product_ids = item_encoder.inverse_transform([x[0] for x in top_indices])

    # Bỏ sản phẩm đã mua
    purchased = df[df['user_id'] == user_id_str]['product_id'].unique()
    recommended = [pid for pid in product_ids if pid not in purchased]

    if len(recommended) < top_k:
        remaining = top_k - len(recommended)
        additional = [pid for pid in popular_list if pid not in recommended][:remaining]
        recommended.extend(additional)

    return recommended

# Gợi ý thử
# test_user = df['user_id'].iloc[0]
test_user = '6835e1bd82ac426ecb896bdf'
print("Đã mua:", df[df['user_id'] == test_user]['product_id'].unique())
print("\nĐược gợi ý:", recommend(test_user))
print("Gợi ý cho người dùng mới:", recommend("new_user_id"))
