import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import re
from datetime import datetime

# Hàm xử lý từ khóa tìm kiếm
def extract_keywords(search_query):
    # Loại bỏ ký tự đặc biệt và tách từ khóa
    query = re.sub(r'[^\w\s]', '', search_query.lower())
    return set(query.split())

# Rule-based recommendation dựa trên từ khóa
def rule_based_recommendation(search_history, product_data, top_n=5):
    if not search_history:
        return []
    latest_query = search_history[-1]
    keywords = extract_keywords(latest_query)
    
    # Tìm sản phẩm có tên hoặc mô tả chứa từ khóa
    matched_products = []
    for idx, product in product_data.iterrows():
        product_text = f"{product['name']} {product['description']}".lower()
        if any(keyword in product_text for keyword in keywords):
            matched_products.append((idx, product['name']))
    
    return matched_products[:top_n]

# Popularity-based recommendation cho người dùng mới
def popularity_based_recommendation(product_data, top_n=5):
    # Sắp xếp sản phẩm theo số lượt mua hoặc lượt xem (giả định có cột 'popularity_score')
    return product_data.sort_values(by='popularity_score', ascending=False)[['id', 'name']].head(top_n).values.tolist()

# Neural Collaborative Filtering Model
class NCFModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size=50):
        super(NCFModel, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_ids, item_ids = inputs
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        concat = tf.keras.layers.Concatenate()([user_vec, item_vec])
        x = self.dense1(concat)
        x = self.dense2(x)
        return self.output_layer(x)

# Hàm xây dựng hệ thống gợi ý
class RecommendationSystem:
    def __init__(self, product_data, user_interactions, embedding_size=50):
        self.product_data = product_data
        self.user_interactions = user_interactions
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Mã hóa user_id và item_id
        self.user_interactions['user_id'] = self.user_encoder.fit_transform(self.user_interactions['user_id'])
        self.user_interactions['item_id'] = self.item_encoder.fit_transform(self.user_interactions['item_id'])
        
        # Khởi tạo mô hình NCF
        num_users = len(self.user_encoder.classes_)
        num_items = len(self.item_encoder.classes_)
        self.model = NCFModel(num_users, num_items, embedding_size)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Huấn luyện mô hình
        self.train_model()

    def train_model(self, epochs=10, batch_size=32):
        # Chuẩn bị dữ liệu huấn luyện
        X = [self.user_interactions['user_id'].values, self.user_interactions['item_id'].values]
        y = self.user_interactions['interaction'].values  # 1: tương tác tích cực, 0: không
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def recommend(self, user_id, search_history, top_n=5):
        # Kiểm tra xem user_id có trong hệ thống không
        if user_id not in self.user_encoder.classes_:
            # Người dùng mới: Sử dụng popularity-based
            return popularity_based_recommendation(self.product_data, top_n)
        
        # Rule-based: Kiểm tra lịch sử tìm kiếm
        rule_based_results = rule_based_recommendation(search_history, self.product_data, top_n)
        if rule_based_results:
            return rule_based_results
        
        # NCF-based: Sử dụng mô hình NCF
        encoded_user_id = self.user_encoder.transform([user_id])[0]
        item_ids = np.arange(len(self.item_encoder.classes_))
        user_ids = np.array([encoded_user_id] * len(item_ids))
        
        # Dự đoán điểm số
        predictions = self.model.predict([user_ids, item_ids], verbose=0)
        top_indices = np.argsort(predictions[:, 0])[::-1][:top_n]
        top_item_ids = self.item_encoder.inverse_transform(top_indices)
        
        # Lấy thông tin sản phẩm
        recommended_products = self.product_data[self.product_data['id'].isin(top_item_ids)][['id', 'name']].values.tolist()
        return recommended_products

# Ví dụ sử dụng
if __name__ == "__main__":
    # Dữ liệu giả lập
    product_data = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['Laptop Dell', 'Smartphone Samsung', 'Tai nghe Sony', 'Máy tính bảng Apple'],
        'description': ['Laptop hiệu năng cao', 'Điện thoại thông minh', 'Tai nghe không dây', 'Tablet mỏng nhẹ'],
        'popularity_score': [100, 200, 150, 120]
    })
    
    user_interactions = pd.DataFrame({
        'user_id': ['user1', 'user1', 'user2', 'user2'],
        'item_id': [1, 2, 2, 3],
        'interaction': [1, 1, 1, 0]
    })
    
    search_history = ['laptop cao cấp', 'smartphone']
    
    # Khởi tạo hệ thống
    recommender = RecommendationSystem(product_data, user_interactions)
    
    # Gợi ý cho người dùng
    recommendations = recommender.recommend('user1', search_history)
    print("Gợi ý cho user1:", recommendations)
    
    # Gợi ý cho người dùng mới
    recommendations_new_user = recommender.recommend('user3', [])
    print("Gợi ý cho user3 (mới):", recommendations_new_user)