# Cài đặt thư viện
# !pip install pymongo pandas scikit-learn tensorflow matplotlib

import os
import pickle
import schedule
import time
import pandas as pd
import numpy as np
from datetime import datetime

from utils.database import connect_mongodb, get_reviews, get_products
from utils.data_processor import clean_data, encode_data
from utils.model import build_ncf_model, prepare_data, train_model
from utils.recommender import recommend_products
from utils.visualizer import plot_training_history, plot_model_comparison



# Kết nối MongoDB
db = connect_mongodb()
if db is None:
    exit()

# Trích xuất dữ liệu từ MongoDB
df_reviews = get_reviews(db)
df_products = get_products(db)

# Làm sạch và chuẩn bị dữ liệu
df = clean_data(df_reviews, df_products)
df, user_encoder, item_encoder, category_encoder = encode_data(df)

# Lưu encoders
os.makedirs('models', exist_ok=True)
mappings = {
    'user_encoder': user_encoder,
    'item_encoder': item_encoder,
    'category_encoder': category_encoder
}
with open('models/mappings.pickle', 'wb') as f:
    pickle.dump(mappings, f)

# Phân tích dữ liệu
print("\nSố lượng người dùng:", df['user_id'].nunique())
print("Số lượng sản phẩm:", df['product_id'].nunique())

# Chuẩn bị dữ liệu cho NCF
X_train, X_test, y_train, y_test = prepare_data(df)

# Tạo và huấn luyện mô hình
model = build_ncf_model(
    num_users=df['user_enc'].nunique(),
    num_items=df['item_enc'].nunique(),
    num_categories=df['category_enc'].nunique()
)

history = train_model(model, X_train, y_train, X_test, y_test)

# Đánh giá mô hình
loss, mae = model.evaluate([X_test[:, 0], X_test[:, 1], X_test[:, 2]], y_test)
print(f"\nTest MSE: {loss:.4f}, Test MAE: {mae:.4f}")

# Lưu model
model.save('models/reviewModel.keras')

# Vẽ biểu đồ
plot_training_history(history)
plot_model_comparison()

# Hàm gợi ý
def recommend(user_id_str, top_k=30):
    return recommend_products(model, df, user_id_str, user_encoder, item_encoder, top_k)

# Gợi ý thử
test_user = '6835e1bd82ac426ecb896bdf'  # Thay bằng user_id có trong df
print("Đã mua:", df[df['user_id'] == test_user]['product_id'].unique())
print("\nĐược gợi ý:", recommend(test_user))
print("Gợi ý cho người dùng mới:", recommend("new_user_id"))

def evaluate_precision_recall_at_k(k=5):
    # Tạo DataFrame từ X_test
    test_df = pd.DataFrame({
        'user_enc': X_test[:, 0],
        'item_enc': X_test[:, 1]
    })

    # Chuyển đổi lại về ID gốc
    test_df['user_id'] = user_encoder.inverse_transform(test_df['user_enc'])
    test_df['product_id'] = item_encoder.inverse_transform(test_df['item_enc'])

    # Danh sách người dùng trong test
    user_ids_test = test_df['user_id'].unique()

    precisions = []
    recalls = []

    for uid in user_ids_test:
        # Sản phẩm thực tế user đã tương tác trong tập test
        actual_items = test_df[test_df['user_id'] == uid]['product_id'].values
        if len(actual_items) == 0:
            continue

        # Sản phẩm được gợi ý
        recommended_items = recommend(uid, top_k=k)

        # Giao giữa 2 tập
        hit_items = set(recommended_items) & set(actual_items)

        # Precision và Recall
        precision = len(hit_items) / k
        recall = len(hit_items) / len(actual_items)

        precisions.append(precision)
        recalls.append(recall)

    # Trả về trung bình
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    return avg_precision, avg_recall

precision_at_5, recall_at_5 = evaluate_precision_recall_at_k(k=5)
print(f"\n🔍 Precision@5: {precision_at_5:.4f}")
print(f"🔁 Recall@5: {recall_at_5:.4f}")

def train_model_scheduled():
    print(f"\n=== Bắt đầu tái huấn luyện mô hình lúc {datetime.now()} ===")
    
    # Lấy dữ liệu mới từ MongoDB
    df_reviews = get_reviews(db)
    df_products = get_products(db)
    
    # Xử lý dữ liệu
    df = clean_data(df_reviews, df_products)
    df, user_encoder, item_encoder, category_encoder = encode_data(df)
    
    # Chuẩn bị dữ liệu cho NCF
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Huấn luyện mô hình mới
    global model
    model = build_ncf_model(
        num_users=df['user_enc'].nunique(),
        num_items=df['item_enc'].nunique(),
        num_categories=df['category_enc'].nunique()
    )
    
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Lưu model và mappings mới
    model.save('models/reviewModel.keras')
    mappings = {
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'category_encoder': category_encoder
    }
    with open('models/mappings.pickle', 'wb') as f:
        pickle.dump(mappings, f)
    
    print(f"=== Hoàn thành tái huấn luyện mô hình lúc {datetime.now()} ===")

# Lên lịch chạy mỗi tuần
schedule.every().sunday.at("18:55").do(train_model_scheduled)

# Vòng lặp chạy schedule
print("Đã lên lịch tái huấn luyện mô hình mỗi thứ 2 lúc 00:00")
while True:
    schedule.run_pending()
    time.sleep(60)  # Kiểm tra mỗi phút