from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Set
import numpy as np
import pandas as pd
from pymongo import MongoClient
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import logging
from functools import lru_cache
import time
from collections import Counter
from datetime import datetime, timedelta

# ────────────────────────────────────────────────────────────────────────────────
# 1) Setup logging
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# 2) Khởi tạo FastAPI + CORS
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4000"],   # domain React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────────
# 3) Kết nối MongoDB & load dữ liệu
# ────────────────────────────────────────────────────────────────────────────────

# Kết nối MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["sendo_database"]

print("🔄 Đang tải dữ liệu MongoDB ...")

# Lấy dữ liệu từ MongoDB
reviews = list(db["reviews"].find({}, {
    "userId": 1,
    "productId": 1,
    "rating": 1,
    "createdAt": 1
}))

products = list(db["products"].find({}, {
    "_id": 1,
    "category": 1,
    "originalPrice": 1,
    "newPrice": 1
}))

# Chuyển đổi sang DataFrame
df_reviews = pd.DataFrame(reviews)
df_products = pd.DataFrame(products)

# Chuẩn hóa dữ liệu reviews
df_reviews["user_id"] = df_reviews["userId"].astype(str)
df_reviews["product_id"] = df_reviews["productId"].astype(str)
df_reviews["timestamp"] = pd.to_datetime(df_reviews["createdAt"])

# Chuẩn hóa dữ liệu products
df_products["product_id"] = df_products["_id"].astype(str)

# Xử lý category: có thể là ObjectId hoặc dict {"$oid": ...}
df_products["category_id"] = df_products["category"].apply(
    lambda x: str(x.get("$oid")) if isinstance(x, dict) else str(x)
)

# Ưu tiên lấy newPrice nếu có, fallback về originalPrice
df_products["price"] = df_products["newPrice"].fillna(df_products["originalPrice"])

# Gộp dữ liệu reviews với thông tin sản phẩm
df = pd.merge(
    df_reviews,
    df_products[["product_id", "category_id", "price"]],
    on="product_id",
    how="left"
)

print("✅ Dữ liệu đã được xử lý và gộp thành công!")


# ────────────────────────────────────────────────────────────────────────────────
# 4) LabelEncoder & Caching
# ────────────────────────────────────────────────────────────────────────────────
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
category_encoder = LabelEncoder()

df["user_enc"] = user_encoder.fit_transform(df["user_id"])
df["item_enc"] = item_encoder.fit_transform(df["product_id"])
df["category_enc"] = category_encoder.fit_transform(df["category_id"].fillna("unknown"))

@lru_cache(maxsize=1)
def get_popular_list() -> List[str]:
    """Lấy danh sách sản phẩm phổ biến (top 10)"""
    return df["product_id"].value_counts().head(10).index.tolist()

@lru_cache(maxsize=1)
def get_category_map() -> pd.Series:
    """Map giữa item_enc và category_enc"""
    return (
        df.drop_duplicates("item_enc")[["item_enc", "category_enc"]]
          .set_index("item_enc")["category_enc"]
    )

# ────────────────────────────────────────────────────────────────────────────────
# 5) Load mô hình
# ────────────────────────────────────────────────────────────────────────────────
print("🔄 Đang load mô hình ...")
model = load_model("model.h5")
print("✅  Đã load xong mô hình")

# ────────────────────────────────────────────────────────────────────────────────
# 6) Pydantic schema
# ────────────────────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = 10

class RecommendationItem(BaseModel):
    product_id: str
    predicted_rating: float
    category_id: str
    price: float

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]

# ────────────────────────────────────────────────────────────────────────────────
# 7) Rule-based filters
# ────────────────────────────────────────────────────────────────────────────────
def get_purchased_items(user_id: str) -> Set[str]:
    """Lấy danh sách sản phẩm đã mua của user"""
    return set(df[df["user_id"] == user_id]["product_id"].unique())

def get_recent_products(days: int = 30) -> Set[str]:
    """Lấy danh sách sản phẩm mới trong N ngày gần đây"""
    recent_date = datetime.now() - timedelta(days=days)
    return set(
        df[df["timestamp"] >= recent_date]["product_id"].unique()
    )

def apply_category_diversity(recs: List[Dict], max_same_category: int = 3) -> List[Dict]:
    """Giới hạn số lượng sản phẩm cùng category trong danh sách khuyến nghị"""
    category_counts = Counter()
    filtered_recs = []
    
    for rec in recs:
        cat = rec["category_id"]
        if category_counts[cat] < max_same_category:
            filtered_recs.append(rec)
            category_counts[cat] += 1
            
    return filtered_recs

def apply_price_diversity(recs: List[Dict], price_ranges: List[tuple] = [(0, 100000), (100000, 500000), (500000, float('inf'))]) -> List[Dict]:
    """Đảm bảo đa dạng về giá trong danh sách khuyến nghị"""
    price_range_counts = Counter()
    filtered_recs = []
    
    for rec in recs:
        price = rec["price"]
        for i, (min_price, max_price) in enumerate(price_ranges):
            if min_price <= price < max_price:
                if price_range_counts[i] < 4:  # Max 4 sản phẩm mỗi khoảng giá
                    filtered_recs.append(rec)
                    price_range_counts[i] += 1
                break
                
    return filtered_recs

def boost_recent_products(recs: List[Dict], boost_factor: float = 1.2) -> List[Dict]:
    """Tăng điểm cho sản phẩm mới"""
    recent_products = get_recent_products()
    
    for rec in recs:
        if rec["product_id"] in recent_products:
            rec["predicted_rating"] = min(5.0, rec["predicted_rating"] * boost_factor)
            
    return sorted(recs, key=lambda x: x["predicted_rating"], reverse=True)

# ────────────────────────────────────────────────────────────────────────────────
# 8) Core recommend() – xử lý an toàn
# ────────────────────────────────────────────────────────────────────────────────
def recommend(user_id: str, top_k: int = 10) -> List[Dict]:
    start_time = time.time()
    logger.info(f"Starting recommendation for user {user_id}")
    
    # 1. Xử lý user mới
    if user_id not in df["user_id"].values:
        logger.info(f"User {user_id} not found, returning popular items")
        popular_items = get_popular_list()
        recs = []
        for pid in popular_items:
            product_info = df_products[df_products["product_id"] == pid].iloc[0]
            recs.append({
                "product_id": pid,
                "predicted_rating": 4.5,
                "category_id": product_info["category_id"],
                "price": product_info["price"]
            })
        return apply_category_diversity(recs)[:top_k]

    # 2. Lấy user encoding
    try:
        uid = user_encoder.transform([user_id])[0]
    except ValueError:
        logger.warning(f"User {user_id} not in encoder, falling back to popular items")
        return recommend(user_id, top_k)  # Recursive call với user mới

    # 3. Dự đoán rating cho tất cả sản phẩm
    all_items = df["item_enc"].unique()
    user_ids = np.full(len(all_items), uid)
    item_ids = all_items
    categories = get_category_map().loc[all_items].values

    preds = model.predict([user_ids, item_ids, categories], verbose=0).flatten()

    # 4. Scale predictions về [1, 5]
    if preds.max() > preds.min():
        scaled_preds = 1 + (preds - preds.min()) * 4 / (preds.max() - preds.min())
    else:
        scaled_preds = np.full_like(preds, 4.0)

    # 5. Tạo danh sách khuyến nghị ban đầu
    purchased = get_purchased_items(user_id)
    recs = []
    
    for enc_id, score in sorted(zip(item_ids, scaled_preds), key=lambda x: x[1], reverse=True):
        pid = item_encoder.inverse_transform([enc_id])[0]
        if pid not in purchased:
            product_info = df_products[df_products["product_id"] == pid].iloc[0]
            recs.append({
                "product_id": pid,
                "predicted_rating": float(round(score, 2)),
                "category_id": product_info["category_id"],
                "price": product_info["price"]
            })
        if len(recs) >= top_k * 2:  # Lấy nhiều hơn để có thể lọc
            break

    # 6. Áp dụng các rules
    recs = apply_category_diversity(recs)
    recs = apply_price_diversity(recs)
    recs = boost_recent_products(recs)
    
    # 7. Bổ sung popular items nếu thiếu
    if len(recs) < top_k:
        logger.info(f"Adding popular items to reach {top_k} recommendations")
        for pid in get_popular_list():
            if pid not in purchased and pid not in [x["product_id"] for x in recs]:
                product_info = df_products[df_products["product_id"] == pid].iloc[0]
                recs.append({
                    "product_id": pid,
                    "predicted_rating": 4.5,
                    "category_id": product_info["category_id"],
                    "price": product_info["price"]
                })
            if len(recs) >= top_k:
                break

    execution_time = time.time() - start_time
    logger.info(f"Recommendation completed in {execution_time:.2f}s for user {user_id}")
    return recs[:top_k]

# ────────────────────────────────────────────────────────────────────────────────
# 9) API endpoint
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendation(req: RecommendRequest):
    try:
        recs = recommend(req.user_id, req.top_k)
        return {"recommendations": recs}
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ────────────────────────────────────────────────────────────────────────────────
# 10) Run dev server
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
