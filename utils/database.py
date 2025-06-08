from pymongo import MongoClient
import pandas as pd

def connect_mongodb():
    """Kết nối đến MongoDB"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        client.admin.command('ping')
        print("MongoDB connection successful!")
        return client["sendo_database"]
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

def get_reviews(db):
    """Lấy dữ liệu đánh giá từ MongoDB"""
    reviews = list(db["reviews"].find({}, {
        "userId": 1,
        "productId": 1,
        "rating": 1,
    }))
    return pd.DataFrame(reviews)

def get_products(db):
    """Lấy dữ liệu sản phẩm từ MongoDB"""
    products = list(db["products"].find({}, {
        "_id": 1,
        "categoryId": 1,
    }))
    return pd.DataFrame(products) 