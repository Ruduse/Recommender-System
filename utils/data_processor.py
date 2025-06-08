import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df_reviews, df_products):
    """Làm sạch và chuẩn bị dữ liệu"""
    # Xử lý dữ liệu reviews
    df_reviews['user_id'] = df_reviews['userId'].apply(str)
    df_reviews['product_id'] = df_reviews['productId'].apply(str)
    df_reviews = df_reviews.dropna(subset=['user_id', 'product_id', 'rating'])
    df_reviews = df_reviews[df_reviews['rating'].between(1, 5)]
    df_reviews = df_reviews.sort_values('rating', ascending=False).drop_duplicates(['user_id', 'product_id'])
    
    # Xử lý dữ liệu products
    df_products['product_id'] = df_products['_id'].apply(str)
    df_products['category_id'] = df_products['category'].apply(
        lambda x: x.get('$oid') if isinstance(x, dict) else str(x)
    )
    df_products = df_products.dropna(subset=['product_id', 'category_id'])
    
    # Merge dữ liệu
    df = pd.merge(df_reviews, df_products[['product_id', 'category_id']], 
                 on='product_id', how='left')
    
    return df

def encode_data(df):
    """Encode dữ liệu categorical"""
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    
    df['user_enc'] = user_encoder.fit_transform(df['user_id'])
    df['item_enc'] = item_encoder.fit_transform(df['product_id'])
    df['category_enc'] = category_encoder.fit_transform(
        df['category_id'].fillna('unknown')
    )
    
    return df, user_encoder, item_encoder, category_encoder 