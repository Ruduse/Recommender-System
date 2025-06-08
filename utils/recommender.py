import numpy as np

def get_popular_products(df, top_k=10):
    """Lấy danh sách sản phẩm phổ biến"""
    return df.groupby('product_id').size().sort_values(
        ascending=False
    ).head(top_k).index.tolist()

def recommend_products(model, df, user_id_str, user_encoder, item_encoder, top_k=10):
    """Gợi ý sản phẩm cho người dùng"""
    # Gợi ý phổ biến nếu người dùng mới
    popular_list = get_popular_products(df, top_k)
    
    if user_id_str not in df['user_id'].values:
        return popular_list

    uid = user_encoder.transform([user_id_str])[0]
    all_items = df['item_enc'].unique()

    user_ids = np.array([uid] * len(all_items))
    item_ids = np.array(all_items)

    # Lấy category tương ứng từng item
    category_map = df.drop_duplicates('item_enc')[
        ['item_enc', 'category_enc']
    ].set_index('item_enc')
    categories = category_map.loc[all_items, 'category_enc'].values

    # Dự đoán
    preds = model.predict(
        [user_ids, item_ids, categories], 
        verbose=0
    ) * 5
    
    top_indices = sorted(
        zip(all_items, preds.flatten()), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]
    
    product_ids = item_encoder.inverse_transform([x[0] for x in top_indices])

    # Bỏ sản phẩm đã mua
    purchased = df[df['user_id'] == user_id_str]['product_id'].unique()
    recommended = [pid for pid in product_ids if pid not in purchased]

    if len(recommended) < top_k:
        remaining = top_k - len(recommended)
        additional = [pid for pid in popular_list if pid not in recommended][:remaining]
        recommended.extend(additional)

    return recommended 