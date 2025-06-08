import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np

def build_ncf_model(num_users, num_items, num_categories, embed_size=64):
    """Xây dựng mô hình NCF"""
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
    mlp = layers.Concatenate()([
        layers.Flatten()(user_mlp), 
        layers.Flatten()(item_mlp), 
        layers.Flatten()(category_mlp)
    ])
    mlp = layers.Dense(128, activation='relu')(mlp)
    mlp = layers.Dropout(0.2)(mlp)
    mlp = layers.Dense(64, activation='relu')(mlp)
    mlp = layers.Dropout(0.2)(mlp)

    # Kết hợp
    combined = layers.Concatenate()([gmf, mlp])
    output = layers.Dense(1, activation='linear')(combined)

    model = keras.Model(
        inputs=[user_input, item_input, category_input], 
        outputs=output
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model

def prepare_data(df):
    """Chuẩn bị dữ liệu cho training"""
    X = df[['user_enc', 'item_enc', 'category_enc']].values
    y = df['rating'].values / 5.0
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
    """Huấn luyện mô hình"""
    return model.fit(
        [X_train[:, 0], X_train[:, 1], X_train[:, 2]], y_train,
        validation_data=([X_test[:, 0], X_test[:, 1], X_test[:, 2]], y_test),
        epochs=epochs,
        batch_size=batch_size
    ) 