"""
الخطوة ٣: بناء وتدريب نموذج Transformer
للتعرف على لغة الإشارة العربية
"""
import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===== الإعدادات =====
PROCESSED_PATH = os.path.join('..', 'data', 'processed')
MODEL_SAVE_PATH = os.path.join('..', 'data', 'model')
SEQUENCE_LENGTH = 30
FEATURES = 258
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4


# ===== Positional Encoding =====
class PositionalEncoding(layers.Layer):
    """
    يضيف معلومات الموقع الزمني لكل فريم
    عشان الـ Transformer يعرف ترتيب الحركات
    """
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        
        # حساب الـ encoding مسبقاً
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]
    
    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return config


# ===== Transformer Encoder Block =====
class TransformerBlock(layers.Layer):
    """
    وحدة Transformer واحدة:
    Multi-Head Attention → Add & Norm → FFN → Add & Norm
    """
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config


# ===== بناء النموذج الكامل =====
def build_model(num_classes, seq_len=SEQUENCE_LENGTH, features=FEATURES,
                d_model=256, num_heads=8, ff_dim=512, num_layers=4, dropout=0.3):
    """
    بناء نموذج Transformer للتعرف على الإشارات
    
    المدخل: (batch, 30 فريم, 258 feature)
    المخرج: (batch, عدد_الإشارات)
    """
    inputs = layers.Input(shape=(seq_len, features))
    
    # ١. تقليص الأبعاد: 258 → 256
    x = layers.Dense(d_model)(inputs)
    
    # ٢. Positional Encoding
    x = PositionalEncoding(seq_len, d_model)(x)
    x = layers.Dropout(dropout)(x)
    
    # ٣. Transformer Encoder × 4
    for i in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim, dropout, name=f"transformer_{i}")(x)
    
    # ٤. Global Average Pooling — نجمع كل الفريمات في vector واحد
    x = layers.GlobalAveragePooling1D()(x)
    
    # ٥. Classification Head
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    return model


def main():
    # ===== تحميل البيانات =====
    print("📂 جاري تحميل البيانات...")
    
    X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train.npy'))
    y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
    X_val = np.load(os.path.join(PROCESSED_PATH, 'X_val.npy'))
    y_val = np.load(os.path.join(PROCESSED_PATH, 'y_val.npy'))
    X_test = np.load(os.path.join(PROCESSED_PATH, 'X_test.npy'))
    y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
    
    with open(os.path.join(PROCESSED_PATH, 'class_names.json'), 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    
    num_classes = len(class_names)
    
    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    print(f"   عدد الإشارات: {num_classes}")
    
    # ===== بناء النموذج =====
    print("\n🏗️ جاري بناء النموذج...")
    model = build_model(num_classes)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # ===== Callbacks =====
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    callbacks = [
        # حفظ أفضل نسخة من النموذج
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_SAVE_PATH, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # إيقاف مبكر لو ما فيه تحسن
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # تقليل سرعة التعلم تلقائياً
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]
    
    # ===== التدريب =====
    print("\n🚀 بدء التدريب...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # ===== التقييم النهائي =====
    print("\n📊 التقييم على بيانات الاختبار:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   الدقة: {test_acc * 100:.2f}%")
    print(f"   الخسارة: {test_loss:.4f}")
    
    # ===== حفظ النموذج النهائي =====
    model.save(os.path.join(MODEL_SAVE_PATH, 'final_model.keras'))
    
    # حفظ التاريخ
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(MODEL_SAVE_PATH, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f)
    
    print(f"\n✅ تم حفظ النموذج في: {MODEL_SAVE_PATH}")
    print(f"   best_model.keras — أفضل نسخة")
    print(f"   final_model.keras — النسخة النهائية")
    print(f"   training_history.json — تاريخ التدريب")


if __name__ == '__main__':
    main()
