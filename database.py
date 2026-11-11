import sqlite3

DB_NAME = "econocausal.db"

def init_db():
    """Database aur Table setup karta hai."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dynamic_pricing_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT NOT NULL,
            current_price REAL NOT NULL,
            proposed_price REAL NOT NULL,
            estimated_effect REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_log(product_id: str, current_price: float, proposed_price: float, effect: float):
    """Prediction outputs ko database me save karta hai."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO dynamic_pricing_logs (product_id, current_price, proposed_price, estimated_effect)
        VALUES (?, ?, ?, ?)
    ''', (product_id, current_price, proposed_price, effect))
    conn.commit()
    conn.close()
