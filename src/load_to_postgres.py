from sqlalchemy import create_engine
import pandas as pd


# PostgreSQL connection details
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "1234"

# Create SQLAlchemy engine
engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# CSV file paths
path = r"D:\vs_code\GUVI\blinkit_project\data_raw\\"

files = {
    "orders": "Blinkit - blinkit_orders.csv",
    "order_items": "Blinkit - blinkit_order_items.csv",
    "products": "Blinkit - blinkit_products.csv",
    "customers": "Blinkit - blinkit_customers.csv",
    "marketing_performance": "Blinkit - blinkit_marketing_performance.csv",
    "feedback": "Blinkit - blinkit_customer_feedback.csv"
}

for table, file_name in files.items():
    file_path = path + file_name
    print(f"Loading {file_name} into {table} table...")

    df = pd.read_csv(file_path)

    # Push to PostgreSQL
    df.to_sql(table, engine, if_exists="replace", index=False)

    print(f"✔ Successfully loaded {len(df)} rows into '{table}'")

print("🎯 All tables imported successfully!")
