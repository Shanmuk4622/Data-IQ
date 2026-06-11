"""Script to generate sample datasets for DataIQ demo mode."""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
out = Path("assets/sample_datasets")
out.mkdir(parents=True, exist_ok=True)

# ── 1. Titanic-style dataset ────────────────────────────────────────────────
n = 891
titanic = pd.DataFrame({
    "PassengerId": range(1, n+1),
    "Survived": np.random.choice([0, 1], size=n, p=[0.62, 0.38]),
    "Pclass": np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55]),
    "Name": [f"Person_{i}" for i in range(n)],
    "Sex": np.random.choice(["male", "female", "Male", "Female"], size=n, p=[0.45, 0.45, 0.05, 0.05]),
    "Age": np.where(np.random.rand(n) < 0.2, np.nan,
                    np.clip(np.random.normal(29, 14, n), 0.5, 80).round(1)),
    "SibSp": np.random.choice([0,1,2,3,4,5], size=n, p=[0.68,0.23,0.05,0.02,0.01,0.01]),
    "Parch": np.random.choice([0,1,2,3], size=n, p=[0.76,0.13,0.08,0.03]),
    "Ticket": [f"{'A' if np.random.rand()<0.3 else ''}{np.random.randint(10000,999999)}" for _ in range(n)],
    "Fare": np.where(np.random.rand(n) < 0.05, np.nan,
                     np.clip(np.random.exponential(32, n), 0, 512).round(2)),
    "Cabin": pd.array(np.where(np.random.rand(n) < 0.77, None,
                      np.random.choice(["C85","B96","E101","D56","A14","C123"], n)), dtype=object),
    "Embarked": pd.array(np.where(np.random.rand(n) < 0.02, None,
                         np.random.choice(["S","C","Q"], n, p=[0.72,0.19,0.09])), dtype=object),
})
titanic.to_csv(out / "titanic.csv", index=False)
print("✓ titanic.csv")

# ── 2. California Housing-style dataset ─────────────────────────────────────
n = 2000
housing = pd.DataFrame({
    "longitude": np.random.uniform(-124.3, -114.3, n).round(4),
    "latitude":  np.random.uniform(32.5, 42.0, n).round(4),
    "housing_median_age": np.random.choice(range(1, 53), n),
    "total_rooms":   np.clip(np.random.lognormal(7, 0.7, n), 10, 30000).astype(int),
    "total_bedrooms": np.where(np.random.rand(n) < 0.01, np.nan,
                               np.clip(np.random.lognormal(5.5, 0.7, n), 2, 6000)).astype(float),
    "population":  np.clip(np.random.lognormal(6.5, 0.8, n), 5, 30000).astype(int),
    "households":  np.clip(np.random.lognormal(5.5, 0.7, n), 2, 5000).astype(int),
    "median_income": np.clip(np.random.lognormal(1.5, 0.6, n), 0.5, 15).round(4),
    "ocean_proximity": np.random.choice(
        ["<1H OCEAN","INLAND","NEAR OCEAN","NEAR BAY","ISLAND"],
        n, p=[0.44, 0.32, 0.13, 0.10, 0.01]),
    "median_house_value": np.clip(
        np.random.lognormal(12.0, 0.5, n) * np.random.uniform(0.8, 1.2, n),
        15000, 500001).round(-2).astype(int),
})
# introduce some duplicates
dup_idx = np.random.choice(range(len(housing)), 40, replace=False)
housing = pd.concat([housing, housing.iloc[dup_idx]], ignore_index=True)
housing.to_csv(out / "california_housing.csv", index=False)
print("✓ california_housing.csv")

# ── 3. Messy E-commerce dataset ─────────────────────────────────────────────
n = 1500
dates = pd.date_range("2022-01-01", periods=n, freq="6h")
dates_arr = dates.to_numpy().copy()
np.random.shuffle(dates_arr)
dates = pd.DatetimeIndex(dates_arr)
ecommerce = pd.DataFrame({
    "order_id":       range(10001, 10001+n),
    "order_date":     dates.strftime("%Y-%m-%d"),
    "customer_id":    np.random.randint(1, 400, n),
    "customer_name":  np.random.choice(
        ["Alice Johnson","Bob Smith","Carol White","Dave Brown","Eve Davis"], n),
    "product_category": np.random.choice(
        ["Electronics","Clothing","Books","Home & Garden","Sports","Toys","Beauty"],
        n, p=[0.30, 0.22, 0.12, 0.14, 0.10, 0.07, 0.05]),
    "product_name": np.random.choice(
        ["Laptop Pro","Wireless Headset","Running Shoes","Python Book","Coffee Maker",
         "Yoga Mat","Skincare Set","LED Lamp","Bluetooth Speaker","Desk Chair"], n),
    "quantity":    np.random.choice([1,1,1,2,2,3,5,10], n),
    "unit_price":  np.where(np.random.rand(n) < 0.03,
                            [f"${round(p, 2)}" for p in np.random.uniform(5,500,n)],
                            np.random.uniform(5, 500, n).round(2)).tolist(),
    "discount_pct": np.where(np.random.rand(n) < 0.15, np.nan,
                              np.random.choice([0, 5, 10, 15, 20, 25], n)),
    "shipping_country": np.random.choice(
        ["USA","UK","Canada","Germany","France","Australia","India"],
        n, p=[0.45,0.15,0.12,0.10,0.08,0.06,0.04]),
    "payment_method": np.random.choice(
        ["Credit Card","PayPal","Debit Card","Bank Transfer","Crypto"],
        n, p=[0.42, 0.28, 0.18, 0.10, 0.02]),
    "order_status": np.random.choice(
        ["Delivered","Processing","Shipped","Cancelled","Returned"],
        n, p=[0.65, 0.15, 0.10, 0.07, 0.03]),
    "review_score":  np.where(np.random.rand(n) < 0.30, np.nan,
                               np.random.choice([1,2,3,4,5], n, p=[0.04,0.06,0.15,0.35,0.40])),
    "return_flag":   np.random.choice([0, 1], n, p=[0.92, 0.08]),
    "notes":         pd.array(np.where(np.random.rand(n) < 0.7, None,
                               np.random.choice(
                                   ["Great product!","Arrived late","Wrong item sent",
                                    "Excellent quality, will buy again","Packaging was damaged",
                                    "Good value for money"], n)), dtype=object),
})
ecommerce.to_csv(out / "ecommerce_sales.csv", index=False)
print("✓ ecommerce_sales.csv")
print("\nAll sample datasets generated successfully.")
