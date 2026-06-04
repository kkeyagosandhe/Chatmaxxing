import pandas as pd

df = pd.read_csv("data/customer_support_tickets.csv")

# Keep only resolved tickets with satisfaction ratings
df_clean = df.dropna(subset=["Resolution", "Customer Satisfaction Rating"]).copy()

# Fix template placeholders row by row
df_clean["Ticket Description"] = df_clean.apply(
    lambda row: row["Ticket Description"].replace("{product_purchased}", row["Product Purchased"]),
    axis=1
)

# Save cleaned version
df_clean.to_csv("data/tickets_clean.csv", index=False)

print(f"Clean rows: {len(df_clean)}")
print(f"Sample resolution:\n{df_clean['Resolution'].iloc[0]}")