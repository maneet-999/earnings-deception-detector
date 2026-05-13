from database import SessionLocal, FeatureVector

FRAUD_TICKERS = ["NUAN", "GE", "WLTW", "IEX", "PRGO"]

db = SessionLocal()

# First reset all labels
for v in db.query(FeatureVector).all():
    v.label = None

# Label clean companies as 0
clean_count = 0
for v in db.query(FeatureVector).all():
    if v.ticker not in FRAUD_TICKERS:
        v.label = 0
        clean_count += 1

# Label fraud companies as 1
fraud_count = 0
for v in db.query(FeatureVector).all():
    if v.ticker in FRAUD_TICKERS:
        v.label = 1
        fraud_count += 1

db.commit()
db.close()
print(f"Clean (0): {clean_count}")
print(f"Fraud (1): {fraud_count}")