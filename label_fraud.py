from database import SessionLocal, FeatureVector

FRAUD_TICKERS = ["NUAN", "GE", "WLTW", "IEX", "PRGO"]

db = SessionLocal()
count = 0
for v in db.query(FeatureVector).all():
    if v.ticker in FRAUD_TICKERS and v.label is None:
        v.label = 1
        count += 1
db.commit()
db.close()
print(f"Labelled {count} fraud vectors as high risk (1)")