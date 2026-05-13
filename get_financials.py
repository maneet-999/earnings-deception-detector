from financial_fetcher import FinancialFetcher

fetcher = FinancialFetcher()
tickers = ["MSFT", "GOOGL", "TSLA", "AMZN"]

for ticker in tickers:
    print(f"\nFetching financials for {ticker}...")
    fin_df = fetcher.get_quarterly_financials(ticker, start_year=2023)
    restatements = fetcher.get_restatement_flags(ticker)
    fetcher.save_financials(ticker, fin_df, restatements)
    print(f"Done {ticker}")

print("\nAll done!")