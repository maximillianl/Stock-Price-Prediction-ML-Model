from test_subjects import *
from data import *
from features import *

import sqlite3

def print_table(db_path: str, table: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table}")
    rows = cur.fetchall()
    for r in rows:
        print(r)
    conn.close()

def search_ticker_in_db(symbol: str):
    symbol = symbol.strip().upper()
    cached = set(s.upper() for s in list_cached_tickers())
    return symbol in cached

def count_rows(db_path, table):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        return cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

def main():
    # # snp500_stocks_api = get_snp500_stocks_api()
    # # print(snp500_stocks)
    snp500_stocks = get_snp500_stocks()
    # # print(snp500_stocks)
    # russell1000_stocks = get_russell1000_stocks()
    # # # print(russell1000_stocks)
    # russell2000_stocks = get_russell2000_stocks()
    # # # print(russell2000_stocks)
    # russell3000_stocks = get_russell3000_stocks()
    # # # print(russell3000_stocks)

    normalized_snp500 = normalize_list(snp500_stocks)


    # print(normalized_snp500)
    # normalized_r1000 = normalize_list(russell1000_stocks)
    # normalized_r2000 = normalize_list(russell2000_stocks)
    # normalized_r3000 = normalize_list(russellu3000_stocks)
    # merged_list = merge_stock_lists(normalized_snp500, normalized_r1000, normalized_r2000, normalized_r3000)
    # list_to_csv(merged_list, "all_stocks.csv")

    # list_to_csv(normalized_r3000, "russell3000_stocks.csv")
    list_to_csv(normalized_snp500, "snp500_stocks.csv")

    #when going through stocks, if ticker not found try adding dash before last letter of ticker symbol

    # remove_ticker_from_db("brkb")
    cache_stock_to_db("snp500_stocks.csv")





    # print_table("stocks_cache.db", "stocks_table")
    # print(search_ticker_in_db("brkb"))
    # print(search_ticker_in_db("BRK-B"))
    # print(list_cached_tickers())
    # print(print_table("stocks_cache.db", "stocks_table"))
    # print(count_rows("stocks_cache.db", "stocks_table"))


    date_range = ("2025-01-01", "2025-12-31")
    # print(get_stock_info("AAPL"))
    # print(get_stock_info("BRK-B", date_range))
    print(get_stock_info("brkb", date_range))


    # graph_stock_info("mgm", date_range, "Volume")




    # with sqlite3.connect("stocks_cache.db") as conn:
    #     cur = conn.cursor()
    #     print("Rows for MGM:", cur.execute(6
    #         "SELECT COUNT(*) FROM stocks_table WHERE ticker_symbol='MGM'"
    #     ).fetchone()[0])

    #     print("First 5 MGM rows:", cur.execute(
    #         "SELECT ticker_symbol, date FROM stocks_table WHERE ticker_symbol='MGM' ORDER BY date LIMIT 5"
    #     ).fetchall())

    print(load_stock_from_db("AAPL"))
    

if __name__ == "__main__":
    main()