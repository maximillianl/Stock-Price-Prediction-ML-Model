from test_subjects import *

def main():
    # # snp500_stocks_api = get_snp500_stocks_api()
    # # print(snp500_stocks)
    snp500_stocks = get_snp500_stocks()
    # print(snp500_stocks)
    russell1000_stocks = get_russell1000_stocks()
    # # print(russell1000_stocks)
    russell2000_stocks = get_russell2000_stocks()
    # # print(russell2000_stocks)
    russell3000_stocks = get_russell3000_stocks()
    # # print(russell3000_stocks)

    normalized_snp500 = normalize_list(snp500_stocks)
    print(normalized_snp500)
    normalized_r1000 = normalize_list(russell1000_stocks)
    normalized_r2000 = normalize_list(russell2000_stocks)
    normalized_r3000 = normalize_list(russell3000_stocks)
    merged_list = merge_stock_lists(normalized_snp500, normalized_r1000, normalized_r2000, normalized_r3000)
    list_to_csv(merged_list, "all_stocks.csv")

    list_to_csv(normalized_r3000, "russell3000_stocks.csv")
    list_to_csv(normalized_snp500, "snp500_stocks.csv")





if __name__ == "__main__":
    main()