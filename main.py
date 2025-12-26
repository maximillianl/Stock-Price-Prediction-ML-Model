from test_subjects import *




def main():
    # snp500_stocks_api = get_snp500_stocks_api()
    # print(snp500_stocks)
    snp500_stocks = get_snp500_stocks()
    print(snp500_stocks)
    russell1000_stocks = get_russell1000_stocks()
    print(russell1000_stocks)
    russell2000_stocks = get_russell2000_stocks()
    print(russell2000_stocks)
    russell3000_stocks = get_russell3000_stocks()
    print(russell3000_stocks)





if __name__ == "__main__":
    main()