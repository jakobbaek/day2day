import requests
import sys
import pandas as pd
import polars as pl
from datetime import datetime
import yfinance as yf
from datetime import datetime, timedelta
import time
import json

ACCESS_TOKEN = "eyJhbGciOiJFUzI1NiIsIng1dCI6IjI3RTlCOTAzRUNGMjExMDlBREU1RTVCOUVDMDgxNkI2QjQ5REEwRkEifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiSkp8LU45VVBRN09xVldOWnNVb3VrZz09IiwiY2lkIjoiSkp8LU45VVBRN09xVldOWnNVb3VrZz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiNWQyNjk2MzEyMDBiNDQ1OTgwZjBjMjVlOTcwMTliYjIiLCJkZ2kiOiI4NCIsImV4cCI6IjE3NDg2MTIxNzAiLCJvYWwiOiIxRiIsImlpZCI6ImI4OTIzZWM2NThjNDQxMTE1YmFmMDhkZDQxZDkyM2UxIn0.pOgbC2AkhCdY8gJl90FSJNKW6In_nMLqwqs3bs1dFcM68d_39mt2sT7O5PhAqekWI_QYO_N7-dWAIwCZBZPsYA"

def generate_date_intervals(start_date: str, end_date: str, interval=10) -> list:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # List to store the result dates in string format
    dates = []
    delta = timedelta(days=interval)
    
    # Loop through dates, adding a 10-day interval each time
    current_date = start
    while current_date <= end:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += delta
    if end_date not in dates: dates.append(end_date)
    return dates

def curate_danish_stocks(update=False):

    file_name = "cop_saxo_1m"
    d_stocks = get_danish_stocks(ACCESS_TOKEN)
    new_data = []
    if update:
        prev_df = pl.read_csv(f"ins_data/{file_name}.csv")
        #prev_data = {uic:list([dt[:10] for dt in prev_df.filter(pl.col("uic")==uic)["datetime"].to_list()]) for uic in prev_df["uic"].to_list()}
        prev_data = {
        uic: dates
        for uic, dates in (
        prev_df
          .with_columns(pl.col("datetime").str.slice(0, 10).alias("date"))
          .groupby("uic")
          .agg("date")   # ← un‐named agg of "date" ⇒ list[str] per group :contentReference[oaicite:1]{index=1}
          .rows()        # yields (uic, dates) tuples
            )
            }
    for stc in d_stocks["Data"]:
        uic = stc["Identifier"]
        sym = stc["Symbol"]
        name = stc["Description"]
        print (name)
        for dt in generate_date_intervals("2023-01-01","2025-05-29",interval=2):
            if update and uic in prev_data and datetime.strptime(dt, "%Y-%m-%d")-timedelta(days=1) <= datetime.strptime(max(prev_data[uic]), "%Y-%m-%d"):
                continue
            print (dt)
            stc_data = get_saxo_ins_hist(ACCESS_TOKEN,uic,dt)
            for dat in stc_data["Data"]:
                save_data = {"datetime":dat["Time"],
                            "High":dat["High"],
                            "Low":dat["Low"],
                            "Open":dat["Open"],
                            "Close":dat["Close"],
                            "uic":uic,
                            "name":name,
                            "ticker":sym}
                new_data.append(save_data)
            time.sleep(0.95)
    df = pl.DataFrame(new_data)
    if update:
        if not df.is_empty():
            final_df = pl.concat([prev_df,df])
        else:
            final_df = prev_df
    else:
        final_df = df
    final_df = final_df.unique(["uic","datetime"])
    final_df = final_df.sort(["uic","datetime"],descending=False)
    final_df.write_csv(f"ins_data/{file_name}.csv")

def get_danish_stocks(access_token, limit=1000):
    """
    Retrieve a list of Danish stocks from Saxo Bank OpenAPI.
    
    Parameters:
    - access_token (str): Valid OAuth access token for authentication
    - limit (int): Maximum number of results to return (default 1000)
    
    Returns:
    - dict: JSON response containing Danish stocks or error message
    """
    
    url = "https://gateway.saxobank.com/sim/openapi/ref/v1/instruments"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    params = {
        "AssetTypes": "Stock",          # Filter for stocks only
        "ExchangeId": "CSE",           # Nasdaq Copenhagen exchange ID
        "$top": limit,                 # Limit the number of results
        "IncludeNonTradable": "false"  # Exclude non-tradable instruments
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Request failed with status code {response.status_code}",
                "details": response.text
            }
    except requests.exceptions.RequestException as e:
        return {
            "error": "Request exception occurred",
            "details": str(e)
        }
def get_saxo_ins_hist(access_token, instrument_uic, upto_date="2025-10-01"):

    """
    Retrieve historical price data using the /chart/v1/charts endpoint from Saxo Bank OpenAPI.
    
    Parameters:
    - access_token (str): Valid OAuth access token for authentication
    - instrument_uic (int): Unique Instrument Code (UIC) for the desired instrument
    - horizon (int): Time frame in minutes (e.g., 5 for 5-minute intervals)
    - from_date (str): Start date in YYYY-MM-DD format
    - to_date (str): End date in YYYY-MM-DD format
    
    Returns:
    - dict: JSON response containing historical data or error message
    """
    
    url = "https://gateway.saxobank.com/sim/openapi/chart/v3/charts"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    params = {
        "Uic": instrument_uic,
        "AssetType": "Stock",          # Adjust based on instrument type (e.g., FxSpot, Stock)
        "Horizon": 1,            # Time frame in minutes (e.g., 5 for 5-minute)
        "Time": f"{upto_date}T00:00:00Z", # End time for data
        "Mode": "UpTo",                # Fetch data from 'From' date up to 'Time'  # Start time
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Request failed with status code {response.status_code}",
                "details": response.text
            }
    except requests.exceptions.RequestException as e:
        return {
            "error": "Request exception occurred",
            "details": str(e)
        }

def get_stock_historical_data(ticker, start_date=None, end_date=None, interval="5m"):
    """
    Fetch historical OHLC data for a stock ticker using yfinance.

    Parameters:
    - ticker (str): Stock ticker symbol (e.g., 'AAPL').
    - start_date (str, optional): Start date in YYYY-MM-DD. Defaults to 30 days ago.
    - end_date (str, optional): End date in YYYY-MM-DD. Defaults to today.
    - interval (str, optional): Data interval (e.g., '1d' for daily, '1h' for hourly). Default is '1d'.

    Returns:
    - pd.DataFrame: OHLC data, or None if the request fails.
    """
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval=interval)
        if df.empty:
            print(f"No data returned for {ticker}.")
            return None
        
        # Rename columns to match Saxo-style output
        df = df.reset_index().rename(columns={
            "Datetime": "Datetime",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close"
        })
        return df[["Datetime", "Open", "High", "Low", "Close"]]
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None
    
def download_hist_data(recollect=False, start_date="2025-01-01", end_date="2025-02-24", main_file="cop_nasd_5m"):

    currencies = [
    "DKK/HUF", "DKK/CZK", "DKK/PLN", "DKK/SEK", "DKK/BGN", "DKK/CHF",
    "DKK/NOK", "DKK/UAH", "DKK/GBP", "DKK/RON", "DKK/RUB", "DKK/TRY",
    "DKK/AED", "DKK/BHD", "DKK/ILS", "DKK/JOD", "DKK/LBP", "DKK/OMR",
    "DKK/QAR", "DKK/SAR", "DKK/AUD", "DKK/NZD", "DKK/JPY", "DKK/SGD",
    "DKK/THB", "DKK/CNY", "DKK/HKD", "DKK/INR", "DKK/MYR", "DKK/PHP",
    "DKK/TWD", "DKK/IDR", "DKK/KRW", "DKK/LKR", "DKK/NPR", "DKK/PKR",
    "DKK/COP", "DKK/ARS", "DKK/BRL", "DKK/CLP", "DKK/VES", "DKK/BBD",
    "DKK/JMD", "DKK/XCD", "DKK/ZAR", "DKK/EGP", "DKK/KES", "DKK/MAD",
    "DKK/NAD", "DKK/XAF", "DKK/MXN", "DKK/CAD", "DKK/USD", "DKK/PAB",
    "EUR/AUD", "EUR/CAD", "EUR/CHF", "EUR/CNH", "EUR/CZK", "EUR/GBP",
    "EUR/HKD", "EUR/HUF", "EUR/ILS", "EUR/JPY", "EUR/MXN", "EUR/NOK",
    "EUR/NZD", "EUR/PLN", "EUR/SEK", "EUR/SGD", "EUR/TRY", "EUR/USD",
    "EUR/ZAR", "EUR/CNY"
    ]
    
    usd_currency_pairs = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", "USD/BRL", "USD/CNY", "USD/CZK", "USD/DKK", "USD/HKD", "USD/HUF", "USD/IDR", "USD/ILS", "USD/INR", "USD/KRW", "USD/MXN", "USD/MYR", "USD/NOK", "USD/PHP", "USD/PLN", "USD/RON", "USD/RUB", "USD/SEK", "USD/SGD", "USD/THB", "USD/TRY", "USD/TWD", "USD/ZAR"]
    currencies += usd_currency_pairs

    ticker = "NOVO-B.CO"
    data = pl.read_csv("data/copenhagen_nasdaq_list.csv")
    new_data = []
    prev_syms = set([])
    #prev_dfs = pl.read_csv("ins_data/cop_nasd_5m.csv")
    #prev_dfs = prev_dfs.with_columns((pl.col("Datetime").str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S")).alias("Datetime"))
    #prev_syms = set(list(prev_dfs["ticker"].to_list()))
    symbols = list(data["Symbol"].to_list())
    prev_dfs = pl.read_csv(f"ins_data/{main_file}.csv")
    prev_dfs = prev_dfs.with_columns((pl.col("Datetime").str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S")).alias("Datetime"))
    prev_syms = set(list(prev_dfs["ticker"].to_list()))
    #symbols = currencies
    for sym in symbols:
        real_sym = sym.replace(".","-")+".CO"
        #real_sym = sym.replace("/","")+"=X"
        #print (sym)
        #print (real_sym)
        #sys.exit()
        if recollect:
            df = get_stock_historical_data(real_sym, start_date=start_date, end_date=end_date)
            print (real_sym)
            if df is not None:
                print (len(df))
                df["ticker"]=real_sym
                df = pl.from_pandas(df)
                df = df.with_columns((pl.col("Datetime").cast(pl.Utf8).str.slice(0, length=19)))
                df = df.with_columns((pl.col("Datetime").str.to_datetime("%Y-%m-%d %H:%M:%S")).alias("Datetime"))
                new_data = pl.concat([prev_dfs,df])
                #new_data = df
                print (len(new_data))
                new_data = new_data.unique(["ticker","Datetime"])
                print (len(new_data))
                new_data.write_csv(f"ins_data/{main_file}.csv")
                prev_dfs = new_data
                time.sleep(1)
        elif not recollect and real_sym in prev_syms:
            pass
        else:
            df = get_stock_historical_data(real_sym, start_date=start_date, end_date=end_date)
            print (real_sym)
            if df is not None:
                print (len(df))
                df["ticker"]=real_sym
                df = pl.from_pandas(df)
                df = df.with_columns((pl.col("Datetime").cast(pl.Utf8).str.slice(0, length=19)))
                df = df.with_columns((pl.col("Datetime").str.to_datetime("%Y-%m-%d %H:%M:%S")).alias("Datetime"))
                new_data = pl.concat([prev_dfs,df])
                #new_data = df
                new_data.write_csv(f"ins_data/{main_file}.csv")
                prev_dfs = new_data
                time.sleep(1)
            else:
                print ("NO DATA")

if __name__ == "__main__":  

    #download_hist_data(recollect=True, start_date="2025-02-19", end_date="2025-03-19")

    #sys.exit()

    curate_danish_stocks()
    sys.exit()

    result = get_danish_stocks(ACCESS_TOKEN)
    for d in result["Data"]:
        print (d)
    print (len(result["Data"]))
    sys.exit()

    uic = 15629  # Example: EURUSD
    result = get_saxo_ins_hist(ACCESS_TOKEN, uic, "2025-01-01", "2025-05-27")
    print(json.dumps(result, indent=2))
    sys.exit()

    try:
        instruments = list_instruments(ACCESS_TOKEN, query_params=None)
        print("Available Instruments:")
        print(instruments)
    except Exception as e:
        print("Error retrieving instruments:", e)
