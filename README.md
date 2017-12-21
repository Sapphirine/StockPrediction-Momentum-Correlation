# StockPrediction-Momentum-Correlation

Member: Di Zuo (dz2357), Nijia Yu (jn2585)

Project: Stock prediction based on momentum and correlation

Project ID: 201712-9

Data source: https://stooq.com/db/h/

Note:
1. If you want to run my code, please download the data from stooq (U.S., 5 minutes, ASCII, 75MByte), unzip it.

You should have the following files:
5_us_txt/

pre_5min - nysemkt stocks.py

correlation_5min.py

correlations_0.9.csv

Run pre_5min - nysemkt stocks.py first, then correlation_5min.py.

2. Be aware that everything is tested under Windows 10, the file system is different from Linux in using the slashes.

3. If you want to compute correlation on your own, please uncommnet code between line 297 and 303

4. all_correlations_0.9.csv and all_correlation_0.95.csv are correlations of all stocks, while the other three csv files are correlations of 360 stocks in NYSE market.
