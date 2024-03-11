#Purpose: This file will employ a trading strategy based on ML training and output resulting return in comparison to market.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
import statistics

style.use("ggplot")

FEATURES =  ['DE Ratio',
             'Trailing P/E',
             'Price/Sales',
             'Price/Book',
             'Profit Margin',
             'Operating Margin',
             'Return on Assets',
             'Return on Equity',
             'Revenue Per Share',
             'Market Cap',
             'Enterprise Value',
             'Forward P/E',
             'PEG Ratio',
             'Enterprise Value/Revenue',
             'Enterprise Value/EBITDA',
             'Revenue',
             'Gross Profit',
             'EBITDA',
             'Net Income Avl to Common ',
             'Diluted EPS',
             'Earnings Growth',
             'Revenue Growth',
             'Total Cash',
             'Total Cash Per Share',
             'Total Debt',
             'Current Ratio',
             'Book Value Per Share',
             'Cash Flow',
             'Beta',
             'Held by Insiders',
             'Held by Institutions',
             'Shares Short (as of',
             'Short Ratio',
             'Short % of Float',
             'Shares Short (prior ']

def Build_Data_Set():
    data_df = pd.read_csv("key_stats_acc_perf_NO_NA_enhanced.csv") #with or w/o get similar results

    #data_df = data_df[:100]
    data_df = data_df.reindex(np.random.permutation(data_df.index))
    data_df = data_df.replace(np.nan,0).fillna(0) #could replace with -999 and would be outliers

    X = np.array(data_df[FEATURES].values)

    y = (data_df["Status"]
         .replace("underperform",0)
         .replace("outperform",1)
         .values.tolist())

    X = preprocessing.scale(X)

    Z = np.array(data_df[["stock_p_change","sp500_p_change"]])

    return X,y,Z

def Analysis():

    test_size = 1000

    invest_amount = 10000
    total_invests = 0
    if_market = 0 
    if_strat = 0

    X, y , Z = Build_Data_Set()
    print(len(X))
    #print(np.any(np.isnan(X)))
    #print(np.all(np.isfinite(X)))

    clf = svm.SVC(kernel="linear", C= 1.0) #specifying what our classifier is
    clf.fit(X[:-test_size],y[:-test_size]) #training our classifier. We are leaving out the last 500 data samples to not test on, so you don't test on data you trained against

    correct_count = 0

    for x in range(1,test_size+1):
        if clf.predict(X[[-x]])[0] == y[-x]: #clf.predict outputs answer in a list so we want the 0th element = to the answer we know is true
            correct_count += 1

        if clf.predict(X[[-x]])[0] == 1:
            invest_return = invest_amount + (invest_amount* (Z[-x][0]/100)) #Z[-x][0]? Z is an array of arrays. 0th element will be stock_p_change & 1sth element is sp500_p_change
            market_return = invest_amount + (invest_amount* (Z[-x][1]/100))
            total_invests += 1
            if_market += market_return
            if_strat += invest_return


    print("Accuracy", (correct_count/test_size)*100.00)

    print("Total Trades:", total_invests)
    print("Ending with Strategy",if_strat)
    print("Ending with Market",if_market)

    compared = ((if_strat - if_market)/if_market)*100.0
    do_nothing = total_invests * invest_amount

    avg_market = ((if_market - do_nothing)/do_nothing)*100.0
    avg_strat = ((if_strat - do_nothing)/do_nothing)*100.0


    print("Compared to market, we earn", str(compared)+ "% more")
    print("Average investment return:", str(avg_strat)+"%")
    print("Average market return:", str(avg_market)+"%")


Analysis()