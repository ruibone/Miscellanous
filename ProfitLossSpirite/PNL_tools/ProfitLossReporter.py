''' Import packages '''
import os
import requests
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from bs4 import BeautifulSoup


''' Class of ProfitSpirits '''
'''  property
statement -> statement dataframe download from fubon app
startdate -> time range for statement
enddate -> time range for statement
all_stock_name -> list of all stocks I used to possess
closing_price -> the closing stock price crawled from TWSE website
unrealized_summary -> summary of unrealized pnl
realized_summary -> summary of realized pnl
diviend_history -> a dict recording diviend history of each stock
'''
# preprocessing fubon statement -> stock price crawler -> realized & unrealized pnl -> diviend crawler -> diviend profit -> total pnl from stock market -> dashboard -> connect to MySQL for data storage
class PNLReporter:
    
    # input statement dataframe
    def __init__(self, statement):
        self.statement = statement
        self.all_stock_name = self.statement['股票名稱'].unique()


    # select given time interval
    def interval_selector(self, startdate = '2021-01-01', enddate = 'now'):
        self.startdate = startdate 
        if enddate == 'now':  # determine whcih date is the last trading day
            latest_date = datetime.date.today() 
            if latest_date.isoweekday() == 6:
                latest_date -= datetime.timedelta(days = 1)
            elif latest_date.isoweekday() == 7:
                latest_date -= datetime.timedelta(days = 2)
            self.enddate = str(latest_date)
        else:
            self.enddate = enddate
        
        time_filter = [np.all([x >= pd.Timestamp(self.startdate), x <= pd.Timestamp(self.enddate)]) for x in self.statement['成交日期']]
        self.statement = self.statement[time_filter]
        print(f'StartDate: {self.startdate}; EndDate: {self.enddate};')    
        

    # lookup for the stock price at the enddate (download data from Taiwan Stock Exchange Center & Taipei Exchange)
    def price_lookup(self):
        # web crawler from TWSE
        target_date = self.enddate.replace('-', '')
        url_twse = f'https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={target_date}&type=ALL&response=csv'
        req_twse = requests.get(url_twse).text
        price_twse = req_twse.split('\n')

        # get the stock price from twse
        lookup_stockname = [x.split('(')[0] for x in self.all_stock_name] + ['證券名稱']
        lookup_stocksymbol = ['"' + x.split('(')[1][:-1] + '"' for x in self.all_stock_name] + ['證券代號']
        lookup_info = list()
        for row in price_twse:
            if np.any([np.all([x in row, y in row]) for x, y in zip(lookup_stockname, lookup_stocksymbol)]):
                lookup_info.append(row)        

        # data cleansing
        all_price_list = list()
        for row in lookup_info:
            element = row.split('"')[1::2]
            all_price_list.append(element)

        price_df = pd.DataFrame(all_price_list)
        price_df.columns = price_df.iloc[0]
        price_df = price_df.drop(index = 0)   
        price_key =  price_df['證券名稱'] + '(' + price_df['證券代號'] + ')'
        price_value = price_df['收盤價']
        price_dict = dict()
        for key, value in zip(price_key, price_value):
            price_dict[key] = float(value)
        self.closing_price = price_dict

        # web crawler from Taipei Exchange
        t_year, t_month, t_date = str(int(target_date[0:4]) - 1911), target_date[4:6], target_date[6:]
        url_tpex = f'https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php?l=zh-tw&o=htm&d={t_year}/{t_month}/{t_date}&se=EE&s=0,asc,0'
        req_tpex = requests.get(url_tpex).text
        html_tpex = BeautifulSoup(req_tpex, 'html.parser')
        table_tpex = html_tpex.find('table')
        record_tpex = table_tpex.find_all('tr')

        # reconstruct the table
        rows_tpex = list()
        for idx, row in enumerate(record_tpex[1:]):
            temp_row = [x.get_text() for x in row.find_all('td')]
            if idx == 0:
                header = temp_row
            else:
                rows_tpex.append(temp_row)
        price_tpex = pd.DataFrame(rows_tpex, columns = header)   

        # get the stock price in tpex     
        for idx, row in price_tpex.iterrows():
            keyname = str(row['名稱']) + '(' + str(row['代號']) + ')'
            if keyname in self.all_stock_name:
                price_dict[keyname] = row['收盤']
            

    # calculate unrealized profit of each stock & create a unrealized profit table
    def unrealized_calculator(self):
        # sum up all exchange record to find all the unrealized stocks
        state = self.statement.copy()
        state['持有股數'] = (-1)*state['成交股數']*state['淨收付金額']/abs(state['淨收付金額'])
        state['持有股數'] = state['持有股數'].astype('int')
        state_arrange = state.drop(columns = ['成交日期', '交易類別', '成交單價'])
        state_group = state_arrange.groupby(['股票名稱']).sum()
        state_group['收盤價'] =  [self.closing_price[x] for x in state_group.index]
        state_group['未實現總價'] = state_group['持有股數']*state_group['收盤價'].astype('float')

        # calculate average purchase price & pnl
        unrealized_df = state_group[state_group['未實現總價'] > 0].copy()
        avg_price, total_cost = list(), list()
        for idx, row in unrealized_df.iterrows():
            if row['持有股數'] == row['成交股數']:
                temp_avg = (-1)*row['淨收付金額'] / row['持有股數']
                temp_cost = abs(row['淨收付金額'])
            else:
                target_state = self.statement[self.statement['股票名稱'] == idx].copy()
                purchase_state = target_state[target_state['淨收付金額'] < 0]
                temp_avg = (-1)*np.sum(purchase_state['淨收付金額']) / np.sum(purchase_state['成交股數'])
                temp_cost = temp_avg*row['持有股數']
            avg_price.append(temp_avg)
            total_cost.append(temp_cost)
        unrealized_df['成交均價'] = np.round(avg_price, 2)
        unrealized_df['付出成本'] = np.round(total_cost, 0)
        unrealized_df['未實現損益'] = np.round(unrealized_df['未實現總價'] - avg_price*unrealized_df['持有股數'], 0)
        unrealized_df['未實現損益率'] = np.round(unrealized_df['未實現損益'] / unrealized_df['付出成本'] * 100, 2)
        unrealized_df['未實現損益率'] = unrealized_df['未實現損益率'].astype('str') + '%'

        # summarize all unrealized pnl information
        unrealized_summary = dict()
        unrealized_summary['total_invest'] = int(np.sum(avg_price*unrealized_df['持有股數']))
        unrealized_summary['total_value'] = int(np.sum(unrealized_df['未實現總價']))
        unrealized_summary['total_pnl'] = int(np.sum(unrealized_df['未實現損益']))
        unrealized_summary['total_pnl%'] = np.round((unrealized_summary['total_pnl'] / unrealized_summary['total_invest'])*100, 2)
        unrealized_summary['table'] = unrealized_df[['持有股數', '成交均價', '付出成本', '收盤價', '未實現總價', '未實現損益', '未實現損益率']]       
        self.unrealized_summary = unrealized_summary

        print('--------------------------未實現損益結算--------------------------')
        print('投資成本: ', unrealized_summary['total_invest'])
        print('投資現值: ', unrealized_summary['total_value'])
        print('未實現總損益: ', unrealized_summary['total_pnl'])
        print('未實現總損益率: ', str(unrealized_summary['total_pnl%']) + '%')
        print('-----------------------------------------------------------------')

     # calculate realized profit of each stock & create 2 realized profit tables
    def realized_calculator(self):
        # calculate average pruchase price for each stock
        avg_price_dict = dict()
        for stock in self.all_stock_name:
            state_df = self.statement.copy()
            target_state = state_df[state_df['股票名稱'] == stock]
            target_purchase = target_state[target_state['淨收付金額'] < 0]
            avg_price = (-1)*np.sum(target_purchase['淨收付金額']) / np.sum(target_purchase['成交股數'])
            avg_price_dict[stock] = avg_price

        # select purchase record from statement and calculate pnl for each records
        state_purchase = state_df[state_df['淨收付金額'] > 0].reset_index(drop = True)
        state_purchase['購入均價'] = [avg_price_dict[x] for x in state_purchase['股票名稱']]
        state_purchase['購入均價'] = state_purchase['購入均價'].astype('float')
        state_purchase['單筆成本'] = np.round(state_purchase['購入均價']*state_purchase['成交股數'], 0)
        state_purchase['單筆損益'] = state_purchase['淨收付金額'] - state_purchase['單筆成本']
        state_purchase['單筆損益率'] = np.round((state_purchase['單筆損益'] / state_purchase['單筆成本'])*100, 2)
        state_purchase['單筆損益率'] = state_purchase['單筆損益率'].astype('str') + '%'
        state_realized = state_purchase[['成交日期', '股票名稱', '成交股數', '成交單價', '淨收付金額', '購入均價', '單筆成本', '單筆損益', '單筆損益率']]

        # group by each stock and calculate pnl
        sub_state = state_df[state_df['淨收付金額'] > 0].reset_index(drop = True)
        sub_purchase = sub_state.drop(columns = ['成交日期', '交易類別']).groupby('股票名稱').sum()
        sub_purchase['購入均價'] = [avg_price_dict[x] for x in sub_purchase.index]
        sub_purchase['購入均價'] = np.round(sub_purchase['購入均價'].astype('float'), 2)
        sub_purchase['購入成本'] = np.round(sub_purchase['購入均價']*sub_purchase['成交股數'], 0)
        sub_purchase['損益'] = sub_purchase['淨收付金額'] - sub_purchase['購入成本']
        sub_purchase['損益率'] = np.round((sub_purchase['損益'] / sub_purchase['購入成本'])*100, 2)
        sub_purchase['損益率'] = sub_purchase['損益率'].astype('str') + '%'
        sub_realized = sub_purchase[['成交股數', '淨收付金額', '購入均價', '購入成本', '損益', '損益率']]

        # summarize all realized pnl information
        realized_summary = dict()
        realized_summary['trading_record'] = state_realized
        realized_summary['group_record'] = sub_realized
        realized_summary['total_invest'] = int(np.sum(sub_realized['購入成本']))
        realized_summary['total_value'] = int(np.sum(sub_realized['淨收付金額']))
        realized_summary['realized_pnl'] = int(np.sum(sub_realized['損益']))
        realized_summary['realized_pnl%'] = np.round((realized_summary['realized_pnl'] / realized_summary['total_invest'])*100, 2)
        self.realized_summary = realized_summary

        print('--------------------------已實現損益結算--------------------------')
        print('投資成本: ', realized_summary['total_invest'])
        print('投資報酬: ', realized_summary['total_value'])
        print('已實現總損益: ', realized_summary['realized_pnl'])
        print('已實現總損益率: ', str(realized_summary['realized_pnl%']) + '%')
        print('-----------------------------------------------------------------')


    # lookup diviend history from TSWE & TPEX
    def diviend_lookup(self):
        # TWSE crawler
        url_twse = 'https://www.twse.com.tw/rwd/zh/exRight/TWT49U?'
        payload_twse = {
            'startDate': self.startdate.replace('-', ''),
            'endDate': self.enddate.replace('-', ''),
            '_': 1708569071037
        }
        res_twse = requests.post(url_twse, payload_twse)

        # organize diviend table from TWSE
        json_twse = res_twse.json()
        pd_twse = pd.DataFrame(json_twse['data'], columns = json_twse['fields'])
        pd_twse.columns = ['date_ROC', 'stock_code', 'stock_name', 'before_ex-diviend', 'reference_ex-diviend', 'diviend', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        pd_twse['stock'] = pd_twse['stock_name'] + '(' + pd_twse['stock_code'] + ')'
        date_split = pd_twse['date_ROC'].str.replace('月', '-').str.replace('日', '').str.split('年')
        date_combine = [str(int(x[0]) + 1911) + '-' + str(x[1]) for x in date_split]
        pd_twse['date'] = date_combine
        hist_twse = pd_twse[['date', 'stock', 'before_ex-diviend', 'reference_ex-diviend', 'diviend']].copy()

        # TPEX crawler
        start_AD = (str(int(self.startdate[0:4]) - 1911) + self.startdate[4:]).replace('-', '/')
        end_AD = str(int(self.enddate[0:4]) - 1911) + self.enddate[4:].replace('-', '/')
        url_tpex = f'https://www.tpex.org.tw/web/stock/exright/dailyquo/exDailyQ_result.php?l=zh-tw&d={start_AD}&ed={end_AD}&_=1708702284675'
        res_tpex = requests.post(url_tpex)

        # organize table from TPEX
        list_tpex = res_tpex.json()['aaData']
        column_name = ['date_ROC', 'stock_code', 'stock_name', 'before_ex-diviend', "reference_ex-diviend", 'a', 'b', 'diviend', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
        pd_tpex = pd.DataFrame(list_tpex, columns = column_name)
        pd_tpex['stock'] = pd_tpex['stock_name'].str.strip() + '(' + pd_tpex['stock_code'] + ')'
        date_split = pd_tpex['date_ROC'].str.split('/')
        date_combine = list(map(lambda x: str(int(x[0]) + 1911) + '-' + str(x[1]) + '-' + str(x[2]), date_split))
        pd_tpex['date'] = date_combine
        hist_tpex = pd_tpex[['date', 'stock', 'before_ex-diviend', "reference_ex-diviend", 'diviend']].copy()

        hist_divi = pd.concat([hist_twse, hist_tpex], axis = 0).reset_index(drop = True)
        hist_divi['date'] = pd.to_datetime(hist_divi['date'])
        hist_divi['diviend'] = hist_divi['diviend'].astype('float')
        hist_divi['before_ex-diviend'] = hist_divi['before_ex-diviend'].str.replace(',', '').astype('float')
        hist_divi['diviend_yield'] = np.round(hist_divi['diviend'] / hist_divi['before_ex-diviend'], 4)

        # filter out targeted stocks
        divi_dict = dict()
        for stock in self.all_stock_name:
            temp_df = hist_divi[hist_divi['stock'] == stock].reset_index(drop = True)
            if len(temp_df) > 0:
                divi_dict[stock] = temp_df
        self.diviend_history = divi_dict

    
    # calculate and sum up dividend profit from all stocks
    def diviend_calculator(self):
        # select a stock, view each dividend record and calculate how many share of stocks we had then
        cumulation_dict = dict()
        total_divi = 0
        for stock in self.diviend_history:
            divi_df = self.diviend_history[stock]
            state_df = self.statement[self.statement['股票名稱'] == stock].copy()
            state_df['股數變化'] = (-1)*state_df['成交股數']*(abs(state_df['淨收付金額']) / state_df['淨收付金額'])

            cumulation_divi = 0
            for _, record in divi_df.iterrows():
                divi_date = record['date']
                early_state = state_df[state_df['成交日期'] < divi_date]
                early_share = np.sum(early_state['股數變化'])
                if early_share == 0:
                    continue
                temp_divi = np.max([early_share*record['diviend'] - 10, 10])  # deduct the transaction fee
                cumulation_divi += temp_divi

            if cumulation_divi > 0:
                cumulation_dict[stock] = int(cumulation_divi)
                total_divi += cumulation_divi
        
        self.diviend_profit = cumulation_dict
        print('--------------------------除權息結果結算--------------------------')
        print('總配息收益: ', int(total_divi))
        print('-----------------------------------------------------------------')
