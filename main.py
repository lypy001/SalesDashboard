# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import itertools
import os
from math import sqrt

import matplotlib as matplotlib
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from pmdarima import auto_arima
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime
from dateutil import relativedelta

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import pmdarima as pm
from pandas.tseries.offsets import DateOffset
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
import streamlit as st
import plotly.graph_objects as go
# from djangoProject1.wsgi import get_wsgi_application
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
# application = get_wsgi_application()
# implementing streamlit
st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

# def check_password():
#     """Returns `True` if the user had the correct password."""

#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         if st.session_state["password"] == st.secrets["password"]:
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # don't store password
#         else:
#             st.session_state["password_correct"] = False

#     if "password_correct" not in st.session_state:
#         # First run, show input for password.
#         st.text_input(
#             "Password", type="password", on_change=password_entered, key="password"
#         )
#         return False
#     elif not st.session_state["password_correct"]:
#         # Password not correct, show input + error.
#         st.text_input(
#             "Password", type="password", on_change=password_entered, key="password"
#         )
#         st.error("😕 Password incorrect")
#         return False
#     else:
#         # Password correct.
#         return True

# if check_password():
#     st.write("Here goes your normal Streamlit app...")
#     st.button("Click me")

DATA_URL = ('RetailSalesIndex.csv')
@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)

    return data
data = load_data()

#data = pd.read_csv('RetailSalesIndex.csv')
pd.set_option('display.max_columns', None)
# Check Atrribute In File
# print(data.head(5))
# Drop columns that is not needed
data = data.drop(columns=['Total'])
data = data.drop(columns=['Total (Excluding Motor Vehicles)'])
data = data.drop(columns=['Others'])
# Check Data Size
print(data.size)

# Check For Null values(NO NULL VALUES)
# print(data.isnull().sum())

# If there is null value
data = data.dropna()
# print(data.isnull().sum())

# Check Dtype in data
# print(data.info())

# Statistical detail of dataset
# print(data.describe())

# #Split attribute Date into Month Year
# string obj convert to datetime
data['Date'] = pd.to_datetime(data['Data Series'])
data = data.drop(columns=['Data Series'])
# Make Data the index
data.index = pd.to_datetime(data['Date'])
# data.index = pd.DatetimeIndex(data.index)
data.drop(columns='Date', inplace=True)
data.sort_values(by='Date', inplace=True)

dept_stores_data = data[['Department Stores']]

# #dept_stores_data.rename(index = str, columns={'Department Stores': 'Sales'}, inplace = True)
#
supermarket_data = data[['Supermarkets & Hypermarkets']]

mart_data = data[['Mini-Marts & Convenience Stores']]
food_data = data[['Food & Alcohol']]
motorvhe_data = data[['Motor Vehicles']]
petrol_data = data[['Petrol Service Stations']]
cosemetic_data = data[['Cosmetics, Toiletries & Medical Goods']]
apparel_data = data[['Wearing Apparel & Footwear']]
hseholdEqm_data = data[['Furniture & Household Equipment']]
rctGoods_data = data[['Recreational Goods']]
accessory_data = data[['Watches & Jewellery']]
electronics_data = data[['Computer & Telecommunications Equipment']]
optical_data = data[['Optical Goods & Books']]

#
# data = data.drop('2022-01-01')
# data = data.drop('2022-02-01')




#
# ADF Test reference from https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/
def ADFtest(name):
    result = adfuller(data[name].dropna())
    print("ADF test for:" + name)
    print(f'ADF Statistic: {result[0]}')
    print(f'n_lags: {result[1]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')


def kpss_test(name):
    print("Results of KPSS Test:")
    kpsstest = kpss(data[name].dropna(), regression="c")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)


# ADFtest('Department Stores')
# kpss_test('Department Stores')

# ADFtest('Supermarkets & Hypermarkets')
# ADFtest('Mini-Marts & Convenience Stores')
# ADFtest('Food & Alcohol')
# ADFtest('Motor Vehicles')
# ADFtest('Petrol Service Stations')
# ADFtest('Cosmetics, Toiletries & Medical Goods')
# ADFtest('Wearing Apparel & Footwear')
# ADFtest('Furniture & Household Equipment')
# ADFtest('Recreational Goods')
# ADFtest('Watches & Jewellery')
# ADFtest('Computer & Telecommunications Equipment')
# ADFtest('Optical Goods & Books')
#
# Make data stationary by calculating the difference in sales month over month
def get_diff(data, name):
    # data['Sales_diff'] = data[name] - data[name].shift(12)
    # print(data[[name, 'Sales_diff']].head(13))

    data['Sales_diff'] = data[name].diff()
    data = data.dropna()

    # data = data[['Date','Sales_diff', name]]
    # #ploting the date and sales diff
    # plt.plot_date(data.index, data['Sales_diff'], linestyle='-')
    # plt.xlabel(name)
    #
    #
    # plt.show()


#     return data



# stationary_data = get_diff( data, 'Supermarkets & Hypermarkets' )
# ADFtest('Sales_diff')
# kpss_test('Sales_diff')
# search for the best perimeters for SARIMAX
# reference from https://www.bounteous.com/insights/2020/09/15/forecasting-time-series-model-using-python-part-two/
# def sarima_grid_search(y, seasonal_period):
#     p = d = q = range(0, 2)
#     pdq = list(itertools.product(p, d, q))
#     seasonal_pdq = [(x[0], x[1], x[2], seasonal_period) for x in list(itertools.product(p, d, q))]
#
#     mini = float('+inf')
#
#     for param in pdq:
#         for param_seasonal in seasonal_pdq:
#             try:
#                 mod = sm.tsa.statespace.SARIMAX(y,
#                                                 order=param,
#                                                 seasonal_order=param_seasonal,
#                                                 enforce_stationarity=False,
#                                                 enforce_invertibility=False)
#
#                 results = mod.fit()
#
#                 if results.aic < mini:
#                     mini = results.aic
#                     param_mini = param
#                     param_seasonal_mini = param_seasonal
#
#             #                 print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
#             except:
#                 continue
#     print('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini,
#                                                                                        mini))


# print(sarima_grid_search(supermarket_data,12))


# method for prediction
# train = supermarket_data[:110]
# test = supermarket_data[110:]
# final_model=sm.tsa.statespace.SARIMAX(train,order=(0,1,1),seasonal_order=(1,1,1,12))
# results = final_model.fit()
# time_index = supermarket_data.reset_index()[['Date']] #replace utc with your index name
# supermarket_data = supermarket_data.reset_index()
#
# start=len(train)
# end=len(train)+len(test)-1
# prediction = results.predict(start= start, end=end, dynamic=False, typ='levels').rename('SARIMA(1,1,1)(2,1,2,12) Predictions')
# prediction = pd.DataFrame(prediction)
# prediction = prediction.join(time_index)
# prediction.set_index('Date', inplace=True)
# print(prediction)
# close

# Another method for prediction
# model = sm.tsa.statespace.SARIMAX(supermarket_data, order=(0, 1, 1), seasonal_order=(1, 1, 2, 12))
# results = model.fit()

# supermarket_data['forecast']=results.predict(start=169,end= 189,dynamic=True)
#
# supermarket_data[['Supermarkets & Hypermarkets','forecast']].plot(figsize=(12,8))
# plt.show()
#
# pred_date=[data.index[-1]+ DateOffset(months=x)for x in range(0,24)]
# pred_date=pd.DataFrame(index=pred_date[1:],columns=supermarket_data.columns)
# print(supermarket_data.info())
# #supermarket_data=pd.concat([data,pred_date])
# supermarket_data= pd.concat([supermarket_data,pred_date])
# supermarket_data['Supermarkets & Hypermarkets'] = supermarket_data['Supermarkets & Hypermarkets'].astype(np.float64)
#
#
# supermarket_data['forecast']=results.predict(start=165,end= 189,dynamic=True)
# supermarket_data[['Supermarkets & Hypermarkets','forecast']].plot(figsize=(12,8))
# plt.show()

# data['Supermarkets & Hypermarkets'] = data['Supermarkets & Hypermarkets'].astype(np.float64)



def save_uploadedfile(uploadedfile):
    with open(os.path.join( uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} ".format(uploadedfile.name))

# request sales file from SME
datafile = st.sidebar.file_uploader("Upload CSV",type=['csv'])
if datafile is not None:
   file_details = {"FileName":datafile.name,"FileType":datafile.type}
   df  = pd.read_csv(datafile)
   st.dataframe(df)
   save_uploadedfile(datafile)





selected_industry = st.sidebar.selectbox('Select a industry', options=list(data.columns))
# selected_data = data[select_industry]
chart_visual = st.sidebar.selectbox('Select Charts/Plot type',
                                    ('Line Chart', 'Bar Chart'))
selecteddate = st.sidebar.date_input("Sales Prediction for next month", datetime.date.today())
nextmonth = selecteddate.replace(day=1) + relativedelta.relativedelta(months=1)

startdate = st.sidebar.date_input("Predict Start Date", datetime.date.today())
enddate = startdate.replace(day=1) + relativedelta.relativedelta(months=12)

# st.line_chart(supermarket_data[['Supermarkets & Hypermarkets','forecast']])

fig = go.Figure()



if selected_industry == 'Department Stores':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime("%Y-%m-%d"))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
        model = sm.tsa.statespace.SARIMAX(dept_stores_data[selected_industry], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=dept_stores_data.columns)

        dept_stores_data = pd.concat([dept_stores_data, pred_date])
        dept_stores_data[selected_industry] = dept_stores_data[selected_industry].astype(np.float64)

        dept_stores_data['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"),
                                                       end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(dept_stores_data[[selected_industry, 'forecast']])
            
        if chart_visual == 'Bar Chart':
            st.write("Bar Chart For Prediction Value")
            st.bar_chart(dept_stores_data[['forecast']], width=250, height=250)



if selected_industry == 'Supermarkets & Hypermarkets':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
        model = sm.tsa.statespace.SARIMAX(supermarket_data['Supermarkets & Hypermarkets'], order=(0, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=supermarket_data.columns)

        supermarket_data = pd.concat([supermarket_data, pred_date])
        supermarket_data['Supermarkets & Hypermarkets'] = supermarket_data['Supermarkets & Hypermarkets'].astype(
            np.float64)

        supermarket_data['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"),
                                                       end=enddate.strftime("%Y-%m-%d"), dynamic=True)


        if chart_visual == 'Line Chart':
            st.line_chart(supermarket_data[[selected_industry, 'forecast']])
            value = supermarket_data.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.write("Bar Chart For Prediction Value")
            st.bar_chart(supermarket_data[['forecast']], width=250, height=250)

if selected_industry == 'Mini-Marts & Convenience Stores':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
        model = sm.tsa.statespace.SARIMAX(mart_data[selected_industry], order=(0, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=mart_data.columns)

        mart_data = pd.concat([mart_data, pred_date])
        mart_data[selected_industry] = mart_data[selected_industry].astype(np.float64)

        mart_data['forecast'] = results.predict(start=169, end=194, dynamic=True)

        if chart_visual == 'Line Chart':
            st.line_chart(mart_data[[selected_industry, 'forecast']])
            value = mart_data.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.write("Bar Chart For Prediction Value")
            st.bar_chart(mart_data[['forecast']], width=250, height=250)



if selected_industry == 'Food & Alcohol':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
        model = sm.tsa.statespace.SARIMAX(food_data[selected_industry], order=(0, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=food_data.columns)

        food_data = pd.concat([food_data, pred_date])
        food_data[selected_industry] = food_data[selected_industry].astype(np.float64)

        food_data['forecast'] = results.predict(start=169, end=194, dynamic=True)

        if chart_visual == 'Line Chart':
            st.line_chart(food_data[[selected_industry, 'forecast']])
            value = food_data.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.write("Bar Chart For Prediction Value")
            st.bar_chart(food_data[['forecast']], width=250, height=250)


if selected_industry == 'Motor Vehicles':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
        model = sm.tsa.statespace.SARIMAX(motorvhe_data[selected_industry], order=(1, 1, 1),
                                          seasonal_order=(0, 1, 1, 12))
        results = model.fit()

        pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=food_data.columns)

        motorvhe_data = pd.concat([motorvhe_data, pred_date])
        motorvhe_data[selected_industry] = motorvhe_data[selected_industry].astype(np.float64)

        motorvhe_data['forecast'] = results.predict(start=169, end=194, dynamic=True)

        if chart_visual == 'Line Chart':
            st.line_chart(motorvhe_data[[selected_industry, 'forecast']])
            value = motorvhe_data.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.write("Bar Chart For Prediction Value")
            st.bar_chart(motorvhe_data[['forecast']], width=250, height=250)


if selected_industry == 'Petrol Service Stations':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
        model = sm.tsa.statespace.SARIMAX(petrol_data[selected_industry], order=(1, 0, 1),
                                          seasonal_order=(0, 1, 1, 12))
        results = model.fit()

        pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=food_data.columns)

        petrol_data = pd.concat([petrol_data, pred_date])
        petrol_data[selected_industry] = petrol_data[selected_industry].astype(np.float64)

        petrol_data['forecast'] = results.predict(start=169, end=194, dynamic=True)

        if chart_visual == 'Line Chart':
            st.line_chart(petrol_data[[selected_industry, 'forecast']])
            value = petrol_data.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.write("Bar Chart For Prediction Value")
            st.bar_chart(petrol_data[['forecast']], width=250, height=250)


if selected_industry == 'Cosmetics, Toiletries & Medical Goods':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
            model = sm.tsa.statespace.SARIMAX(cosemetic_data[selected_industry], order=(0, 1, 1),
                                              seasonal_order=(0, 1, 1, 12))
            results = model.fit()

            pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
            pred_date = pd.DataFrame(index=pred_date[1:], columns=food_data.columns)

            cosemetic_data = pd.concat([cosemetic_data, pred_date])
            cosemetic_data[selected_industry] = cosemetic_data[selected_industry].astype(np.float64)

            cosemetic_data['forecast'] = results.predict(start=169, end=194, dynamic=True)


            if chart_visual == 'Line Chart':
                st.line_chart(cosemetic_data[[selected_industry, 'forecast']])
                value = cosemetic_data.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
                if value > 0:
                    st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
                else:
                    st.write("Prediction: Sales Decrease For Next Month")
            if chart_visual == 'Bar Chart':
                st.write("Bar Chart For Prediction Value")
                st.bar_chart(cosemetic_data[['forecast']], width=250, height=250)


if selected_industry == 'Wearing Apparel & Footwear':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
        model = sm.tsa.statespace.SARIMAX(apparel_data[selected_industry], order=(0, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=food_data.columns)

        apparel_data = pd.concat([apparel_data, pred_date])
        apparel_data[selected_industry] = apparel_data[selected_industry].astype(np.float64)

        apparel_data['forecast'] = results.predict(start=169, end=194, dynamic=True)

        if chart_visual == 'Line Chart':
            st.line_chart(apparel_data[[selected_industry, 'forecast']])
            value = apparel_data.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.write("Bar Chart For Prediction Value")
            st.bar_chart(apparel_data[['forecast']], width=250, height=250)


if selected_industry == 'Furniture & Household Equipment':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
        model = sm.tsa.statespace.SARIMAX(hseholdEqm_data[selected_industry], order=(1, 1, 1),
                                          seasonal_order=(0, 1, 1, 12))
        results = model.fit()

        pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=food_data.columns)

        hseholdEqm_data = pd.concat([hseholdEqm_data, pred_date])
        hseholdEqm_data[selected_industry] = hseholdEqm_data[selected_industry].astype(np.float64)

        hseholdEqm_data['forecast'] = results.predict(start=169, end=194, dynamic=True)

        if chart_visual == 'Line Chart':
            st.line_chart(hseholdEqm_data[[selected_industry, 'forecast']])
            value = hseholdEqm_data.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.write("Bar Chart For Prediction Value")
            st.bar_chart(hseholdEqm_data[['forecast']], width=250, height=250)


if selected_industry == 'Recreational Goods':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
        model = sm.tsa.statespace.SARIMAX(rctGoods_data[selected_industry], order=(1, 0, 1),
                                          seasonal_order=(0, 1, 1, 12))
        results = model.fit()

        pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=food_data.columns)

        rctGoods_data = pd.concat([rctGoods_data, pred_date])
        rctGoods_data[selected_industry] = rctGoods_data[selected_industry].astype(np.float64)

        rctGoods_data['forecast'] = results.predict(start=169, end=194, dynamic=True)

        if chart_visual == 'Line Chart':
            st.line_chart(rctGoods_data[[selected_industry, 'forecast']])
            value = rctGoods_data.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.write("Bar Chart For Prediction Value")
            st.bar_chart(rctGoods_data[['forecast']], width=250, height=250)


if selected_industry == 'Watches & Jewellery':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
        model = sm.tsa.statespace.SARIMAX(accessory_data[selected_industry], order=(1, 0, 1),
                                          seasonal_order=(0, 1, 1, 12))
        results = model.fit()

        pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=food_data.columns)

        accessory_data = pd.concat([accessory_data, pred_date])
        accessory_data[selected_industry] = accessory_data[selected_industry].astype(np.float64)

        accessory_data['forecast'] = results.predict(start=169, end=194, dynamic=True)

        if chart_visual == 'Line Chart':
            st.line_chart(accessory_data[[selected_industry, 'forecast']])
            value = accessory_data.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.write("Bar Chart For Prediction Value")
            st.bar_chart(accessory_data[['forecast']], width=250, height=250)


if selected_industry == 'Computer & Telecommunications Equipment':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
        model = sm.tsa.statespace.SARIMAX(electronics_data[selected_industry], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=food_data.columns)

        electronics_data = pd.concat([electronics_data, pred_date])
        electronics_data[selected_industry] = electronics_data[selected_industry].astype(np.float64)

        electronics_data['forecast'] = results.predict(start=169, end=194, dynamic=True)

        if chart_visual == 'Line Chart':
            st.line_chart(electronics_data[[selected_industry, 'forecast']])
            value = electronics_data.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.write("Bar Chart For Prediction Value")
            st.bar_chart(electronics_data[['forecast']], width=250, height=250)


if selected_industry == 'Optical Goods & Books':
    if datafile is not None:

        df = pd.read_csv(datafile.name)


        df['Date'] = pd.to_datetime(df['Data Series'])
        df = df.drop(columns=['Data Series'])
        # Make Data the index
        df.index = pd.to_datetime(df['Date'])

        df.drop(columns='Date', inplace=True)
        df.sort_values(by='Date', inplace=True)
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1),
                                          seasonal_order=(1, 1, 1, 12))
        results = model.fit()

        pred_date = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=df.columns)

        df = pd.concat([df, pred_date])
        df["Sales"] = df["Sales"].astype(np.float64)

        df['forecast'] = results.predict(start=startdate.strftime("%Y-%m-%d"), end=enddate.strftime("%Y-%m-%d"), dynamic=True)
        if chart_visual == 'Line Chart':
            st.line_chart(df[['Sales', 'forecast']])
            value = df.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.bar_chart(df[['forecast']], width=250, height=250)
            st.write('Bar Chart for '+startdate.strftime("%Y-%m-%d")+' '+enddate.strftime("%Y-%m-%d"))
    else:
        model = sm.tsa.statespace.SARIMAX(optical_data[selected_industry], order=(1, 1, 1),
                                          seasonal_order=(0, 1, 1, 12))
        results = model.fit()

        pred_date = [data.index[-1] + DateOffset(months=x) for x in range(0, 24)]
        pred_date = pd.DataFrame(index=pred_date[1:], columns=food_data.columns)

        optical_data = pd.concat([optical_data, pred_date])
        optical_data[selected_industry] = optical_data[selected_industry].astype(np.float64)

        optical_data['forecast'] = results.predict(start=169, end=194, dynamic=True)

        if chart_visual == 'Line Chart':
            st.line_chart(optical_data[[selected_industry, 'forecast']])
            value = optical_data.diff()._get_value(nextmonth.strftime("%Y-%m-%d"), 'forecast')
            if value > 0:
                st.write("Prediction: Sales Increase For Next Month " + nextmonth.strftime('%B %Y'))
            else:
                st.write("Prediction: Sales Decrease For Next Month")
        if chart_visual == 'Bar Chart':
            st.write("Bar Chart For Prediction Value")
            st.bar_chart(optical_data[['forecast']], width=250, height=250)



# Close


# Observing lags 12 month
# Reference from https://github.com/mollyliebeskind/sales_forecasting/blob/master/notebooks/01_data_cleaning_and_eda.ipynb
# def plots(data, lags=None):
#     # Convert dataframe to datetime index
#     data = data.set_index('Date')
#     data.dropna(axis=0)
#
#     layout = (1, 3)
#     raw = plt.subplot2grid(layout, (0, 0))
#     acf = plt.subplot2grid(layout, (0, 1))
#     pacf = plt.subplot2grid(layout, (0, 2))
#
#     data.plot(ax=raw, figsize=(12, 5), color='mediumblue')
#     smt.graphics.plot_acf(data, lags=lags, ax=acf, color='mediumblue')
#     smt.graphics.plot_pacf(data, lags=lags, ax=pacf, color='mediumblue')
#     sns.despine()
#     plt.tight_layout()
#     plt.show()
#
# plots(stationary_data, lags= 24)
#
# plt.show()

# Generate supervided data
# Referemce from https://towardsdatascience.com/5-machine-learning-techniques-for-sales-forecasting-598e4984b109

def generate_supervised(data):
    supervised_df = data.copy()

    # create column for each lag
    for i in range(1, 13):
        col = 'lag_' + str(i)
        supervised_df[col] = supervised_df['Sales_diff'].shift(i)

    # drop null values
    supervised_df = supervised_df.dropna().reset_index(drop=True)
    # supervised_df.to_csv('../data/model_df.csv', index=False)

    return supervised_df

# Observe the lags
# model_df = generate_supervised(stationary_data )
# print(model_df)


# model=ARIMA(stationary_data['Department Stores'],order=(1,1,1))
# history=model.fit()
