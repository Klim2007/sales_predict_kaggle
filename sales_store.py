
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import *


#Adding data
df_oil_price = pd.read_csv('C:\\Users\\SberUser\\PycharmProjects\\data_science\\data_sc-env\\Include\\oil.csv') # optional  parse_dates=['date'],infer_datetime_format=True
df_sales_test = pd.read_csv('C:\\Users\\SberUser\\PycharmProjects\\data_science\\data_sc-env\\Include\\test.csv') #onpromotion- number of goods
df_sales_train = pd.read_csv('C:\\Users\\SberUser\\PycharmProjects\\data_science\\data_sc-env\\Include\\train.csv.zip')#the same as test set but with sales
stores = pd.read_csv('C:\\Users\\SberUser\\PycharmProjects\\data_science\\data_sc-env\\Include\\stores.csv')
transactions = pd.read_csv('C:\\Users\\SberUser\\PycharmProjects\\data_science\\data_sc-env\\Include\\transactions.csv.zip')
df_holidays = pd.read_csv('C:\\Users\\SberUser\\PycharmProjects\\data_science\\data_sc-env\\Include\\holidays_events.csv')
sample_submission = pd.read_csv('C:\\Users\\SberUser\\PycharmProjects\\data_science\\data_sc-env\\Include\\sample_submission.csv').drop(columns = ['sales'])


# Oil preprocessing and merge to train_data / test_data
oil = df_oil_price.fillna(method = 'pad')
oil = oil.fillna(method = 'bfill')
oil.set_index('date', inplace = True)

# MERGING ALL SETS / filling NaN values
train_data_oil = pd.merge(df_sales_train, oil, on = "date", how = 'left') # adding oil prices
train_data_oil = train_data_oil.fillna(method = 'pad') # filling NaN with (bfill) backfill 

test_data_oil = pd.merge(df_sales_test, oil, on = "date", how = 'left') 
test_data_oil = test_data_oil.fillna(method = 'pad')

train_data_oil_holiday = pd.merge(train_data_oil, df_holidays, on = "date", how = 'left')
train_data_oil_holiday = train_data_oil_holiday.fillna('Empty')

test_data_oil_holiday = pd.merge(test_data_oil, df_holidays, on = "date", how = 'left')
test_data_oil_holiday = test_data_oil_holiday.fillna('Empty')

#ADDING transaction
train_data_oil_holiday_transactions = pd.merge(train_data_oil_holiday, transactions, on = ['date', 'store_nbr'], how = 'left')
train_data_oil_holiday_transactions['transactions'] = train_data_oil_holiday_transactions['transactions'].fillna(0)

test_data_oil_holiday_transactions = pd.merge(test_data_oil_holiday, transactions, on = ['date', 'store_nbr'], how = 'left')
test_data_oil_holiday_transactions['transactions'] = test_data_oil_holiday_transactions['transactions'].fillna(0)

#Adding stores
train_data_oil_holiday_transactions = pd.merge(train_data_oil_holiday_transactions, stores, on = 'store_nbr', how = 'left')
test_data_oil_holiday_transactions = pd.merge(test_data_oil_holiday_transactions, stores, on =  'store_nbr', how = 'left')


train_data_oil_holiday_transactions['sales'].astype('float32')
train_data_oil_holiday_transactions['store_nbr'].astype('category')
train_data_oil_holiday_transactions['family'].astype('category')

# Year, Month, Day, weekend, weekdays columns merge to train_data / test_data

def split_year(time):
    return (time.split('-')[0])
def split_month(time):
    return (time.split('-')[1])
def split_day(time):
    return (time.split('-')[2])
def weekend(date):
    weekend = []
    a = pd.to_datetime(date)
    for i in range(len(a)):
        if a.iloc[i].weekday() >= 5 :
            weekend.append(1)
        else:
            weekend.append(0)
    return weekend
def weekday(date):
    weekday = []
    a = pd.to_datetime(date)
    for i in range(len(a)):
        weekday.append(a.iloc[i].weekday())
    return weekday

train_data_oil_holiday_transactions['Year'] = train_data_oil_holiday_transactions['date'].apply(split_year)
train_data_oil_holiday_transactions['Month'] = train_data_oil_holiday_transactions['date'].apply(split_month)
train_data_oil_holiday_transactions['Day'] = train_data_oil_holiday_transactions['date'].apply(split_day)
train_data_oil_holiday_transactions['Weekend'] = weekend(train_data_oil_holiday_transactions['date'])
train_data_oil_holiday_transactions['Weekday'] = weekday(train_data_oil_holiday_transactions['date'])

test_data_oil_holiday_transactions['Year'] = test_data_oil_holiday_transactions['date'].apply(split_year)
test_data_oil_holiday_transactions['Month'] = test_data_oil_holiday_transactions['date'].apply(split_month)
test_data_oil_holiday_transactions['Day'] = test_data_oil_holiday_transactions['date'].apply(split_day)
test_data_oil_holiday_transactions['Weekend'] = weekend(test_data_oil_holiday_transactions['date'])
test_data_oil_holiday_transactions['Weekday'] = weekday(test_data_oil_holiday_transactions['date'])

#Scatter plot to the see correlation between average unit sold and oil price each day
sales_oil = df_sales_train[['date','sales']]
sales_oil = sales_oil.groupby('date').mean()['sales']
sales_oil = sales_oil.reset_index()
sales_oil = pd.merge(sales_oil, oil, on ='date', how='left')
sales_oil['dcoilwtico'] = sales_oil['dcoilwtico'].ffill()

# we don't have all the oil prices available, we impute them   ffill()
previous_price = 93.14

for index, row in sales_oil.iterrows():
    if math.isnan(row['dcoilwtico']):
        sales_oil.loc[sales_oil['date'] == row['date'], 'dcoilwtico'] = previous_price
    else: 
        previous_price = row['dcoilwtico']

# Corelation between average sales and oil price
fig = px.scatter(sales_oil, x="dcoilwtico", y="sales",size='sales', color='sales',color_continuous_scale="Viridis", trendline="ols", trendline_color_override="black")

fig.update_layout({"title": f'Correlation between Oil Prices and Sales',
                   "xaxis": {"title":"Oil Price"},
                   "yaxis": {"title":"Avegare of all product sales on each day"},
                   "showlegend": False})

#Volume of goods sales by families
sales_review = train_data_oil_holiday_transactions[['family','sales']]
sales_review_family = sales_review.groupby('family').sales.sum().sort_values(ascending = False).reset_index()
fig_2 = px.bar(sales_review_family, y = "family", x="sales", color = "family", title = "Top selling product families - Daily cummulative", height=830)
fig_2.show()

#Mean sales
sales_review_family = sales_review.groupby('family').sales.mean().sort_values(ascending = False).reset_index()
fig_2 = px.bar(sales_review_family, y = "sales", x="family", color = "family", title = "Top selling product families - Daily average", height=1000)
fig_2.show()

# Store efficiency
shop_efficiency = train_data_oil_holiday_transactions[['store_nbr','sales']]
shop_efficiency = shop_efficiency.groupby('store_nbr').sales.sum().sort_values(ascending = True).reset_index()
fig_3 = px.bar(shop_efficiency, y = "sales", x="store_nbr", color = "sales", title = "Top selling prgitoduct  Stores ",  width= 1800, height=1000)
fig_3.show()

train_data_oil_holiday_transactions.rename(columns = {'type_x' : 'holiday_type', 'type_y' : 'shop_type'}, inplace = True)
test_data_oil_holiday_transactions.rename(columns = {'type_x' : 'holiday_type', 'type_y' : 'shop_type'}, inplace = True)

#Considering earthquake in 2016 (main sales is grocery diary beverages meat = food
disaster = train_data_oil_holiday_transactions[(train_data_oil_holiday_transactions['date'] >= '2016-04-16') & (train_data_oil_holiday_transactions['date'] <= '2016-10-16')]
disaster.sort_values('sales', ascending = False)
#print (disaster)
fig_4_eathquake = px.bar(disaster, y = "family", x="sales", color = "sales", title = "Selling while disaster",  width= 1800, height=1000)
fig_4_eathquake.show()

#Excluding sales in earthqake period from data set
train_data_oil_holiday_transactions =train_data_oil_holiday_transactions.drop(disaster.index)

#Featuring
#Store_nbr features relate the sales means
store_nbr_sales_means = train_data_oil_holiday_transactions.groupby('store_nbr').agg({'sales' : 'mean'}).reset_index().sort_values(by='sales', ascending=False)

sns.set()
plt.figure(figsize = (20,5))
sns.lineplot(x = store_nbr_sales_means.store_nbr, y = store_nbr_sales_means.sales, color = 'r', label = 'Means', alpha = 0.3,linewidth = 5)
plt.legend()
plt.xticks(range(1,60))
plt.title('Store_nbr : Comparsion with Mean')

#Family features relate the sales means
family_sales_mean = train_data_oil_holiday_transactions.groupby('family').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)[:20]
sns.set()
plt.figure(figsize = (20, 5))
sns.barplot(x = family_sales_mean.family, y = family_sales_mean.sales, color = 'r', label = 'Means', alpha = 0.3)
plt.legend()
plt.xticks(rotation = 60)
plt.title('Family : Comparsion with Mean')

#Onpromotion features relate the sales means
onpromotion_sales_means = train_data_oil_holiday_transactions.groupby('onpromotion').agg({'sales' : 'mean'}).reset_index().sort_values(by='sales', ascending=False)
sns.set()
plt.figure(figsize = (20,5))
sns.lineplot(x = onpromotion_sales_means.onpromotion, y = onpromotion_sales_means.sales, color = 'r', label = 'Means', alpha = 0.3)
plt.legend()
plt.xticks(range(1,1000))
plt.title('Hoilday_type : Comparsion with Mean')

#Dcoilwtico features relate the sales means¶ 3000999 (3000999 is not in train set its in test set but test set has no sales metric)
dcoilwtico_sales_means = train_data_oil_holiday_transactions.groupby('dcoilwtico').agg({'sales' : 'mean'}).reset_index().sort_values(by='sales', ascending=False)
sns.set()
plt.figure(figsize = (20,5))
sns.lineplot(x = dcoilwtico_sales_means.dcoilwtico, y = dcoilwtico_sales_means.sales, color = 'r', label = 'Means', alpha = 0.3)
plt.legend()
plt.title('Dcoilwtico : Comparsion with Mean')
plt.show()
plt.clf()


# Holiday_type features relate the sales means¶
holiday_type_sales_means = train_data_oil_holiday_transactions.groupby('holiday_type').agg({'sales' : 'mean'}).reset_index().sort_values(by='sales', ascending=False)
sns.set()
plt.figure(figsize = (20,5))
sns.barplot(x = holiday_type_sales_means.holiday_type, y = holiday_type_sales_means.sales, color = 'r', label = 'Means', alpha = 0.3)
plt.title('Hoilday_type : Comparsion with Mean')

#Locale features relate the sales means
sns.set()
plt.figure(figsize = (20, 5))
locale_sales_mean = train_data_oil_holiday_transactions.groupby('locale').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)[:20]
sns.barplot(x = locale_sales_mean.locale, y = locale_sales_mean.sales, color = 'r', label = 'Means', alpha = 0.3)
plt.legend()
plt.xticks(rotation = 90)
plt.title('locale Comparsion with Mean')

#Description features relate the sales means
sns.set()
plt.figure(figsize = (20, 5))
description_sales_mean = train_data_oil_holiday_transactions.groupby('description').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)[:20]
sns.barplot(x = description_sales_mean.description, y = description_sales_mean.sales, color = 'r', label = 'Means', alpha = 0.3)
plt.legend()
plt.xticks(rotation = 90)
plt.title('description Comparsion with Mean')

#State features relate the sales means
sns.set()
plt.figure(figsize = (20, 5))
state_mean = train_data_oil_holiday_transactions.groupby('state').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)[:20]
sns.barplot(x = state_mean.state, y = state_mean.sales, color = 'r', label = 'Means', alpha = 0.3)
plt.legend()
plt.xticks(rotation = 90)
plt.title('state Comparsion with Mean')

#Shop_type features relate the sales means
sns.set()
plt.figure(figsize = (20, 5))
shop_type_mean = train_data_oil_holiday_transactions.groupby('shop_type').agg({"sales" : "mean"}).reset_index().sort_values(by='sales', ascending=False)[:20]
sns.barplot(x = shop_type_mean.shop_type, y = shop_type_mean.sales, color = 'r', label = 'Means', alpha = 0.3)
plt.legend()
plt.xticks(rotation = 90)
plt.title('state Comparsion with Mean')

#Cluster relate sales mean
cluster_sales_means = train_data_oil_holiday_transactions.groupby('cluster').agg({'sales' : 'mean'}).reset_index().sort_values(by='sales', ascending=False)
sns.set()
plt.figure(figsize = (20,5))
sns.lineplot(x = cluster_sales_means.cluster, y = cluster_sales_means.sales, color = 'r', label = 'Means', alpha = 0.3)
plt.legend()
plt.title('cluster : Comparsion with Mean')
plt.show()
plt.clf()

#Year features relate the sales means
Year_sales_means = train_data_oil_holiday_transactions.groupby('Year').agg({'sales' : 'mean'}).reset_index().sort_values(by='sales', ascending=False)
sns.set()
plt.figure(figsize = (20,5))
sns.barplot(x = Year_sales_means.Year, y = Year_sales_means.sales, color = 'r', label = 'Means', alpha = 0.3)
plt.legend()
plt.title('Year : Comparsion with Mean')
plt.show()
plt.clf()

#Month features relate the sales means
Month_sales_means = train_data_oil_holiday_transactions.groupby('Month').agg({'sales' : 'mean'}).reset_index().sort_values(by='sales', ascending=False)
sns.set()
plt.figure(figsize = (20,5))
sns.barplot(x = Month_sales_means.Month, y = Month_sales_means.sales, color = 'r', label = 'Means', alpha = 0.3)
plt.legend()
plt.title('Month : Comparsion with Mean')
plt.show()
plt.clf()
plt.close()

#Data Feature's drop¶
train_data_oil_holiday_transactions = train_data_oil_holiday_transactions.drop(columns = ['id', 'date', 'transferred', 'Day', 'Weekday','Year', 'Month'])
test_data_oil_holiday_transactions = test_data_oil_holiday_transactions.drop(columns = ['id','date', 'transferred', 'Day', 'Weekday','Year', 'Month'])

#Feature's encoder by LabelEncoder
encoder_family = LabelEncoder()
train_data_oil_holiday_transactions['family'] = encoder_family.fit_transform(train_data_oil_holiday_transactions['family'])
test_data_oil_holiday_transactions['family'] = encoder_family.transform(test_data_oil_holiday_transactions['family'])

encoder_type = LabelEncoder()
train_data_oil_holiday_transactions['holiday_type'] = encoder_type.fit_transform(train_data_oil_holiday_transactions['holiday_type'])
test_data_oil_holiday_transactions['holiday_type'] = encoder_type.transform(test_data_oil_holiday_transactions['holiday_type'])

encoder_locale = LabelEncoder()
train_data_oil_holiday_transactions['locale'] = encoder_locale.fit_transform(train_data_oil_holiday_transactions['locale'])
test_data_oil_holiday_transactions['locale'] = encoder_locale.transform(test_data_oil_holiday_transactions['locale'])

encoder_description = LabelEncoder()
train_data_oil_holiday_transactions['description'] = encoder_description.fit_transform(train_data_oil_holiday_transactions['description'])
test_data_oil_holiday_transactions['description'] = encoder_description.transform(test_data_oil_holiday_transactions['description'])

encoder_locale_name = LabelEncoder()
train_data_oil_holiday_transactions['locale_name'] = encoder_locale_name.fit_transform(train_data_oil_holiday_transactions['locale_name'])
test_data_oil_holiday_transactions['locale_name'] = encoder_locale_name.transform(test_data_oil_holiday_transactions['locale_name'])

encoder_city = LabelEncoder()
train_data_oil_holiday_transactions['city'] = encoder_city.fit_transform(train_data_oil_holiday_transactions['city'])
test_data_oil_holiday_transactions['city'] = encoder_city.transform(test_data_oil_holiday_transactions['city'])

encoder_state = LabelEncoder()
train_data_oil_holiday_transactions['state'] = encoder_state.fit_transform(train_data_oil_holiday_transactions['state'])
test_data_oil_holiday_transactions['state'] = encoder_state.transform(test_data_oil_holiday_transactions['state'])

encoder_shop_type = LabelEncoder()
train_data_oil_holiday_transactions['shop_type'] = encoder_shop_type.fit_transform(train_data_oil_holiday_transactions['shop_type'])
test_data_oil_holiday_transactions['shop_type'] = encoder_shop_type.transform(test_data_oil_holiday_transactions['shop_type'])

#Model Building:- XGBoostRegressor
data = train_data_oil_holiday_transactions.drop(columns = 'sales') # try train set from train_data_oil_holiday_transactions and
target = train_data_oil_holiday_transactions['sales']

# spliting train t
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.8, random_state = 5) #try stratify = target!!!
XG = xgb.XGBRegressor(objective = 'reg:squarederror' , learning_rate = 0.1,max_depth = 10, n_estimators = 100).fit(x_train, y_train)
XG.score(x_train,y_train) #0.9566953158867476
y_pred_XG = XG.predict(x_test)
print (y_pred_XG)
print('Training Accuracy :',XG.score(x_train,y_train))
print('Testing Accuracy :',XG.score(x_test,y_test))

def relu(x):
    relu = []
    for i in x:
        if i < 0:
            relu.append(0)
        else:
            relu.append(i)
    return relu

plt.scatter(y_test, relu(y_pred_XG))
plt.plot([10000*x for x in range(10)], [10000*x for x in range(10)], color = 'r')
plt.xlabel("Reality")
plt.ylabel("Predicted")
plt.title('XG BoostRegressor')
plt.clf()

#Results to file
sub = XG.predict(test_data_oil_holiday_transactions)
sample_submission['sales'] = relu(sub)
sample_submission.to_csv('sample_submission.csv', index=False)

