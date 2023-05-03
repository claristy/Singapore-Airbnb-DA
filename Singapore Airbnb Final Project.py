#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import os


# In[5]:


# Input data
listings = pd.read_csv(r'DQLab_listings(22Sep2022).csv')
neighbourhood = pd.read_csv(r'DQLab_nieghbourhood(22Sep2022).csv')
reviews = pd.read_csv(r'DQLab_reviews(22Sep2022).csv')


# # Data Cleaning

# - merge all data
# - cek null
# - cek duplikat
# - cek data type

# In[6]:


listings.head()


# In[7]:


listings.price.describe()


# In[8]:


# Check for the outlier for listing price using boxplot
listings.boxplot(column='price')
plt.show()


# In[9]:


q1 = listings['price'].quantile(0.25)
q3 = listings['price'].quantile(0.75)
upper_limit = q3 + 1.5 * (q3-q1)
lower_limit = q3 - 1.5 * (q3-q1)


# In[10]:


# Filtering outliers
listings_cleaned = listings[listings['price'] > lower_limit]
listings_cleaned = listings[listings['price'] < upper_limit]

listings_cleaned['price'].describe()


# In[11]:


# Boxplot (after cleaned)
listings_cleaned.boxplot(column='price')
plt.show()


# In[12]:


listings_cleaned.drop('Unnamed: 0', inplace = True, axis=1)


# In[13]:


neighbourhood.head()


# In[14]:


neighbourhood.rename(columns = {'neighbourhood_group':'region'}, inplace = True)


# In[15]:


neighbourhood.drop('Unnamed: 0', inplace = True, axis=1)


# In[16]:


reviews.head()


# In[17]:


reviews.rename(columns = {'listing_id':'id'}, inplace = True)


# In[18]:


reviews.drop('Unnamed: 0', inplace = True, axis=1)


# In[19]:


# Add year & month column
reviews['yearmonth'] = reviews['date'].str.slice(0,7)
reviews.head()


# In[20]:


# Merge listings and reviews
df1 = listings_cleaned.merge(reviews, on='id')
df1.head()


# In[21]:


# Merge df1 and neighbourhood
df = df1.merge(neighbourhood, on='neighbourhood')
df


# In[22]:


df.info()


# In[23]:


# Change 'date' type
df['date'] = pd.to_datetime(df.date)


# In[24]:


# Drop duplicate data
df = df.drop_duplicates()
df


# # Business Questions
# 
# 1. How’s the rental trend?
# 2. In what month and year were the fewest and most numerous listing rental activities?
# 3. How's the price distribution?
# 4. Which type of room is most in demand?
# 5. How's the rental trend in region? What and why?
# 6. How is the rental activity from the top 100 listings? 

# # Rental Trend

# In[25]:


trend = df.groupby('yearmonth')['yearmonth'].count()
trend


# In[26]:


fig, ax = plt.subplots(figsize=(18, 4))
plt.plot(trend, color='#E96479')
plt.setp(ax.get_xticklabels(), fontsize=10, rotation=90, horizontalalignment="right")
plt.title('Singapore Airbnb Rental Trend \n01 Jan 2018 - 22 Sep 2022', fontsize=21)
plt.grid(linewidth=0.25)
plt.show()


# # The Fewest and The Most Active Rental Activities

# In[27]:


rental = df[['date']].copy()
rental


# In[28]:


rental = rental.groupby(['date'])['date'].size().reset_index(name='counts')
rental


# In[29]:


rental[rental.counts == rental.counts.max()]


# In[30]:


rental[rental.counts == rental.counts.min()]


# # Price Distribution

# In[31]:


plt.hist(listings['price'], bins=np.arange(0,1000,20), color='#7DB9B6', edgecolor='black')
plt.xlabel('Price')
plt.title('Singapore Airbnb Price Distribution')
plt.grid(linewidth=0.25)
plt.show()


# In[32]:


listings.price.describe()


# # Rental Activities in Neighbourhood Region

# In[33]:


reg_act = df[['region', 'date']].copy()
reg_act.sort_values('date')


# In[34]:


reg_act['region'].unique()


# In[35]:


fig, ax = plt.subplots(figsize=(15,7))
reg_act.groupby([reg_act.date.dt.year, reg_act.region]).count()['date'].unstack().plot(ax=ax)
plt.title('Region Rental Trend', fontsize=17)
plt.xlabel('Year (half-yearly)', fontsize=12)
plt.ylabel('Rent Activity', fontsize=12)
plt.grid(linewidth=0.20)
plt.show()


# In[36]:


reg_act1 = reg_act.groupby([reg_act.region, reg_act.date.dt.year])['date'].count().reset_index(name='counts')
reg_act1


# # Price Trend in Neighbourhood Region

# In[37]:


reg_price = listings_cleaned[['neighbourhood', 'price']].copy()
reg_price = reg_price.drop_duplicates()
reg_price


# In[38]:


reg_price = reg_price.groupby(['neighbourhood']).mean()['price'].reset_index(name='price')
reg_price = neighbourhood.merge(reg_price, on='neighbourhood')
reg_price.sort_values('region')


# In[39]:


fig, ax = plt.subplots(figsize=(15,7))
plt.bar(reg_price['neighbourhood'], reg_price['price'])
plt.setp(ax.get_xticklabels(), fontsize=12, rotation=40, horizontalalignment="right")

plt.title('Neighborhood Average Price', fontsize = 24)
plt.ylabel('Price (SGD)', fontsize = 12)

plt.axvspan(-0.5, 18.5, color='#009FBD', alpha=0.3)
plt.text(0, 380, "Central Region", va = 'top', rotation = 90, fontsize = 20)
plt.axvspan(18.5, 21.5, color='#F9E2AF', alpha=0.3)
plt.text(19, 380, "East Region", va = 'top', rotation = 90, fontsize = 20)
plt.axvspan(21.5, 27.5, color='#EB455F', alpha=0.3)
plt.text(22, 380, "North-East Region", va = 'top', rotation = 90, fontsize = 20)
plt.axvspan(27.5, 34.5, color='#3CCF4E', alpha=0.3)
plt.text(28, 380, "North Region", va = 'top', rotation = 90, fontsize = 20)
plt.axvspan(34.5, 43.5, color='#77037B', alpha=0.3)
plt.text(35, 380, "West Region", va = 'top', rotation = 90, fontsize = 20)
ax.grid(axis='y', linewidth=0.5)

plt.show()


# In[40]:


reg_price1 = reg_price.groupby(['region']).mean()['price'].reset_index(name='price')
reg_price1 = reg_price1.sort_values('price', ascending=False)
reg_price1


# In[41]:


fig, ax = plt.subplots(figsize=(9,7))
plt.barh(reg_price1['region'], reg_price1['price'], color='#009FBD', edgecolor='black')
plt.title('Region Average Price', fontsize=21)
plt.show()


# # Room Type 

# In[42]:


room_type = df[['room_type']].copy()
room_type = room_type.groupby(['room_type']).size().reset_index(name='counts')
room_type.sort_values('counts')


# In[43]:


plt.pie(room_type['counts'], labels=room_type['room_type'], autopct='%1.1f%%',
       colors=['#7DB9B6', '#E96479', '#89C4E1', '#F5E9CF'])
plt.legend(room_type['room_type'], loc='center left', bbox_to_anchor=(1,0.7))
plt.show()


# In[44]:


room_reg = listings_cleaned[['neighbourhood', 'room_type']].copy()
room_reg = room_reg.merge(neighbourhood, on='neighbourhood')
room_reg = room_reg.groupby(['region', 'room_type']).size().reset_index(name='counts')
room_reg


# In[45]:


pivot = pd.pivot_table(data=room_reg, index=['region'], columns=['room_type'], values='counts')
pivot = pivot.sort_values('Entire home/apt', ascending=False)
pivot


# In[46]:


ax = pivot.plot.bar(stacked=True, color =['#4D455D', '#E96479', '#7DB9B6', '#F5E9CF'], figsize=(8,6))
ax.set_title('Room Type per Region', fontsize=20)
ax.set_ylim(0,2500)
ax.set_xticklabels(['Central Region', 'North Region', 'West Region', 'East Region', 'North-East Region'], 
                   rotation=20)
plt.xlabel('')
plt.show()


# # Top 100 Listings

# In[47]:


top_list = df[['id', 'region', 'room_type']].copy()
top_list = top_list.groupby(['id', 'region', 'room_type'])['id'].count().reset_index(name='count')
top_list = top_list.sort_values('count', ascending=False)
top_list = top_list.head(100)
top_list


# In[48]:


pivot2 = pd.pivot_table(data=top_list, index=['region'], columns=['room_type'], values='count', aggfunc='sum')
pivot2 = pivot2.sort_values(['Entire home/apt', 'Private room'], ascending=False)
pivot2


# In[49]:


ax = pivot2.plot.bar(stacked=True, color =['#4D455D', '#E96479', '#7DB9B6', '#F5E9CF'], figsize=(7,11))
ax.set_title('Top 100 Listings', fontsize=22)
ax.set_ylim(0,10800)
ax.set_xticklabels(['North Region', 'Central Region', 'West Region', 'East Region', 'North-East Region'], 
                   rotation=20)
plt.xlabel('')
plt.show()


# In[50]:


top_list2 = top_list.merge(df[['id', 'price']], on='id')
top_list2 = top_list2.drop_duplicates()
top_list2


# In[51]:


top_list2['price'].describe()


# In[52]:


sns.set(rc={"figure.figsize":(3, 6)})
sns.set_style("white")
sns.boxplot(data = top_list2, y='price', showfliers = True, color='#F5E9CF')
plt.title('Top 100 Listing Price Distribution')
plt.xlabel('')
plt.ylabel('Price (SGD)')
plt.ylim(0,200)
plt.grid(linewidth=0.5, axis='y')
plt.show()


# In[ ]:




