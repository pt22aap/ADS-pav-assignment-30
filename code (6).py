import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def remove_missing_features(df):
    
    # create a copy of the dataframe to avoid changing the original
    df_cp=df.copy()
    
    # find features with non-zero missing values
    n_missing_vals=df.isnull().sum()

    # get the index list of the features with non-zero missing values
    n_missing_index_list = list(n_missing_vals.index)
    
    # calculate the percentage of missing values
    # shape[0] gives the number of rows in the dataframe, hence, by diving the no. number of missing values by the total
    # no. of rows we get the ratio of missing values - multipled by 100 to get percentage
    # here missing_percentage consists of key value pairs - column name: percentage of missing values
    missing_percentage = n_missing_vals[n_missing_vals!=0]/df.shape[0]*100

    # list to maintain the columns to drop
    cols_to_trim=[]
    
    # iterate over each key value pair
    for i,val in enumerate(missing_percentage):
        # if percentage value is > 75
        if val > 75:
            # add the corresponding column to the list of cols_to_trim
            cols_to_trim.append(n_missing_index_list[i])

    if len(cols_to_trim) > 0:
        # drop the columns identified using the dataframe drop() method
        df_cp=df_cp.drop(columns=cols_to_trim)
        print("Dropped Columns:" + str(cols_to_trim))
    else:
        print("No columns dropped")

    # return the updated dataframe
    return df_cp

def process_world_bank_data(filename, countries=[], indicators=[]):
    # read data into a pandas dataframe
    df = pd.read_csv(filename, skiprows=4)

    # select any specific countries if given
    if countries:
        df = df[df["Country Name"].isin(countries)]

    # select any specific indicators if given
    if indicators:
        df = df[df["Indicator Name"].isin(indicators)]

    # drop unnecessary columns
    drop_cols = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(columns=drop_cols)
    
    # select data only from years 1990 to 2019, by changing the columns to numeric first
    df.columns = ['Country Name', 'Indicator Name'] + list(df.columns[2:].map(int))
    df = df[['Country Name', 'Indicator Name']+list(range(1990, 2019))]

    # lets merge reshape the dataframe and make Indicators as columns, for easier data cleansing
    df = df.set_index(['Country Name', 'Indicator Name']).T.unstack().unstack(level=1).reset_index().rename(columns={'level_1':'Year'})
    
    # we will use remove_missing_features to remove those indicators which have more than 75% missing values
    df = remove_missing_features(df)
    
    # convert years to columns
    df_years = df.set_index(['Year', 'Country Name']).unstack(level=0).swaplevel(axis=1).sort_index(axis=1, level=0)

    # convert countries to columns
    df_countries = df.set_index(['Year', 'Country Name']).unstack(level=1).swaplevel(axis=1).sort_index(axis=1, level=0)
    
    return df_years, df_countries

countries = ['United Arab Emirates', 'United States', 'Japan', 'Singapore', 'South Africa', 'Malaysia']
indicators = [
    'Access to electricity (% of population)',
    'Forest area (sq. km)',
    'PFC gas emissions (thousand metric tons of CO2 equivalent)',
    'Cereal yield (kg per hectare)',
    'Renewable electricity output (% of total electricity output)',
    'CO2 emissions from liquid fuel consumption (kt)',
    'Population growth (annual %)',
    'CO2 emissions from solid fuel consumption (kt)',
    'Urban land area where elevation is below 5 meters (sq. km)',
    'Electricity production from natural gas sources (% of total)'
    ]

df_years, df_countries = process_world_bank_data("ClimateChangeData.csv", countries=countries, indicators=indicators)

# customize the xticks & yticks for a cleaner heatmap plots
xticks=yticks=['Electricity Access', 'CO2 Emissions - Liquid Fuel', 'CO2 Emissions - Solid Fuel', 
'Natural Gas Electricity Production', 'PFC gas emissions', 'Population growth', 'Renewable electricity output', 'Elevated Urban Land Area']

# plot heatmap for Japan and save the figure
ax = sns.heatmap(df_countries['Japan'].corr(), annot=True)
ax.set_title("Correlation Matrix for Japan"), ax.set_xlabel(''), ax.set_ylabel(''), ax.set_xticklabels(xticks, fontsize=8), ax.set_yticklabels(yticks, fontsize=8)
plt.savefig('corrJapan.png')

# plot a bar graph for Liquid Fuel - CO2 Emissions and set xlabel, ylabel & title
plt.figure(figsize=(6, 4))
ax = sns.barplot(x = 'Country Name', y = 'CO2 emissions from liquid fuel consumption (kt)', hue = 'Year',
                data = df_years[list(range(1990, 2016, 5))].unstack().unstack(level=1).reset_index()
                )
ax.set_ylabel(''), ax.set_xlabel(''), ax.set_title("CO2 emissions from liquid fuel consumption (kt)", fontsize=6)
ax.set_xticklabels(['Japan', 'Malaysia', 'Singapore', 'SA', 'UAE', 'USA'], fontsize=8)
plt.savefig('co2barplot.png')

# plot a bar graph for Population Growth and set xlabel, ylabel & title
plt.figure(figsize=(6,4))
ax = sns.barplot(x = 'Country Name', y = 'Population growth (annual %)', hue = 'Year',
                data = df_years[list(range(1990, 2016, 5))].unstack().unstack(level=1).reset_index()
                )
ax.set_ylabel(''), ax.set_xlabel(''), ax.set_title("Population growth", fontsize=6)
ax.set_xticklabels(['Japan', 'Malaysia', 'Singapore', 'SA', 'UAE', 'USA'], fontsize=8)
plt.legend(loc='upper left', fontsize=6)
plt.savefig('popGrowth.png')

# convert the MultiIndex Column Level Countries as a column
df=df_countries.unstack().unstack(level=1).reset_index()

# Combine the CO2 Emissions from solid & liquid fuel consumptions
df['CO2 Emissions'] = df['CO2 emissions from liquid fuel consumption (kt)'] + df['CO2 emissions from solid fuel consumption (kt)']
# create a new column for access to electricity categories
df['Access_cat'] = pd.cut(df['Access to electricity (% of population)'], bins=[0, 50, 70, 90, 100], labels=['Very Low', 'Low', 'Medium', 'High'])

# create a horizontal bar plot for the combined CO2 Emissions vs Access to Electricty
plt.figure(figsize=(4,3))
ax = sns.barplot(x='CO2 Emissions', y='Access_cat', data=df, hue='Country Name')
ax.set(ylabel='Access to Electricity')
plt.legend(loc='upper right', fontsize=6)
plt.savefig('co2vsEAccess.png')

# plot a stacked bar plot for CO2 Emissions
fig = pd.crosstab(pd.cut(df['CO2 Emissions'], 10), df['Country Name']).plot.bar(stacked=True).get_figure()
fig.savefig("stackedCO2Combined.png")

# create a multiple line plot for Renewable Electricity Output
plt.figure(figsize=(4,3))
ax = sns.lineplot(x='Year', y='Renewable electricity output (% of total electricity output)', hue='Country Name', data=df, dashes=True)
for i in ax.lines:
    i.set_linestyle("--")
ax.set_ylabel(ylabel='')
ax.set_title('Renewable electricity output (%)', fontsize=8)
plt.legend(loc='upper left', fontsize='6')
plt.savefig('reLinePlot.png')

# plot heatmap for UAE and save the figure
ax = sns.heatmap(df_countries['United Arab Emirates'].corr(), annot=True)
ax.set_title("Correlation Matrix for UAE")
ax.set_xlabel(''), ax.set_ylabel(''), ax.set_xticklabels(xticks, fontsize=8), ax.set_yticklabels(yticks, fontsize=8)
plt.savefig('corrUAE.png')

# Create a table for the Electricity Production from Natural Gas and save it in csv
col="Electricity production from natural gas sources (% of total)"
newDf = df_years[[(1995, col), (2005, col), (2015, col)]].loc[['United States', 'Japan', 'Singapore', 'Malaysia', 'United Arab Emirates']]
newDf.columns = newDf.columns.droplevel(1)
newDf.to_csv("ngDf.csv")