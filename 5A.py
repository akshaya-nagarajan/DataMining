
# coding: utf-8

# In[95]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="ticks")
player_df = pd.read_csv("~/Downloads/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
numcols = ['ID', 'Age', 'Height', 'Weight', 'Year']
catcols = [ 'Name', 'Sex', 'Team', 'NOC', 'Games', 'Season', 'City', 'Sport', 'Event', 'Medal']

player_df = player_df[numcols + catcols]

player_df.head(5)

#corr = player_df.corr()
#g = sns.heatmap(corr,  vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
#g.figure.set_size_inches(15,15)

filterinfDataframe_Male_1992 = player_df[(player_df['Medal'] == 'Gold') & (player_df['Sex'] == 'M') & (player_df['Year'] == 1992)]
#g = sns.pairplot(filterinfDataframe_Male_1992[['Age','Height', 'Weight']])

filterinfDataframe_1992 = player_df[(player_df['Medal'] == 'Gold') & (player_df['Year'] == 1992)]
#g = sns.pairplot(filterinfDataframe_1992[['Age','Height','Weight']], hue = 'Sex')

#g = sns.swarmplot(y = "Sex", x = 'Age', data = filterinfDataframe_1992)

g = sns.boxplot(y = "Sex", x = 'Age', data = filterinfDataframe_1992, whis=np.inf)
g = sns.swarmplot(y = "Sex", x = 'Age', data = filterinfDataframe_1992, size = 7,color = 'black')

max_age = filterinfDataframe_1992.Age.max()
max_age_player = filterinfDataframe_1992[(player_df['Age'] == max_age)]['Name'].values[0]
plt.annotate(s = max_age_player, xy = (max_age,0), xytext = (40,0.5),  arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03}, backgroundcolor = 'white')
g.figure.set_size_inches(14,10)

plt.show()


# In[61]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
player_df = pd.read_csv("~/Downloads/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
N = 2

filterinfDataframe_Male_1992 = player_df[(player_df['Medal'] == 'Gold') & (player_df['Sex'] == 'M') & (player_df['Year'] == 1992)]
filterinfDataframe_Female_1992 = player_df[(player_df['Medal'] == 'Gold') & (player_df['Sex'] == 'F') & (player_df['Year'] == 1992)]

filterinfDataframe_Male_1988 = player_df[(player_df['Medal'] == 'Gold') & (player_df['Sex'] == 'M') & (player_df['Year'] == 1988)]
filterinfDataframe_Female_1988 = player_df[(player_df['Medal'] == 'Gold') & (player_df['Sex'] == 'F') & (player_df['Year'] == 1988)]

menMeans = (filterinfDataframe_Male_1992.shape[0], filterinfDataframe_Male_1988.shape[0])
womenMeans = (filterinfDataframe_Female_1992.shape[0], filterinfDataframe_Male_1988.shape[0])

ind = np.arange(2)
width = 0.35 

p1 = plt.bar(ind, menMeans, width)
p2 = plt.bar(ind, womenMeans, width, bottom=menMeans)

plt.ylabel('Number of Gold')
plt.title('Medals by year and gender')
plt.xticks(ind, ('1992', '1988'))
plt.yticks(np.arange(0, 200, 425))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()


# In[2]:


import scipy.stats as ss
from collections import Counter
import math 
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

def convert(data, to):
    converted = None
    if to == 'array':
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == 'list':
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == 'dataframe':
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data),to))
    else:
        return converted
    
def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    :param x: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :param y: list / NumPy ndarray / Pandas Series
        A sequence of measurements
    :return: float
    """
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta

def associations(dataset, nominal_columns=None, mark_columns=False, theil_u=False, plot=True,
                          return_results = False, **kwargs):
    """
    Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
    continuous features using:
     - Pearson's R for continuous-continuous cases
     - Correlation Ratio for categorical-continuous cases
     - Cramer's V or Theil's U for categorical-categorical cases
    :param dataset: NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    :param nominal_columns: string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
        columns are categorical, or None (default) to state none are categorical
    :param mark_columns: Boolean (default: False)
        if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
        continuous), as provided by nominal_columns
    :param theil_u: Boolean (default: False)
        In the case of categorical-categorical feaures, use Theil's U instead of Cramer's V
    :param plot: Boolean (default: True)
        If True, plot a heat-map of the correlation matrix
    :param return_results: Boolean (default: False)
        If True, the function will return a Pandas DataFrame of the computed associations
    :param kwargs:
        Arguments to be passed to used function and methods
    :return: Pandas DataFrame
        A DataFrame of the correlation/strength-of-association between all features
    """

    dataset = convert(dataset, 'dataframe')
    columns = dataset.columns
    if nominal_columns is None:
        nominal_columns = list()
    elif nominal_columns == 'all':
        nominal_columns = columns
    corr = pd.DataFrame(index=columns, columns=columns)
    for i in range(0,len(columns)):
        for j in range(i,len(columns)):
            if i == j:
                corr[columns[i]][columns[j]] = 1.0
            else:
                if columns[i] in nominal_columns:
                    if columns[j] in nominal_columns:
                        if theil_u:
                            corr[columns[j]][columns[i]] = theils_u(dataset[columns[i]],dataset[columns[j]])
                            corr[columns[i]][columns[j]] = theils_u(dataset[columns[j]],dataset[columns[i]])
                        else:
                            cell = cramers_v(dataset[columns[i]],dataset[columns[j]])
                            corr[columns[i]][columns[j]] = cell
                            corr[columns[j]][columns[i]] = cell
                    else:
                        cell = correlation_ratio(dataset[columns[i]], dataset[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                else:
                    if columns[j] in nominal_columns:
                        cell = correlation_ratio(dataset[columns[j]], dataset[columns[i]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
                    else:
                        cell, _ = ss.pearsonr(dataset[columns[i]], dataset[columns[j]])
                        corr[columns[i]][columns[j]] = cell
                        corr[columns[j]][columns[i]] = cell
    corr.fillna(value=np.nan, inplace=True)
    if mark_columns:
        marked_columns = ['{} (nom)'.format(col) if col in nominal_columns else '{} (con)'.format(col) for col in columns]
        corr.columns = marked_columns
        corr.index = marked_columns
    if plot:
        plt.figure(figsize=(20,20))#kwargs.get('figsize',None))
        sns.heatmap(corr, annot=kwargs.get('annot',True), fmt=kwargs.get('fmt','.2f'), cmap='coolwarm')
        plt.show()
    if return_results:
        return corr

sns.set(style="ticks")
player_df = pd.read_csv("~/Downloads/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
numcols = ['ID', 'Age', 'Height', 'Weight', 'Year']
catcols = [ 'Name', 'Sex', 'Team', 'NOC', 'Games', 'Season', 'City', 'Sport', 'Event', 'Medal']

player_df = player_df[numcols + catcols]

player_df = player_df.fillna(0)
results = associations(player_df,nominal_columns=catcols,return_results=True)

