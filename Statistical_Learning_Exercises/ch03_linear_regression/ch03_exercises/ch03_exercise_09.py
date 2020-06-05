#!/usr/bin/python

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools as it
import statsmodels.formula.api as smf
import seaborn as sns
import numpy as np
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.simplefilter(action='ignore', category=FutureWarning)
plt.style.use('seaborn-white')
sns.set_style('white')


# # Simple Regression of `mpg` on `horsepower` in `auto` dataset


# ## Preparing the dataset


print(f"\nLoading and preparing the dataset.\n")

auto = pd.read_csv('../../datasets/Auto.csv', index_col=0)
auto.horsepower = pd.to_numeric(auto.horsepower, errors='coerce')

# Impute missing values represented by `'?'` in  `horsepower`

# replace `?` with nans
auto.loc[:, 'horsepower'].apply(lambda x: np.nan if x == '?' else x)

# cast horsepower to numeric dtype
auto.loc[:, 'horsepower'] = pd.to_numeric(auto.horsepower)

# now impute values
auto.loc[:, 'horsepower'] = auto.horsepower.fillna(auto.horsepower.mean())

print("\n", auto.head(), "\n")
print("\n", auto.info(), "\n")

input("Press Enter to continue...")


# ## (a) Scatterplot matrix of `auto`

print(f"\nPlotting distributions and pairwise scatterplots.\n")

sns.pairplot(auto.dropna())

print("Close plot window to continue...")

plt.show()


# ## (b) Correlation matrix of `auto`


print(f"\nPrinting correlation matrix.\n")

print(auto.corr())

input("Press Enter to continue...")


# ## (c) Fitting the model

print(f"\nFitting a linear regression model for mpg with all features\n")

# drop non-numerical columns and rows with null entries
model_df = auto.drop(['name'], axis=1).dropna()
X, Y = model_df.drop(['mpg'], axis=1), model_df.mpg

# add constant
X = sm.add_constant(X)

# create and fit model
model = sm.OLS(Y, X).fit()

# show results summary
model.summary()

input("Press Enter to continue...")


# #### i. Is there a relationship between the predictors and the `mpg`?


print(f"\nThe F-statistic is {model.f_pvalue} so we conclude at least "
      f"one of the predictors has a relationship with mpg\n")

input("Press Enter to continue...")

# #### ii. Which predictors appear to have a statistically significant
# relationship to the response?


print(f"\nThe p-values of the individual coefficients are:\n"
      f"{model.pvalues}\n")

input("Press Enter to continue...")

sig_lev = 0.05
is_stat_sig = model.pvalues < sig_lev

print(f"\nUsing a significance level of {sig_lev}\n")
print(f"\nThe p-values of the statistically significant coefficients are:\n"
      f"{model.pvalues[is_stat_sig]}\n")

input("Press Enter to continue...")

print(f"\nThe p-values of the statistically insignificant coefficients are:"
      f"\n{model.pvalues[~ is_stat_sig]}\n")

input("Press Enter to continue...")

# #### (iii) What does the coefficient for the year variable suggest?

print(f"\nThe coefficient for the year variable suggests that "
      f"fuel efficiency has been improving over time\n")

input("Press Enter to continue...")


# ## (d) Diagnostic plots


# ### Studentized Residuals vs. Fitted plot


print(f"\nPlotting studentized residuals vs. fitted values. to check for "
      f"non-linearity.\n")

# get full prediction results
pred_df = model.get_prediction().summary_frame()

# rename columns to avoid `mean` name conflicts and other confusions
new_names = {}
for name in pred_df.columns:
    if 'mean' in name:
        new_names[name] = name.replace('mean', 'mpg_pred')
    elif 'obs_ci' in name:
        new_names[name] = name.replace('obs_ci', 'mpg_pred_pi')
    else:
        new_names[name] = name
pred_df = pred_df.rename(new_names, axis='columns')

# concat mpg, horsepower and prediction results in dataframe
model_df = pd.concat([X, Y, pred_df], axis=1)

# add studentized residuals to the dataframe
model_df['resid'] = model.resid

# studentized residuals vs. predicted values plot
sns.regplot(model_df.mpg_pred, model_df.resid/model_df.resid.std(),
            lowess=True, line_kws={'color': 'r', 'lw': 1},
            scatter_kws={'facecolors': 'grey', 'edgecolors': 'grey',
                         'alpha': 0.4})
plt.ylabel('studentized resid')
plt.title("Residuals vs. fitted values")

print('Close plot window to continue...')

plt.show()


# ### QQ-plot of Residuals


print(f"\nPlotting QQ-plot of residuals to test normality of errors.\n")

# plot standardized residuals against a standard normal distribution
sm.qqplot(model_df.resid/model_df.resid.std(), color='grey',
          alpha=0.5, xlabel='')
plt.ylabel('studentized resid quantiles')
plt.xlabel('standard normal quantiles')
plt.title('QQ-plot of residuals')

print('Close plot window to continue...')

plt.show()


# ### Scale-location plot


print(f"\nPlotting scale-location to test heteroscedasticity.\n")

sqrt_stud_resid = np.sqrt(np.abs(model_df.resid/model_df.resid.std()))
sns.regplot(model_df.mpg_pred, sqrt_stud_resid, lowess=True,
            line_kws={'color': 'r', 'lw': 1},
            scatter_kws={'facecolors': 'grey', 'edgecolors': 'grey',
                         'alpha': 0.4})
plt.ylabel('√|studentized resid|')
plt.title('Scale-location plot')

print('Close plot window to continue...')

plt.show()


# ### Influence Plot


print(f"\nPlotting influence to check for high influence points.\n")

# scatterplot of leverage vs studentized residuals
axes = sns.scatterplot(model.get_influence().hat_matrix_diag,
                       model_df.resid/model_df.resid.std(),
                       facecolors='grey', edgecolors='grey', alpha=0.4)

# plot Cook's distance contours for D = 0.5, D = 1
x = np.linspace(0.0000001, axes.get_xlim()[1], 50)
plt.plot(x, np.sqrt(0.5*(1 - x)/x), color='red', linestyle='dashed')
plt.plot(x, np.sqrt((1 - x)/x), color='red', linestyle='dashed')
plt.xlabel('leverage')
plt.ylabel('studentized resid')
plt.ylim(bottom=0, top=20)
plt.title('Influence plot')

print('Close plot window to continue...')

plt.show()


print(f"\nFrom these plots we conclude: \n")
print(f"There is non-linearity in the data.")
print(f"There are a handful of outliers (studentized residual.")
print(f"The assumption of normal errors is appropriate.")
print(f"The data shows heteroscedasticity.")
print(f"There are no high influence points.\n")

input("Press Enter to continue...")

# ## (e) Interaction effects

print(f"\nFitting a model consisting of only interaction terms.\n")

# generate formula for interaction terms
names = list(auto.columns.drop('name').drop('mpg'))
pairs = list(it.product(names, names))
terms = [name1 + ' : ' + name2 for (name1, name2) in pairs if name1 != name2]
formula = 'mpg ~ '
for term in terms:
    formula += term + ' + '
formula = formula[:-3]
formula

# fit model
pair_int_model = smf.ols(formula=formula, data=auto).fit()

print(f"\nThe p-values of the statistically significant coefficients "
      f"for the pure interaction model are:\n"
      f"{pair_int_model.pvalues[pair_int_model.pvalues < 5e-2]}\n")

input("Press Enter to continue...")

# Now to use `+` we fit a full model

print(f"\nFitting a full model consisting of predictors "
      f"and interaction terms.\n")

# generate formula for interaction terms
names = list(auto.columns.drop('name').drop('mpg'))
formula = 'mpg ~ '
for name in names:
    formula += name + '*'
formula = formula[:-1]
formula

# fit model
full_int_model = smf.ols(formula=formula, data=auto).fit()

print(f"\nThe p-values of the statistically significant coefficients "
      f"for the full model are:\n"
      f"{full_int_model.pvalues[full_int_model.pvalues < 0.05]}\n")

input("Press Enter to continue...")


# ## (f) Variable transformations


# drop constant before transformation, else const for log(X) will be zero
X = X.drop('const', axis=1)

# transform data
log_X = np.log(X)
sqrt_X = np.sqrt(X)
X_sq = X**2

# fit models with constants
sqrt_model = sm.OLS(Y, sm.add_constant(sqrt_X)).fit()
sq_model = sm.OLS(Y, sm.add_constant(X_sq)).fit()

input("Press Enter to continue...")


# ### The $\log(X)$ model

print(f"\nFitting a model with log transformation\n")

log_model = sm.OLS(Y, sm.add_constant(log_X)).fit()

print(f"\n{log_model.summary()}\n")

input("Press Enter to continue...")

print(f"\nComparing statistical significance of original and log models\n")

# compare statistically significant variables for orig and log models

stat_sig_df = pd.concat([model.pvalues[is_stat_sig],
                         log_model.pvalues[is_stat_sig]],
                        join='outer', axis=1, sort=False)
stat_sig_df = stat_sig_df.rename({0: 'model_pval', 1: 'log_model_pval'},
                                 axis='columns')

print(f"\nThe statistically significant features of the the original "
      f"and log models and their p-values are \n{stat_sig_df}")

stat_insig_df = pd.concat([model.pvalues[~ is_stat_sig],
                           log_model.pvalues[~ is_stat_sig]],
                          join='outer', axis=1, sort=False)
stat_insig_df = stat_sig_df.rename({0: 'model_pval',
                                    1: 'log_model_pval'}, axis='columns')
stat_insig_df

print(f"\nThe insignificant features of the the original "
      f"and log models and their p-values are {stat_sig_df}")

input("Press Enter to continue...")

print(f"\nComparing predictive performance original and log models\n")

# split the data
X, y = auto.dropna().drop(columns=['name', 'mpg']), auto.dropna()['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y)

# transform
log_X_train, log_X_test = np.log(X_train), np.log(X_test)

# train models
reg_model = LinearRegression().fit(X_train, y_train)
log_model = LinearRegression().fit(log_X_train, y_train)

# get train mean squared errors
reg_train_mse = mean_squared_error(y_train, reg_model.predict(X_train))
log_train_mse = mean_squared_error(y_train, log_model.predict(log_X_train))

print(f"\nThe reg model train mse is {reg_train_mse} and "
      f" and the log model train mse is {log_train_mse}\n")

# get test mean squared errors
reg_test_mse = mean_squared_error(y_test, reg_model.predict(X_test))
log_test_mse = mean_squared_error(y_test, log_model.predict(log_X_test))

print(f"\nThe reg model test mse is {reg_test_mse} and "
      f" and the log model test mse is {log_test_mse}\n")

input("Press Enter to continue...")


# ### The $\sqrt{X}$ model


print(f"\nFitting a model with sqrt transformation\n")

sqrt_model = sm.OLS(Y, sm.add_constant(sqrt_X)).fit()

print(f"\n{sqrt_model.summary()}\n")

input("Press Enter to continue...")

print(f"\nComparing statistical significance of original and sqrt models\n")

# compare statistically significant variables for orig and sqrt models

stat_sig_df = pd.concat([model.pvalues[is_stat_sig],
                         sqrt_model.pvalues[is_stat_sig]],
                        join='outer', axis=1, sort=False)
stat_sig_df = stat_sig_df.rename({0: 'model_pval', 1: 'sqrt_model_pval'},
                                 axis='columns')

print(f"\nThe statistically significant features of the the original "
      f"and sqrt models and their p-values are \n {stat_sig_df}")

stat_insig_df = pd.concat([model.pvalues[~ is_stat_sig],
                           sqrt_model.pvalues[~ is_stat_sig]],
                          join='outer', axis=1, sort=False)
stat_insig_df = stat_sig_df.rename({0: 'model_pval',
                                    1: 'sqrt_model_pval'}, axis='columns')
stat_insig_df

print(f"\nThe insignificant features of the the original "
      f"and sqrt models and their p-values are \n {stat_sig_df}")

input("Press Enter to continue...")

print(f"\nComparing predictive performance original and sqrt models\n")

# transform
sqrt_X_train, sqrt_X_test = np.sqrt(X_train), np.sqrt(X_test)

# train models
reg_model = LinearRegression().fit(X_train, y_train)
sqrt_model = LinearRegression().fit(sqrt_X_train, y_train)

# get train mean squared errors
reg_train_mse = mean_squared_error(y_train, reg_model.predict(X_train))
sqrt_train_mse = mean_squared_error(y_train, sqrt_model.predict(sqrt_X_train))

print(f"\nThe reg model train mse is {reg_train_mse} and "
      f" and the sqrt model train mse is {sqrt_train_mse}\n")

# get test mean squared errors
reg_test_mse = mean_squared_error(y_test, reg_model.predict(X_test))
sqrt_test_mse = mean_squared_error(y_test, sqrt_model.predict(sqrt_X_test))

print(f"\nThe reg model test mse is {reg_test_mse} and "
      f" and the sqrt model test mse is {sqrt_test_mse}\n")

input("Press Enter to continue...")


# ### The $X^2$ model

print(f"\nFitting a model with square transformation\n")

sq_model = sm.OLS(Y, sm.add_constant(X_sq)).fit()

print(f"\n{sq_model.summary()}\n")

input("Press Enter to continue...")

print(f"\nComparing statistical significance of original and sq models\n")

# compare statistically significant variables for orig and sq models

stat_sig_df = pd.concat([model.pvalues[is_stat_sig],
                         sq_model.pvalues[is_stat_sig]],
                        join='outer', axis=1, sort=False)
stat_sig_df = stat_sig_df.rename({0: 'model_pval', 1: 'sq_model_pval'},
                                 axis='columns')

print(f"\nThe statistically significant features of the the original "
      f"and sq models and their p-values are {stat_sig_df}")

stat_insig_df = pd.concat([model.pvalues[~ is_stat_sig],
                           sq_model.pvalues[~ is_stat_sig]],
                          join='outer', axis=1, sort=False)
stat_insig_df = stat_sig_df.rename({0: 'model_pval',
                                    1: 'sq_model_pval'}, axis='columns')
stat_insig_df

print(f"\nThe insignificant features of the the original "
      f"and sq models and their p-values are {stat_sig_df}")

input("Press Enter to continue...")

print(f"\nComparing predictive performance original and sq models\n")

# transform
X_sq_train, X_sq_test = X_train**2, X_test**2

# train models
reg_model = LinearRegression().fit(X_train, y_train)
sq_model = LinearRegression().fit(X_sq_train, y_train)

# get train mean squared errors
reg_train_mse = mean_squared_error(y_train, reg_model.predict(X_train))
sq_train_mse = mean_squared_error(y_train, sq_model.predict(X_sq_train))

print(f"\nThe reg model train mse is {reg_train_mse} and "
      f" and the sq model train mse is {sq_train_mse}\n")

# get test mean squared errors
reg_test_mse = mean_squared_error(y_test, reg_model.predict(X_test))
sq_test_mse = mean_squared_error(y_test, sq_model.predict(X_sq_test))

print(f"\nThe reg model test mse is {reg_test_mse} and "
      f" and the sq model test mse is {sq_test_mse}\n")
