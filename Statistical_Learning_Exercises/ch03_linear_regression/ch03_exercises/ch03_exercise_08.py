#!/usr/bin/python

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
plt.style.use('seaborn-white')
sns.set_style('white')


# # Simple Regression of `mpg` on `horsepower` in `auto` dataset


# ## Preparing the dataset


print(f"\nLoading the dataset.\n")

auto = pd.read_csv('../../datasets/Auto.csv')
auto.horsepower = pd.to_numeric(auto.horsepower, errors='coerce')
print("\n", auto.head(), "\n")
print("\n", auto.info(), "\n")

input("Press Enter to continue...")


# ##  (a) Fitting the model

print(f"\nFitting a linear regression model with horsepower as predictor "
      f"and mpg as response.")

# filter out null entries
X = auto.horsepower[auto.mpg.notna() & auto.horsepower.notna()]
Y = auto.mpg[auto.mpg.notna() & auto.horsepower.notna()]

# add constant
X = sm.add_constant(X)

# create and fit model
model = sm.OLS(Y, X)
model = model.fit()

# show results summary
print("\n", model.summary(), "\n")

input("Press Enter to continue...")


# ### i. Is there a relationship between `horsepower` and `mpg`?


print(f"\nThe p-value for the horsepower coefficient is {model.rsquared} "
      f"so we conclude there is a relationship between horsepower and mpg.\n")

input("Press Enter to continue...")


# ### ii. How strong is the relationship?


print(f"\nR^2 for the model is {model.rsquared} so the "
      f"relationship appears somewhat strong.\n")

input("Press Enter to continue...")


# ### iii. Is the relationship positive or negative?


print(f"\nModel parameter estimates are {model.params} so since "
      f"the horsepower coefficent is negative, the relationship "
      f"is negative.\n")


# ### iv. What is the predicted `mpg` associated with a `horsepower` of 98?
# What are the associated 95 % confidence and prediction intervals?


prediction = model.get_prediction([1, 98])
pred_df = prediction.summary_frame()

print(f"\nThe predicted value for `mpg`=98 is {pred_df['mean']}\n")

conf_int = (pred_df['mean_ci_lower'].values[0],
            pred_df['mean_ci_upper'].values[0])

print(f"\nThe associated 95% confidence interval is {conf_int}\n")

pred_int = (pred_df['obs_ci_lower'].values[0],
            pred_df['obs_ci_upper'].values[0])

print(f"\nWhile the 95% prediction interval is {pred_int}\n")


# ## (b) Scatterplot and least squares line plot


print(f"\nPlotting scatterplot and least squares line to visualize fit.\n")

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

# plot
sns.scatterplot(model_df.horsepower, model_df.mpg,
                facecolors='grey', alpha=0.5)
sns.lineplot(model_df.horsepower, model_df.mpg_pred, color='r')
plt.title("Scatterplot with least squares line.")

print('Close plot window to continue...')

plt.show()


# ## (c) Diagnostic plots


# ### Studentized Residuals vs. Fitted plot

print(f"\nPlotting studentized residuals vs. fitted values. to check for "
      f"non-linearity.\n")

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

print(f"\nResiduals vs. fitted values plot indicates non-linearity and "
      f"the presence of outliers.\n")

input("Press Enter to continue...")


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

print(f"\nQQ-plot supports the assumption that" 
      f"the errors/residuals are normally distributed.\n")

input("Press Enter to continue...")


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

print(f"\nScale-location plot indicates the assumption that variance"
      f"of the errors is equal (heteroscedasticity) isn't justified.\n")

input("Press Enter to continue...")


# ### Influence Plot

print(f"\nPlotting influence to check for high influence points.\n")

# scatterplot of leverage vs studentized residuals
axes = sns.regplot(model.get_influence().hat_matrix_diag,
                   model_df.resid/model_df.resid.std(),
                   lowess=True, line_kws={'color': 'r', 'lw': 1},
                   scatter_kws={'facecolors': 'grey',
                                'edgecolors': 'grey',
                                'alpha': 0.4})

# plot Cook's distance contours for D = 0.5, D = 1
x = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 50)
plt.plot(x, np.sqrt(0.5*(1 - x)/x), color='red', linestyle='dashed')
plt.plot(x, np.sqrt((1 - x)/x), color='red', linestyle='dashed')

plt.xlabel('leverage')
plt.ylabel('studentized resid')
plt.title('Influence plot')

print('Close plot window to continue...')

plt.show()

print(f"\nInfluence plot shows no high influence points.\n")

input("Press Enter to continue...")
