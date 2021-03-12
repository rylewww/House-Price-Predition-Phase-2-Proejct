# Kings County House Pricing Regression Analysis

<img src='https://www.kingcounty.gov/~/media/depts/assessor/images/2015/assessors_social.ashx?la=en' alt= 'Prices image'>

(technical presentation located in Final Notebook.ipynb

## Introduction

This project uses linear regression modeling to best predict house prices in the greater Seattle, WA area.

## Data Overview

<img src='https://github.com/rylewww/Phase-2-Project/blob/main/Images/Property%20Price%20Heat%20Map.png' alt= 'Prices image'>

The data set provided represented 17,290 properties sold in Kings County, WA. For each property we were given additional details relating to square feet, condition of the house, number of bredrooms and bathrooms, location, date sold ect. 

## Exploratory Data Analysis

The data provided was clean and void of any null values, so majority of the data prep for our EDA related to dealing with outliers. While exploring our data it became clear how strongly location can relate house prices. You can see in the heatmap and the zipcode barchart below how mucher higher the mean price is for certain parts of the county versus others.

<img src='https://github.com/rylewww/Phase-2-Project/blob/main/Images/Price%20by%20Zipcode.png' alt= 'Zip image'>

During the EDA process, statistcal tests were run on:

  1. Mean house prices located in northern KC vs southern KC - there was a statistical difference; houses in the north had a higher mean.
  2. Mean house prices of homes with waterfront vs without - there was a statistical difference, waterfront homes carried a higher mean.
  3. Mean house prices of homes at certain grades - there was a statistical difference, homes at a higher grade have a higher mean price.
  4. Mean house prices of homes in different condiditions - there was a statistical difference, better condition correlates to higher mean price.

In all of our statistical testings we rejected the null hypothesis - this tell us that these features have an impact on price and could be used in our linear model to come.

## Feature Engineering

Engineering new features out of existing ones can sometimes really help show how certain features can be driving your target.  In my analysis I created new features such as:

  1. Bath:Bed Ratio - do houses with a 1+ ratio tend to have a higher average sales price?
  2. House Age - taking the year the house was built and subtracting that from the current year. Do newer houses sell for a higher price?
  3. Distance from Major Businesses - do houses that are closer to Amazon HQ sell for a higher average price?

<img src='https://github.com/rylewww/Phase-2-Project/blob/main/Images/Distance%20Scatterplot.png' alt= 'Distance image'>

## Inference Modeling

With our data cleaned up, insights from our EDA and some newly engineered features we are ready to create our first model to see how well we can explain the variance around the mean in our target: price. Removing all features that too closely correlate with eachother (.90+ can lead to multicollinearity) and then any additional columns deemed not needed; I was able to create a model with an r^2 = 0.819! I focused my model around 'sqft_living', 'dist_amazon' and  used dummy categories for my zipcodes, grades, conditions and waterfront. This resulted in our mdoel roughly having ~100 features.

From the OLS output I was able to infer that if all other coefficients held constant one addtiional sqft_living would add $155 of value, and could also confirm how important some major zipcodes, number of bathrooms and waterfront as they all had very low p-values of 0.000.

## Prediction Modeling

### Original Model

Running our original model through the 'test_train_split' function returned as:

  1. Training RMSE = 158715
  2. Test RMSE = 151964

This shows us we might be slightly overfitting our model to the test data, as noted with the higher training RMSE. These error scores are good, but can we get them lower using additional feature selecion methods?

### Polynomial Regression Model

In attempt to lower our RMSE, we implemented a Polynomial based approach to see if any of our features raised to the second degree would best explain the relationship between our features and target. Polynomial regression helps fit any non-linear correlation we have in our model. This took our model dataframe of ~100 features to over 6,000. 

<img src='https://animoidin.files.wordpress.com/2018/07/polim_vs_linear.jpg' alt= 'Poly image'>

Running this model through 'train_test_split':

  1. Training RMSE = 93009
  2. Testing RMSE = 183223

While this certainly lowered our training RMSE, our testing RMSE is nearly doubled. This clearly shows that we are very overfitted on the training data set. 

### Select K Best Polynomial Regression Model

A method to reduce over-fitting and also selecting the highest scoring features is to use 'SelectKBest.' SKB is a Scikit-Learn wrapper that scores all features and removes all but the k-highest scores. In takes in two arrays (X,y) and returns a single array of scores and p-values. You set the "k" number to how many features you want to have after the SKB is run. For example we set it to k = 100, giving us the 100 best scoring features out of 6,000 polynomial features. This removes a lot of categorical fetures that got swept up into the 2nd degree. 

After running our model with our SKB columns we got:

  1. Training RMSE = 144835
  2. Testing RMSE = 149626

While this brought up pur training our RMSE quite it bit, it also leveled out our testing RMSE signifying we're not overfitting as drastically. 

## Final Model Choice

After comparing all of our models, I decided to move forward with the SKB Polynomial Model as it had the lowest test RMSE score at 149,926 and the lowest % difference in training vs testing RMSE showing that it is the least overfit of our models. Our final r^2 value = 0.847, showing that it does a strong job explaining the variation around the mean of our target.

## Future Work

Moving forward I would integrate time of year houses are sold into the model, and look into binning zipcodes together to better target which areas of Kings County drive price. 
