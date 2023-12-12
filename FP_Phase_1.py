import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import normaltest
from scipy.stats import shapiro
from scipy.stats import boxcox
from scipy import stats
from scipy.stats import kstest
from prettytable import PrettyTable
import warnings

#pd.options.display.float_format = "{:,.2f}".format
#plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

warnings.filterwarnings("ignore")

url = "/Users/hinjam/Library/CloudStorage/GoogleDrive-iharitha@vt.edu/My Drive/Information Visualization/LAB:Assignments/Myntra Fasion Clothing.csv"

    #url = '/kaggle/input/myntra-fashion-dataset/Myntra Fasion Clothing.csv'

df = pd.read_csv(url)

    # csv_file_path = '/Users/hinjam/Original_myntra_unfiltered.csv'
    # df.to_csv(csv_file_path, index=False)

myntra = df.copy()

myntra.rename(columns={'DiscountPrice (in Rs)': 'Final Price (in Rs)'}, inplace=True)

colors = ['red', 'green', 'blue', 'black', 'white', 'yellow', 'pink', 'navy',
              'olive', 'maroon', 'khaki', 'burgundy', 'grey', 'beige', 'orange',
              'purple', 'lavender', 'brown', 'mauve','peach','violet','magenta']


def extract_first_color(desc):
    for color in colors:
        if color in desc:
            return color
    return None  # Return None if no color is found

    # Apply the function to create a new color column
myntra['Individual_Category_Colour'] = myntra['Description'].apply(extract_first_color)

    # Drop rows where 'Individual_Category_Colour' is None
myntra = myntra.dropna(subset=['Individual_Category_Colour'])

    # Assuming myntra is your DataFrame
myntra = myntra.drop(['URL', 'Product_id', 'Description'], axis=1)

    #print(myntra.head())

    # Check for missing values using isna() and isnull()

missing_values_na = myntra.isna().sum().sum()
missing_values_null = myntra.isnull().sum().sum()

print(f'Missing values using isna(): \n{missing_values_na}')
print(f'Missing values using isnull():\n{missing_values_null}')

myntra['Discount Amount']=myntra['OriginalPrice (in Rs)']-myntra['Final Price (in Rs)']
    # myntra['Discount Amount'].sort_values (ascending=False)

#CleanUp
myntra.dropna(inplace=True)


# Confirming that dataset is clean
missing_values_na_after = myntra.isna().sum().sum()
print(f'Missing values after dropping using isna(): {missing_values_na_after}')

missing_values_null_after = myntra.isnull().sum().sum()
print(f'Missing values after dropping using isnull(): {missing_values_null_after}')

print('After Data Cleaning!')
print(myntra.shape[0])
print(myntra.head(5))
myntra.describe()

# csv_file_path = '/Users/hinjam/1_before_myntra_filtered.csv'
# myntra.to_csv(csv_file_path, index=False)


#Outlier detection & Treatment

# Outlier Detection Function
def find_outliers_IQR(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return q1, q3, iqr, lower_bound, upper_bound, outliers

# Using the Outlier Detection Function
Q1, Q3, IQR, lower_bound, upper_bound, OriginalPrice_outliers = find_outliers_IQR(myntra['OriginalPrice (in Rs)'])

print("Q1, Q3, IQR, Lower Bound, Upper Bound:")
print(Q1, Q3, IQR, lower_bound, upper_bound)

print(f'Prices lower than {lower_bound} and higher than {upper_bound} are considered outliers.')

# Display basic statistics
print(myntra['OriginalPrice (in Rs)'].describe())

# Output number of outliers and their statistics
print(f'Number of outliers: {len(OriginalPrice_outliers)}')
print(f'Max outlier value:  {OriginalPrice_outliers.max()}')
print(f'Min outlier value:  {OriginalPrice_outliers.min()}')

plt.boxplot(myntra['OriginalPrice (in Rs)'])
plt.title('Boxplot of OriginalPrice(s)',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.ylabel('Price (in Indian Rupees)',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.xticks([1], ['OriginalPrice'],fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})

# Highlighting the outlier thresholds
plt.axhline(y=lower_bound, color='r', linestyle='--', label=f'Lower Bound = {lower_bound:.2f} inches')
plt.axhline(y=upper_bound, color='g', linestyle='--', label=f'Upper Bound = {upper_bound:.2f} inches')

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# If you want to remove outliers
myntra_filt = myntra[~myntra['OriginalPrice (in Rs)'].isin(OriginalPrice_outliers)]
#print(myntra_filt)

plt.figure(figsize=(8, 6))
plt.boxplot(myntra_filt['OriginalPrice (in Rs)'])
plt.title("Boxplot of OriginalPrice(s) After Removing Outliers",fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xticks([1], ['OriginalPrice'],fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel("Price (in Indian Rupees)",fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

csv_file_path = '/Users/hinjam/no_out_myntra_filtered.csv'
myntra_filt.to_csv(csv_file_path, index=False)

#PCA Analysis

# a. Standardize the feature space

scaler = StandardScaler()
numerical_cols = ['Final Price (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews', 'Discount Amount']
df_numerical = np.around(myntra_filt[numerical_cols],2)

print(df_numerical)
scaled_data = scaler.fit_transform(df_numerical)
print(scaled_data)

# # b. singular values and condition number for the original feature space

U, S, VT = np.linalg.svd(scaled_data, full_matrices=False)
# condition_number_original = np.max(S) / np.min(S)
print(U)
print(VT)
print("Singular Values for Original Feature Space:", np.around(S, 2))
print("Condition Number for Original Feature Space:", np.around(np.linalg.cond(scaled_data), 2))


# # c. the correlation coefficient matrix between all features
corr_matrix = np.around(np.corrcoef(df_numerical, rowvar=False),2)
print("Correlation Coefficient between features-Original feature space")
print(corr_matrix)
plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix, annot=True)
plt.title("Corr Coeff features-Original feature space",fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.tight_layout()
plt.show()

# # d. PCA analysis and finding the number of features to be removed
# # Create a PCA model

pca = PCA()
pca.fit(scaled_data)
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)
cumulative_variance = np.cumsum(explained_variance_ratio)
print(cumulative_variance)

# # Determine the number of components to achieve 95% explained variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(n_components_95)
print("\n\nExplained Variance Ratio (Original Feature Space):", explained_variance_ratio)
print("Cumulative Explained Variance Ratio (Original Feature Space):", cumulative_variance)
print(f"Number of features to be removed: {scaled_data.shape[1] - n_components_95}")

# e. Graphing the cumulative explained variance

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_)+1,1), 100*np.cumsum(pca.explained_variance_ratio_),
         lw = 3)
plt.axvline(n_components_95, color='red', linestyle='--', label=f'Optimum Number of Features: {n_components_95}')
plt.axhline(95, color='black', linestyle='--', label='95% Explained Variance')
plt.xlabel('Number of Components',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Cumulative Explained Variance (%)',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.title('Cumulative Explained Variance vs. Number of Components',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
# plt.legend()
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# # f. Finding the singular values and condition number for the reduced feature space
n_Components = 0.95
pca = PCA(n_components=n_Components, svd_solver='full')
reduced_data = pca.fit_transform(scaled_data)
U_reduced, S_reduced, VT_reduced = np.linalg.svd(reduced_data, full_matrices=False)
reduced_explained_variance_ratio = pca.explained_variance_ratio_
print("\n\nExplained Variance Ratio (Reduced Feature Space):", np.around(reduced_explained_variance_ratio, 2))
print("Singular Values for Reduced Feature Space:",  np.around(S_reduced, 2))
print("Condition Number for Reduced Feature Space:", np.around(np.linalg.cond(reduced_data), 2))

# # g. Finding the correlation coeff matrix for the reduced feature space
corr_matrix_reduced = np.around(np.corrcoef(reduced_data, rowvar=False), 2)
print("Correlation Coefficient Matrix (Reduced)")
print(corr_matrix_reduced)
plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix_reduced, annot=True)
plt.title("Correlation Coefficient Matrix (Reduced)",fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.show()


# Create a new DataFrame with the transformed columns
reduced_df = pd.DataFrame(reduced_data, columns=[f'Principal col {i + 1}' for i in range(n_components_95)])
print("First 5 rows of the Reduced DataFrame:")
print(np.around(reduced_df.head(), 2))

n_rows = int(n_components_95)
features = [f'Principal col {i + 1}' for i in range(n_components_95)]

    # Create subplots
fig, axes = plt.subplots(n_rows, 1, figsize=(20, 6 * n_rows), sharex=True)

    # Plot histograms for each transformed feature
for i, feature in enumerate(features):
        ax = axes[i]
        ax.hist(reduced_df[feature], bins=30, color='skyblue', edgecolor='black', linewidth=1.2)
        ax.set_title(f'Histogram of {feature}', fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
        ax.set_ylabel('Frequency', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
        ax.grid()

    # Add labels and title
plt.xlabel('Transformed Feature Value', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
#plt.suptitle('Histogram Plots of Transformed Features')

plt.tight_layout()
plt.show()


#Normality Test

def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    print('='*50)
    print(f'K-S test: {title} dataset: statistics= {stats:.2f} p-value = {p:.2f}' )

ks_test(myntra_filt['OriginalPrice (in Rs)'], "Myntra OriginalPrice Normality Test KSTest")

def da_k_squared_test(x, title):
    stats, p = normaltest(x)
    print('='*50)
    print(f'da_k_squared test: {title} dataset: statistics= {stats:.2f} p-value = {p:.2f}' )


da_k_squared_test(myntra_filt['OriginalPrice (in Rs)'], "Myntra OriginalPrice DAK Normality Test")

def shapiro_test(x,title):
    stats, p = shapiro(x)
    print('=' * 50)
    print(f'Shapiro test : statistics = {stats:.2f} p-value of ={p:.2f}')
    if p > 0.01:
        print(f'{title} dataset looks Normal with 99% accuracy')
    else:
        print(f'{title} dataset looks Not Normal with 99% accuracy')

# Perform shapiro for height and weight
shapiro_test(myntra_filt['OriginalPrice (in Rs)'],'Myntra OriginalPrice Normality Test Shapiro Test')

# # # =========================
# # # B0x-Cox transformation
# # # ========================
# # sns.distplot(myntra['Final Price (in Rs)'],hist=True,kde=True)
# transformed_data, best_lamda = boxcox(myntra['Final Price (in Rs)'])
#
# sns.distplot(transformed_data, hist=True, kde=True)
# plt.show()
# print(f'The best lambda is {best_lamda}')
#
# fig, ax = plt.subplots(figsize=(8, 4))
# prob = stats.boxcox_normplot(myntra['Final Price (in Rs)'], -10, 10, plot=ax)
# ax.axvline(best_lamda, color = 'r')
# plt.show()

# da_k_squared_test(transformed_data, "Myntra Final Price Normality Test(transformed)")

# # Ensured all values are positive (Box-Cox requires positive data)
original_prices = myntra_filt['OriginalPrice (in Rs)']  # Add 1 if there are zero values
original_final_price = myntra_filt['Final Price (in Rs)']
original_discount_amount =  myntra_filt['Discount Amount']

# # Apply Box-Cox Transformation
transformed_data, best_lambda = stats.boxcox(original_prices)
transformed_final_price, best_lambda = stats.boxcox(original_final_price)
transformed_discount_amount, best_lambda = stats.boxcox(original_discount_amount)



# print('**Transformation**')
# print(transformed_data)

#myntra_filt['OriginalPrice (in Rs)'] = transformed_data
# myntra_filt['Final Price (in Rs)'] = transformed_final_price
# myntra_filt['Discount Amount'] = transformed_discount_amount
#
print('After Transformation applied')
print(myntra_filt.head(15))

da_k_squared_test(transformed_data, "Myntra Transformed OriginalPrice DAK Normality Test")
ks_test(transformed_data, "Myntra Transformed OriginalPrice KS Normality Test")

# Plotting histograms
plt.figure(figsize=(14, 6))

# Original Data Histogram
plt.subplot(1, 2, 1)
plt.hist(original_prices, bins=30, color='blue', edgecolor='black')
plt.title('Histogram of Original Data',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Original Price (in Rs)', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid()

# Transformed Data Histogram
plt.subplot(1, 2, 2)
plt.hist(transformed_data, bins=30, color='blue', edgecolor='black')
plt.title('Histogram of Transformed Data',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Transformed Values', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

# Final Price Data Histogram
plt.subplot(1, 2, 1)
plt.hist(original_final_price, bins=30, color='blue', edgecolor='black')
plt.title('Histogram of Final Price',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Final Price (in Rs)', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid()

# Transformed Data Histogram
plt.subplot(1, 2, 2)
plt.hist(transformed_final_price, bins=30, color='blue', edgecolor='black')
plt.title('Histogram of Transformed Data',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Transformed Values', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))

# Discount Amount Data Histogram
plt.subplot(1, 2, 1)
plt.hist(original_discount_amount, bins=30, color='blue', edgecolor='black')
plt.title('Histogram of Discount Amount',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Discount Amount', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid()

# Transformed Data Histogram
plt.subplot(1, 2, 2)
plt.hist(transformed_discount_amount, bins=30, color='blue', edgecolor='black')
plt.title('Histogram of Transformed Data',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Transformed Values', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid()
plt.tight_layout()
plt.show()

# Plotting QQ-plots

plt.figure(figsize=(14, 6))

# QQ-plot for Original Data
plt.subplot(1, 2, 1)
stats.probplot(original_prices, dist="norm", plot=plt)
plt.title('QQ-plot of Original Data',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Theoretical Quantiles', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Sample Quantities', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid()

# QQ-plot for Transformed Data
plt.subplot(1, 2, 2)
stats.probplot(transformed_data, dist="norm", plot=plt)
plt.title('QQ-plot of Transformed Data',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Theoretical Quantiles', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Sample Quantities', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid()
plt.tight_layout()
plt.show()

# Plotting QQ-plots
plt.figure(figsize=(14, 6))

# QQ-plot for Final Price Data
plt.subplot(1, 2, 1)
stats.probplot(original_final_price, dist="norm", plot=plt)
plt.title('QQ-plot of Final Data',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Theoretical Quantiles', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Sample Quantities', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid()


# QQ-plot for Transformed Data
plt.subplot(1, 2, 2)
stats.probplot(transformed_final_price, dist="norm", plot=plt)
plt.title('QQ-plot of Transformed Data',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Theoretical Quantiles', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Sample Quantities', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid()
plt.tight_layout()
plt.show()


# Plotting QQ-plots
plt.figure(figsize=(14, 6))

# QQ-plot for Discount Amount Data
plt.subplot(1, 2, 1)
stats.probplot(original_discount_amount, dist="norm", plot=plt)
plt.title('QQ-plot of Discount Amount',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Theoretical Quantiles', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Sample Quantities', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid()

# QQ-plot for Transformed Data
plt.subplot(1, 2, 2)
stats.probplot(transformed_discount_amount, dist="norm", plot=plt)
plt.title('QQ-plot of Transformed Data',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Theoretical Quantiles', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Sample Quantities', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid()
plt.tight_layout()
plt.show()


#Table1

x = PrettyTable()

Category_OriPrice = myntra_filt.groupby('Category')['OriginalPrice (in Rs)'].mean()
Category_DiscountPrice = myntra_filt.groupby('Category')['Final Price (in Rs)'].mean()
Category_DisPrice = myntra_filt.groupby('Category')['Discount Amount'].mean()

Original_by_category = Category_OriPrice.reset_index()
DiscountPrice_by_category = Category_DiscountPrice.reset_index()
DisPrice_by_category = Category_DisPrice.reset_index()

merged_df = pd.merge(Original_by_category, DiscountPrice_by_category, on='Category')
merged_df = pd.merge(merged_df, DisPrice_by_category, on='Category')

merged_df = merged_df.reset_index(drop=True)
print(merged_df)

max_OriPrice_value = Category_OriPrice.max()
max_DiscountPrice_value = Category_DiscountPrice.max()
max_DisPrice_value = Category_DisPrice.max()

min_OriPrice_value = Category_OriPrice.min()
min_DiscountPrice_value = Category_DiscountPrice.min()
min_DisPrice_value = Category_DisPrice.min()


# Finding the categories corresponding to the maximum values
category_max_OriPrice = Category_OriPrice.idxmax()
category_max_DiscountPrice = Category_DiscountPrice.idxmax()
category_max_DisPrice = Category_DisPrice.idxmax()

# Finding the categories corresponding to the minimum values
category_min_OriPrice = Category_OriPrice.idxmin()
category_min_DiscountPrice = Category_DiscountPrice.idxmin()
category_min_DisPrice = Category_DisPrice.idxmin()

x.field_names = ["Category", "OriginalPrice", "Final Price", "Discount Amount"]

for index, row in merged_df.iterrows():
    # Formatting each value before adding it to the table
    x.add_row([row['Category'],
               f"{row['OriginalPrice (in Rs)']:.2f}",
               f"{row['Final Price (in Rs)']:.2f}",
               f"{row['Discount Amount']:.2f}"])

x.add_row(['Maximum Value',f"{max_OriPrice_value:.2f}",f"{max_DiscountPrice_value:.2f}",f"{max_DisPrice_value:.2f}"])
x.add_row(['Minimum Value',f"{min_OriPrice_value:.2f}",f"{min_DiscountPrice_value:.2f}",f"{min_DisPrice_value:.2f}"])

# Add idmax and idmin rows
x.add_row(['Category Max Value', category_max_OriPrice,category_max_DiscountPrice,category_max_DisPrice])
x.add_row(['Category Min Value', category_min_OriPrice,category_min_DiscountPrice,category_min_DisPrice])

print(x.get_string(title="Category-wise Financial Analysis of Fashion Products"))

#Table2
#print(myntra_filt.head())
# Calculating the required statistics for the numerical features
mean_values = myntra_filt[['Final Price (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews', 'Discount Amount']].mean()
variance_values = myntra_filt[['Final Price (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews', 'Discount Amount']].var()
std_dev_values = myntra_filt[['Final Price (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews', 'Discount Amount']].std()
median_values = myntra_filt[['Final Price (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews', 'Discount Amount']].median()

# Creating a PrettyTable
table = PrettyTable()

# Adding the field names (column headers)
table.field_names = ["Statistic", "Final Price (in Rs)", "OriginalPrice (in Rs)", "Ratings", "Reviews", "Discount Amount"]

# Adding rows for Mean, Variance, Standard Deviation, and Median
table.add_row(["Mean", round(mean_values['Final Price (in Rs)'], 2), round(mean_values['OriginalPrice (in Rs)'], 2), round(mean_values['Ratings'], 2), round(mean_values['Reviews'], 2), round(mean_values['Discount Amount'], 2)])
table.add_row(["Variance", round(variance_values['Final Price (in Rs)'], 2), round(variance_values['OriginalPrice (in Rs)'], 2), round(variance_values['Ratings'], 2), round(variance_values['Reviews'], 2), round(variance_values['Discount Amount'], 2)])
table.add_row(["Standard Deviation", round(std_dev_values['Final Price (in Rs)'], 2), round(std_dev_values['OriginalPrice (in Rs)'], 2), round(std_dev_values['Ratings'], 2), round(std_dev_values['Reviews'], 2), round(std_dev_values['Discount Amount'], 2)])
table.add_row(["Median", round(median_values['Final Price (in Rs)'], 2), round(median_values['OriginalPrice (in Rs)'], 2), round(median_values['Ratings'], 2), round(median_values['Reviews'], 2), round(median_values['Discount Amount'], 2)])

print(table.get_string(title="statistics for each numerical feature in the dataset"))

# grouped_tab_data = myntra.groupby(['Category', 'Individual_category'])['Final Price (in Rs)'].mean().reset_index()

# Group by 'Category' and calculate the average discount price
grouped_bar_data = myntra_filt.groupby('Category')['Discount Amount'].mean().reset_index()
grouped_tab_data = myntra_filt.groupby(['Category', 'Individual_category'])['Final Price (in Rs)'].mean().reset_index()

# List of selected brands
selected_brands = [
    "VIMAL", "Go Colors", "max", "Dollar", "Qurvii", "Nanda Silk Mills",
    "Manyavar", "SOUNDARYA", "Indian Terrain", "URBANIC", "Being Human",
    "RAMRAJ COTTON", "Fabindia", "Ethnix by Raymond", "CAVALLO by Linen Club",
    "Linen Club", "Oxemberg", "Armaan Ethnic", "Hastakala", "KLM Fashion Mall",
    "Women Republic", "taruni", "PINKVILLE JAIPUR", "Zanani INDIA"
]

myntra_filt['Ratings'] = pd.to_numeric(myntra_filt['Ratings'], errors='coerce')
grouped_avg_ratings = myntra_filt.groupby(['BrandName', 'category_by_Gender'])['Ratings'].mean().reset_index()



# Calculate the count of SizeOptions for each Category
category_size_counts = myntra_filt.groupby('Category')['SizeOption'].apply(lambda x: ', '.join(x)).reset_index()
category_size_counts['SizeOption'] = category_size_counts['SizeOption'].str.split(', ').apply(len)


#LinePlot

# Create a Dropdown for category selection (outside of Matplotlib)
selected_category = myntra_filt['Category'].iloc[0]

# Filter by the selected category
filtered_data = myntra_filt[myntra_filt['Category'] == selected_category]

# Group by 'Individual_category' and calculate the average discount price
grouped_data = filtered_data.groupby('Individual_category')['Final Price (in Rs)'].mean().reset_index()
print(grouped_data)
# Create the figure using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(grouped_data['Individual_category'], grouped_data['Final Price (in Rs)'], marker='o')
plt.title(f'Average Price After Discount for {selected_category}', fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Individual_category', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Final Price (in Rs)', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
#plt.xticks(rotation=90)
plt.grid(True)
#
# # Show the Matplotlib plot
# plt.show()

#Barchart

bar_fig = px.bar(grouped_bar_data, x='Category', y='Discount Amount', title='Average Discount Amount for Each Category')
bar_fig.update_layout(title_x=0.5,
                      font=dict(family='serif', size=20),
                      title_font_color='blue',
                      xaxis_title_font=dict(family='serif', size=15, color='darkred'),
                      yaxis_title_font=dict(family='serif', size=15, color='darkred'))
bar_fig.show()

#Barchart(Stack)

#Filter the DataFrame to only include the selected brands
filtered_df = myntra_filt[myntra_filt['BrandName'].isin(selected_brands)]

# Group by Brand and Gender, then count the number of products for each combination
grouped_data = filtered_df.groupby(['BrandName', 'category_by_Gender']).size().reset_index(name='Count')

# Create the stacked bar plot
fig = px.bar(grouped_data, x='BrandName', y='Count', color='category_by_Gender',
             title="Brand Preference by Gender for Selected Brands",
             labels={'Count': 'Number of Products'},
             color_discrete_map={'Men': 'blue', 'Women': 'pink'},
             #height=600,  # Adjust height to ensure all brand labels are visible
             #width=1000
            )
fig.update_layout(title_x=0.5,
                      font=dict(family='serif', size=20),
                      title_font_color='blue',
                      xaxis_title_font=dict(family='serif', size=15, color='darkred'),
                      yaxis_title_font=dict(family='serif', size=15, color='darkred'))
fig.show()


#Barchart(Group)

filtered_df = grouped_avg_ratings[grouped_avg_ratings['BrandName'].isin(selected_brands)]


fig = px.bar(filtered_df,
                 x='BrandName',
                 y='Ratings',
                 color='category_by_Gender',
                 barmode='group',
                 title='Average Ratings by Brand and Gender for Selected Brands')
fig.update_layout(title_x=0.5,
                      font=dict(family='serif', size=20),
                      title_font_color='blue',
                      xaxis_title_font=dict(family='serif', size=15, color='darkred'),
                      yaxis_title_font=dict(family='serif', size=15, color='darkred'))
fig.show()

#Countplot

# Set the style
sns.set(style="darkgrid")

# Define bins
bins = [0, 10, 20, 50, 100, 500, 1000, 5000, 10000]

plt.figure(figsize=(12, 7))
myntra_filt['Reviews'] = pd.to_numeric(myntra_filt['Reviews'], errors='coerce')
# Create a new column 'Review_Bins' to store the binned data
myntra_filt['Review_Bins'] = pd.cut(myntra_filt['Reviews'], bins, labels=["0-10", "11-20", "21-50", "51-100", "101-500", "501-1000", "1001-5000", "5001-10000"])

sns.countplot(x='Review_Bins', data=myntra_filt)
#plt.xticks(rotation=45, horizontalalignment='right')

# Display the plot
plt.title("Count of Products by Review Bins",fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel("Review Bins",fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel("Number of Products",fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.show()


#PieChart

indian_wear_men_df = myntra_filt[(myntra_filt['Category'] == 'Indian Wear') & (myntra_filt['category_by_Gender'] == 'Women')]

# Count the occurrences of '28, 30, 32, 34, 36' and 'OneSize'
count_28_to_36 = indian_wear_men_df[indian_wear_men_df['SizeOption'] == '28, 30, 32, 34, 36'].shape[0]
count_one_size = indian_wear_men_df[indian_wear_men_df['SizeOption'] == 'Onesize'].shape[0]

# print(count_28_to_36)
# print(count_one_size)

# Create a DataFrame for the pie chart
size_options_count = pd.DataFrame({
    'Size Option': ['28, 30, 32, 34, 36', 'Onesize'],
    'Count': [count_28_to_36, count_one_size]
})

# Plot the pie chart
fig = px.pie(size_options_count, names='Size Option', values='Count', title="Size Options for Bottom Wear (Men)")
fig.update_layout(title_x=0.5,
                      font=dict(family='serif', size=20),
                      title_font_color='blue',
                      xaxis_title_font=dict(family='serif', size=15, color='darkred'),
                      yaxis_title_font=dict(family='serif', size=15, color='darkred'))
fig.show()

#Displot

# #Filter the dataset for rows where OriginalPrice is below 10,000
# filtered_myntra = myntra_filt[myntra_filt["OriginalPrice (in Rs)"] < 5000]
#
# # Set the style
# sns.set_style("darkgrid")
#
# # Create the displot
# sns.displot(data=filtered_myntra, x="OriginalPrice (in Rs)", kde=True, col="category_by_Gender",binwidth=100)
#
# # Set the plot title and display the plot
# plt.suptitle('Distribution of Original Prices Below 10,000', fontsize=20, color='blue', fontname='serif')
# plt.tight_layout()
# plt.show()

#Correct One below

# Filter the dataset for rows where OriginalPrice is below 10,000
filtered_myntra = myntra_filt[myntra_filt["OriginalPrice (in Rs)"] < 5000]

# Set the style
sns.set_style("darkgrid")

# Create the displot and capture the FacetGrid object
g = sns.displot(data=filtered_myntra, x="OriginalPrice (in Rs)", kde=True, col="category_by_Gender", binwidth=100)
#g = sns.displot(data=filtered_myntra, x="OriginalPrice (in Rs)", kde=True, col="category_by_Gender")
# Set the plot title
g.fig.suptitle('Distribution of Original Prices Below 5,000', fontsize=20, color='blue', fontname='serif')

# Set X and Y labels with custom fontdict attributes
g.set_axis_labels("Original Price (in Rs)", "Density")
g.set_titles(fontsize=15, color='darkred', fontname='serif')
for ax in g.axes.flat:
    ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
    ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})

# Adjust layout
plt.tight_layout()
plt.show()


#Pairplot

#Main

# # Subset the data: select 10% randomly
# subset_myntra_df = myntra_filt.sample(frac=0.1)
#
# # Plot the pairplot using the subsetted data
# sns.pairplot(subset_myntra_df,
#              vars=["Final Price (in Rs)", "OriginalPrice (in Rs)", "Ratings", "Reviews", ""],
#              hue="Category")
#
# plt.show()

# numeric_columns = ['Ratings', 'Reviews', 'Final Price (in Rs)', 'OriginalPrice (in Rs)']
# numeric_df = myntra_filt[numeric_columns].apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric
# sns.pairplot(numeric_df.dropna())  # Drop rows with NaN values which can't be plotted

#Correct one below

sns.set(style="darkgrid")
myntra_filt['Reviews'] = pd.to_numeric(myntra_filt['Reviews'], errors='coerce')
# Subset the data: select 10% randomly
subset_myntra_df = myntra_filt.sample(frac=0.1)

# Plot the pairplot using the subsetted data
pair_plot = sns.pairplot(subset_myntra_df,
             vars=["Final Price (in Rs)", "OriginalPrice (in Rs)", "Ratings", "Reviews", "Discount Amount"],
             hue="Category",
             palette="viridis")


# Set the title and labels using Matplotlib
pair_plot.fig.suptitle('Pairplot of Selected Features', color='blue', size=20, fontname='serif')

# To set the font properties for axis labels, you need to iterate over the axes
for ax in pair_plot.axes.flatten():
    # Set the font properties for x-axis labels
    ax.set_xlabel(ax.get_xlabel(), fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 15})
    # Set the font properties for y-axis labels
    ax.set_ylabel(ax.get_ylabel(), fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 15})

# Show the plot
plt.show()



#Heatmap with cmap

# HeatMap with Color Bar

# Calculate the correlation matrix
# corr_matrix = df_numerical.corr()
# # Plot a heatmap for the correlation matrix
# plt.figure(figsize=(10,8))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Heatmap of Correlation Matrix",fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
# plt.tight_layout()
# plt.show()

# corr_matrix = df_numerical.corr()
# # Plot a heatmap for the correlation matrix
# plt.figure(figsize=(10,8))
# heatmap = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Heatmap of Correlation Matrix", fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
#
# # Set custom font properties for x-axis and y-axis labels
# heatmap.set_xlabel(heatmap.get_xlabel(), fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
# heatmap.set_ylabel(heatmap.get_ylabel(), fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
#
# plt.tight_layout()
# plt.show()

# Calculate the correlation matrix
corr_matrix = df_numerical.corr().round(2)
print(corr_matrix)

# Plot a heatmap for the correlation matrix
plt.figure(figsize=(10,8))
heatmap = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Heatmap of Correlation Matrix", fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})

# Set custom font properties for x-axis and y-axis labels using the axes
plt.xticks(rotation=45, fontsize=15, color='darkred', fontname='serif')
plt.yticks(rotation=45, fontsize=15, color='darkred', fontname='serif')

plt.tight_layout()
plt.show()


#Approach2

# Determine the top 5 brands by some criterion, here we take the first 5 unique brands
top_brands = myntra_filt['BrandName'].value_counts().index[:5]

# Filter the DataFrame to include only the top 5 brands
filtered_df = myntra_filt[myntra_filt['BrandName'].isin(top_brands)]

# Pivot the filtered DataFrame
pivot_df = filtered_df.pivot_table(values='Ratings', index='BrandName', columns='Category', aggfunc='mean')

# Since the data may have NaN values where combinations do not exist, we'll fill them with zeros
pivot_df = pivot_df.fillna(0)

formatted_annotations = np.around(pivot_df.values, decimals=2).astype(str)


# Create the heatmap with white annotations
fig = go.Figure(data=go.Heatmap(
    z=pivot_df.values,
    x=pivot_df.columns,
    y=pivot_df.index,
    colorscale='peach',
    colorbar=dict(title='Average Ratings'),
    text=formatted_annotations,
    texttemplate="%{text}",
    textfont=dict(size=25, color='white')  # Set annotation text font and color here
))

# Update the layout
fig.update_layout(
    title='Heat Map of Average Ratings by Top 5 Brands and Category',
    title_font_color='blue',
    font=dict(family='serif', size=20),
    xaxis_title='Category',
    yaxis_title='BrandName',
    xaxis_title_font=dict(family='serif', size=15, color='darkred'),
    yaxis_title_font=dict(family='serif', size=15, color='darkred')
)
fig.show()


#QQplot

# # Filter out rows where 'Ratings' is null
# myntra_data = myntra_filt[myntra_filt['Ratings']!= 'nill']
#
#
# sample_size = 1000
# myntra_sample = myntra_data.sample(n=sample_size, random_state=42)
#
# #Create a QQ plot
# ratings = myntra_sample['Ratings']
#
# #ratings = myntra_data['Ratings']
# sm.qqplot(ratings, line='45')
# plt.title('QQ Plot of Myntra Ratings')
# plt.show()

#Correct One

# Assuming myntra is a DataFrame that has been previously defined
# Filter out rows where 'Ratings' is null or 'nill'
myntra_data = myntra_filt[myntra_filt['Ratings'] != 'nill']

# Convert 'Ratings' to numeric after filtering
myntra_data['Ratings'] = pd.to_numeric(myntra_data['Ratings'])

# Take a sample size of 1000
sample_size = 1000
myntra_sample = myntra_data.sample(n=sample_size, random_state=42)

# Create a QQ plot
ratings = myntra_sample['Ratings']
fig = sm.qqplot(ratings, line='45')

# Customize the plot with title and labels using fontdict
plt.title('QQ Plot of Myntra Ratings', fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Theoretical Quantiles', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Sample Quantities', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})

# Show the plot
plt.show()


#Kde plot

sns.set(style="darkgrid")
#
columns = ['Reviews','Ratings','Final Price (in Rs)','OriginalPrice (in Rs)']
#
# myntra_filt['Ratings'] = pd.to_numeric(myntra_filt['Ratings'], errors='coerce')
# myntra_filt['Reviews'] = pd.to_numeric(myntra_filt['Reviews'], errors='coerce')
#
# # Correct One below
#
# Set up the matplotlib figure (2 rows, 2 columns in this case)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))


palette = "coolwarm"

# Flatten axes array for easy iterating
axes = axes.flatten()
titles = ['KDE of Reviews', 'KDE of Ratings', 'KDE of Final Prices', 'KDE of Original Prices']

for i, column in enumerate(columns):
    sns.kdeplot(x=myntra_filt[column], hue=myntra_filt['category_by_Gender'], ax=axes[i], fill=True, alpha=0.6, palette=palette, linewidth=2)
    # axes[i].set_xticks(axes[i].get_xticks())  # This line ensures xticks are set based on the data
    # axes[i].tick_params(axis='x', labelsize=14)  # Set the size of the x-axis tick labels
    axes[i].set_title(titles[i], fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
    axes[i].set_xlabel(column, fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
    axes[i].set_ylabel('Density', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
    axes[i].grid(True)  # Ensure grid is turned on for each subplot

plt.tight_layout()
plt.show()

# for i in columns:
#     plt.figure(figsize=(14,7))
#     sns.kdeplot(x=myntra[i], hue=myntra['category_by_Gender'])
#     plt.xticks(fontsize=14)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


#Histogram plot with kde

sns.set(style="darkgrid")
# Filter out the "0" values in the rating column
filtered_ratings = myntra_filt[myntra_filt['Ratings'] != 'nill']['Ratings']

# Create a histogram
plt.figure(figsize=(10, 6))
sns.histplot(filtered_ratings, bins=20, kde=True)

# Set labels and title
plt.xlabel('Rating',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Frequency',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.title('Distribution of Rating Column (excluding no rating)'
          ,fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})

# Display the plot
plt.show()

#regplot

#Correct one

# sns.set(style="darkgrid")
#
# # Filter the DataFrame for 'Indian Wear' category and 'Women' category_by_Gender
# filtered_data = myntra_filt[(myntra_filt['Category'] == 'Indian Wear') & (myntra_filt['category_by_Gender'] == 'Women')]
#
# # Create the lmplot for the filtered data
# #g = sns.lmplot(data=filtered_data, x='OriginalPrice (in Rs)', y='Final Price (in Rs)', height=7)
#
# g = sns.lmplot(data=filtered_data, x='OriginalPrice (in Rs)', y='Final Price (in Rs)', height=7, aspect=1.2,
#                scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
#
# # Adjust layout and show the plot
# plt.tight_layout()
# plt.show()

# Set the aesthetic style of the plots
sns.set(style="darkgrid")

# Filter your DataFrame for the 'Indian Wear' category and 'Women' gender
# Ensure that the column names are correctly specified as they appear in your DataFrame
filtered_data = myntra_filt[(myntra_filt['Category'] == 'Indian Wear') & (myntra_filt['category_by_Gender'] == 'Women')]

# Create the lmplot for the filtered data
# This will automatically include a regression line in the plot
g = sns.lmplot(data=filtered_data, x='OriginalPrice (in Rs)', y='Final Price (in Rs)', height=7, aspect=1.2,
               scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})

# Set the title with the specified font properties
plt.title('lm plot with Regression Line for Women - Indian Wear',
          fontdict={'family': 'serif', 'color': 'blue', 'size': 20})

# Set the x and y labels with the specified font properties
plt.xlabel('Original Price (in Rs)', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
plt.ylabel('Final Price (in Rs)', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()


# # Use regplot to plot the scatter plot with regression line
# sns.regplot(data=myntra_filt, x="OriginalPrice (in Rs)", y="Final Price (in Rs)")
#
# # Set the title with specified font settings
# plt.title('Reg Plot for Original vs Discount Price', fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
#
# # Set X and Y labels with specified font settings
# plt.xlabel('Original Price (in Rs)', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
# plt.ylabel('Discount Price (in Rs)', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
#
# # # Format the tick labels to show numbers with two decimal places
# # ax = plt.gca()  # Get the current Axes instance
# # ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))  # Format x-axis
# # ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.2f}'.format))  # Format y-axis
#
# plt.tight_layout()
# plt.show()

#Boxen Plot

sns.set_style("darkgrid")

myntra_filt['DiscountOffer'] = myntra_filt['DiscountOffer'].astype(str)

# Filter for rows where 'DiscountOffer' is '0% OFF', '1% OFF', or '10% OFF'
filtered_df = myntra_filt[myntra_filt['DiscountOffer'].isin(['0% OFF', '1% OFF', '10% OFF'])]

# Convert 'Ratings' to numeric if it's not already
filtered_df['Ratings'] = pd.to_numeric(filtered_df['Ratings'], errors='coerce')

# Now, create the boxen plot for the filtered data
plt.figure(figsize=(10, 6))
sns.boxenplot(x='category_by_Gender', y='Ratings', data=filtered_df, hue='DiscountOffer')

# Customize the plot
plt.title('Distribution of Ratings by Gender for Specific Discount Offers',
fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Gender Category',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Ratings',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.tight_layout()
plt.grid(True)
plt.legend(title='Discount Offer')
plt.show()

#AreaPlot

    # Group data by category and get the mean of the prices
category_price = myntra_filt.groupby('Category')[['OriginalPrice (in Rs)', 'Final Price (in Rs)']].mean()

    # Sort the DataFrame by one of the price columns for better visualization
category_price = category_price.sort_values(by='OriginalPrice (in Rs)')

    # Create an area plot
category_price.plot(kind='area', stacked=False, figsize=(10, 5))

    # Customize the plot
plt.title('Average Price Before and After Discount by Category',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.ylabel('Price (in Rs)',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.xlabel('Category',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.xticks(rotation=45)  # Rotating category names for better readability
plt.grid(True)
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.show()

#Violin plot

myntra_filt['Ratings'] = pd.to_numeric(myntra_filt['Ratings'], errors='coerce')

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='category_by_Gender', y='Ratings', data=myntra_filt)

# Customize the plot
plt.title('Distribution of Ratings by Gender',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Gender Category',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Ratings',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.grid(True)
plt.tight_layout()
plt.show()

#Joint plot with KDE and scatter representation

# sns.set_style("darkgrid")

# # Convertio 'Ratings' and 'Reviews' to numeric, handling non-numeric 'nill' entries
# myntra_filt['Ratings'] = pd.to_numeric(myntra_filt['Ratings'], errors='coerce')
# myntra_filt['Reviews'] = pd.to_numeric(myntra_filt['Reviews'], errors='coerce')
#
# # Drop NaN values for the plot
# myntra_filt = myntra_filt.dropna(subset=['Ratings', 'Reviews'])
#
# # Create the joint plot
# joint_kde_plot = sns.jointplot(x='Ratings', y='Reviews', data=myntra_filt,kind="kde")
#
# # Customize the plot
# #plt.suptitle('Joint KDE of Ratings and Reviews')
#
# joint_kde_plot.fig.suptitle('Joint KDE of Ratings and Reviews',
#                             color='blue', size=20, fontname='serif')
# joint_kde_plot.set_axis_labels('Ratings', 'Reviews', fontname='serif', color='darkred', size=15)
# plt.tight_layout()
# plt.show()

#This is correct
#Set the style for the plot
#sns.set_style("darkgrid")

# # Convert 'Ratings' and 'Reviews' to numeric, handling non-numeric 'nill' entries
# myntra_filt['Ratings'] = pd.to_numeric(myntra_filt['Ratings'], errors='coerce')
# myntra_filt['Reviews'] = pd.to_numeric(myntra_filt['Reviews'], errors='coerce')
#
# # Drop NaN values for the plot
# myntra_filt = myntra_filt.dropna(subset=['Ratings', 'Reviews'])
#
# # Creation the joint plot with scatter representation
# joint_plot = sns.jointplot(x='Ratings', y='Reviews',data=myntra_filt, hue="category_by_Gender")
#
# # Add a KDE plot on top of the scatter plot
# #joint_plot.plot_joint(sns.kdeplot)
#
# # Customize the plot with specified aesthetics
# joint_plot.fig.suptitle('Joint Plot of Ratings and Reviews', color='blue', size=20, fontname='serif')
# joint_plot.set_axis_labels('Ratings', 'Reviews', fontname='serif', color='darkred', size=15)
#
# # Adjust layout
# plt.tight_layout()
# plt.show()


# Create a jointplot with KDE representation
g = sns.jointplot(data=myntra_filt, x='Ratings', y='Final Price (in Rs)',kind='kde',fill=True)
g.plot_joint(sns.kdeplot, color="b", zorder=5, levels=6)

# Add a scatter plot on top of the KDE
g.plot_joint(sns.scatterplot, color="r", s=30)

plt.title('Joint Plot - Ratings vs Final Price', fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Ratings', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Final Price (in Rs)', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})

plt.tight_layout()
# Display the plot
plt.show()

g = sns.jointplot(data=myntra_filt, x="Ratings", y="Final Price (in Rs)")
g.plot_joint(sns.kdeplot, color="r", zorder=1, levels=6)
g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
plt.tight_layout()
plt.show()
#
# sns.jointplot(data=myntra_filt, x="Ratings", y="Final Price (in Rs)", kind="reg")
# plt.show()



#Rug Plot


# Convert 'Ratings' to numeric, handling non-numeric 'nill' entries
myntra_filt['Ratings'] = pd.to_numeric(myntra_filt['Ratings'], errors='coerce')

# Drop NaN values for the plot
myntra_filt = myntra_filt.dropna(subset=['Ratings'])

puma_men_data = myntra_filt[(myntra_filt['BrandName'] == 'Puma') & (myntra_filt['category_by_Gender'] == 'Men')]



# Create the rug plot

# sns.kdeplot(x='Ratings',data=myntra_filt)
# sns.rugplot(x='Ratings', data=myntra_filt)

sns.scatterplot(data=puma_men_data, x="OriginalPrice (in Rs)", y="Final Price (in Rs)")
sns.rugplot(data=puma_men_data, x="OriginalPrice (in Rs)", y="Final Price (in Rs)",height=-.02, clip_on=False)

# Customize the plot
plt.title('Rug Plot(Puma) - Original vs Final Price', fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('OriginalPrice (in Rs)', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Final Price (in Rs)', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.tight_layout()
plt.show()

#Correct one

# # Filter the DataFrame for 'Puma' brand and 'Men' category
# puma_men_data = myntra_filt[(myntra_filt['BrandName'] == 'Puma') & (myntra_filt['category_by_Gender'] == 'Men')]
#
# # Create the lmplot for the filtered data
# g = sns.rugplot(data=puma_men_data, x='OriginalPrice (in Rs)', y='Final Price (in Rs)', height=7)
#
# # # Customize the plot
# plt.title('Rug Plot(Puma) - Original Price vs Final Price ', fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
# plt.xlabel('OriginalPrice (in Rs)', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
# plt.ylabel('Final Price (in Rs)', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
#
#
# # Adjust layout and show the plot
# plt.tight_layout()
# plt.show()

#ClusterMap

# Convert DiscountOffer to a numeric value
myntra_filt['Discount%'] = myntra_filt['DiscountOffer'].str.extract('(\d+)%').astype(float) / 100

# Convert Ratings and Reviews to numeric values, handling non-numeric data as NaN
myntra_filt['Ratings'] = pd.to_numeric(myntra_filt['Ratings'], errors='coerce')
myntra_filt['Reviews'] = pd.to_numeric(myntra_filt['Reviews'], errors='coerce')

# Now take the first 1000 samples for the subset
subset = myntra_filt.head(1000)

# Selecting features for the ClusterMap from the subset
data_for_clustermap = subset[['Ratings', 'Reviews', 'Final Price (in Rs)', 'OriginalPrice (in Rs)', 'Discount%']]

# Create the ClusterMap
clustermap = sns.clustermap(data_for_clustermap.dropna(), cmap="coolwarm", standard_scale=1)

# Set the title with font properties directly as keyword arguments
clustermap.fig.suptitle("ClusterMap", fontsize=20, color='blue', fontname='serif', va='center')

# Customize the x-axis and y-axis labels using the set_xticklabels and set_yticklabels methods
ax_heatmap = clustermap.ax_heatmap
x_labels = [text.get_text() for text in ax_heatmap.get_xticklabels()]
y_labels = [text.get_text() for text in ax_heatmap.get_yticklabels()]

# ax_heatmap.set_xticklabels(x_labels, fontsize=15, color='darkred', fontname='serif')
# ax_heatmap.set_yticklabels(y_labels, fontsize=15, color='darkred', fontname='serif')


ax_heatmap.set_xticklabels(x_labels, fontsize=15, color='darkred', fontname='serif', rotation=30)
ax_heatmap.set_yticklabels(y_labels, fontsize=15, color='darkred', fontname='serif')


# Adjust the layout of the plot
clustermap.fig.subplots_adjust(top=0.9) # you may need to adjust this value

plt.show()


#Stripplot

filtered_items = myntra_filt[myntra_filt['OriginalPrice (in Rs)']< 2000]

#sns.stripplot(data=filtered_items, x="OriginalPrice (in Rs)", y="Individual_Category_Colour", hue="category_by_Gender", dodge=True)
#sns.stripplot(data=filtered_items, x="OriginalPrice (in Rs)", y="Individual_Category_Colour", dodge=True)
sns.stripplot(data=filtered_items, x="Individual_Category_Colour", y="OriginalPrice (in Rs)",hue="category_by_Gender", dodge=True)

plt.title('stripplot', fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Colour', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('OriginalPrice (in Rs)', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})

plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

#Swarm plot

sns.set_style("darkgrid")
    # Convert 'Ratings' to numeric and 'Reviews' to integer
myntra_filt['Ratings'] = pd.to_numeric(myntra_filt['Ratings'], errors='coerce')
myntra_filt['Reviews'] = pd.to_numeric(myntra_filt['Reviews'], errors='coerce')

    # Drop NaN values
myntra_filt.dropna(subset=['Ratings', 'Reviews'], inplace=True)

    # Calculate the average rating and total number of reviews for each brand
brand_stats = myntra_filt.groupby('BrandName').agg({'Ratings': 'mean', 'Reviews': 'sum'}).reset_index()

    # Choose a threshold or select top N brands, here we take top 10 for demonstration
top_brands = brand_stats.sort_values(by=['Ratings', 'Reviews'], ascending=[False, False]).head(25)

    # Filter the main DataFrame to include only the top brands
top_brands_list = top_brands['BrandName'].tolist()
myntra_filtered = myntra_filt[myntra_filt['BrandName'].isin(top_brands_list)]

# Initialize the matplotlib figure
plt.figure(figsize=(10, 6))

# Create a swarm plot with the filtered data
sns.swarmplot(x='Ratings', y='category_by_Gender', data=myntra_filtered, hue='BrandName')

# Customize the plot
plt.title('Swarm Plot of Ratings by Gender for Top Brands', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 20})
plt.xlabel('Category by Gender', fontname='serif', color='darkred', size=15)
plt.ylabel('Ratings', fontname='serif', color='darkred', size=15)

# Show the legend outside the plot
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)

plt.tight_layout()
plt.show()

# #Approach 2
#
# sample_myntra = myntra_filt.sample(n=500, random_state=1)
# sns.swarmplot(data=sample_myntra, x="Ratings")
# plt.tight_layout()
# plt.show()

#Hexbin plot
#1
# plt.hexbin(myntra_filt['Reviews'], myntra_filt['Ratings'], gridsize=30, cmap='Blues')
# plt.colorbar(label='Number of items')
# plt.xlabel('OriginalPrice (in Rs)')
# plt.ylabel('Ratings')
# plt.title('Hexbin Plot of OriginalPrice (in Rs) vs. Ratings')
# plt.show()
#2
# plt.hexbin(myntra_filt['Ratings'], myntra_filt['Discount Amount'], gridsize=(25,25), cmap=plt.cm.BuGn_r)
# plt.colorbar()
# plt.show()

#3 (Correct)
plt.figure(figsize=(10, 6))
plt.hexbin(myntra_filt['Final Price (in Rs)'], myntra_filt['Ratings'], gridsize=30)
plt.colorbar(label='Count in bin')

plt.title('Hexbin - Final Price After Discount vs Ratings',fontname='serif', color='blue', size=20)
plt.xlabel('Final Price (in Rs)',fontname='serif', color='darkred', size=15)
plt.ylabel('Ratings',fontname='serif', color='darkred', size=15)
plt.tight_layout()
plt.show()

#3D and contour plot

#Correct One

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = myntra_filt['Ratings']
y = myntra_filt['Reviews']
z = myntra_filt['OriginalPrice (in Rs)']  # or 'OriginalPrice (in Rs)'

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('Ratings',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
ax.set_ylabel('Reviews',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
ax.set_zlabel('Original Price (in Rs)',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})

ax.set_title('3D Scatter Plot', fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})



plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(myntra_filt['OriginalPrice (in Rs)'], myntra_filt['Final Price (in Rs)'], myntra_filt['Ratings'])
# ax.set_xlabel('Original Price (in Rs)')
# ax.set_ylabel('Discount Price (in Rs)')
# ax.set_zlabel('Ratings')
# plt.title('3D Plot of Prices and Ratings')
# plt.show()

# x = myntra_filt['Ratings'].to_numpy()
# y = myntra_filt['Reviews'].to_numpy()
# z = myntra_filt['Final Price (in Rs)'].to_numpy()  # or 'OriginalPrice (in Rs)'
#
# # Create grid coordinates
# xi = np.linspace(x.min(), x.max(), 100)
# yi = np.linspace(y.min(), y.max(), 100)
# xi, yi = np.meshgrid(xi, yi)
#
# # Interpolate Z values on grid
# zi = griddata((x, y), z, (xi, yi), method='cubic')
#
# # Plotting
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(xi, yi, zi, cmap='coolwarm', edgecolor='none')
#
# ax.set_xlabel('Ratings')
# ax.set_ylabel('Reviews')
# ax.set_zlabel('Discount Price (in Rs)')
#
# plt.show()

# # contour plot

# sns.set(style="darkgrid")
# plt.figure(figsize=(8, 6))
# ax = sns.kdeplot(data=myntra_filt, x='Ratings', y='Reviews', fill=True)
# ax.set_title("KDE Plot of Ratings vs. Reviews", fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
# ax.set_xlabel('Ratings', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
# ax.set_ylabel('Reviews', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
#
# plt.show()
plt.figure(figsize=(20, 20))
sns.kdeplot(
        data=myntra_filt,
        x="Discount Amount",
        y="Final Price (in Rs)",
        fill=True, # Fill the contour for better visualization
        cmap="viridis" # Use a different colormap for a fresh look
    )

plt.title('Contour Plot - Discount Amount vs Final Price',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Discount Amount (in Rs)',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Final Price (in Rs)',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.tight_layout()
plt.show()

#Multi-variate Kernel Density Estimate

plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=myntra_filt,
    x="Final Price (in Rs)",
    y="Discount Amount",
    hue="category_by_Gender",
    fill=True,
    palette="muted"
)
plt.title('Multivariate Kernel Density Estimate',fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
plt.xlabel('Final Price (in Rs)',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
plt.ylabel('Discount Amount',fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
#plt.grid()
plt.tight_layout()
plt.show()


#Subplot1

top_categories = myntra_filt['Category'].value_counts().head(4)
# Create subplots with 2 rows and 2 columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 30))

# Main title for the figure
fig.suptitle("Pie chart - % of Individual Categories for 4 Categories", fontdict={'family': 'serif', 'color': 'blue', 'size': 20})

# Iterate through top categories and plot pie charts
for i, category in enumerate(top_categories.index):
    top_subcategories = myntra_filt[myntra_filt['Category'] == category]['Individual_category'].value_counts().head(3)
    ax = axes[i // 2, i % 2]

    # Set title for each subplot
    ax.set_title(category, fontdict={'family': 'serif', 'color': 'blue', 'size': 20})

    # Custom labels with specified font properties
    labels = [f"{label}" for label in top_subcategories.index]

    # Plot pie chart
    patches, texts, autotexts = ax.pie(top_subcategories, labels=labels, autopct='%1.1f%%')

    # Customizing the font of the labels in the pie charts
    for text in texts + autotexts:
        text.set_fontsize(15)
        text.set_color('darkred')
        text.set_family('serif')

# Adjust spacing and layout
plt.subplots_adjust(hspace=0.6)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])  # Adjust layout to make room for the main title

plt.show()

#Subplot2

# Getting the value counts for 'Individual_category' and 'SizeOption'
individual_category_counts = myntra_filt['Individual_category'].value_counts()
individual_category_counts = individual_category_counts.head(10)
size_option_counts = myntra_filt['SizeOption'].value_counts()
size_option_counts = size_option_counts.head(10)

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

# Plot for 'Individual_category'
axes[0].bar(individual_category_counts.index, individual_category_counts.values)
axes[0].set_title('Individual Category Counts',fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
axes[0].set_xlabel('Individual Category', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[0].set_ylabel('Count', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[0].tick_params(axis='x', rotation=90)

# Plot for 'SizeOption'
axes[1].bar(size_option_counts.index, size_option_counts.values)
axes[1].set_title('Size Option Counts',fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
axes[1].set_xlabel('Size Option', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[1].set_ylabel('Count', fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[1].tick_params(axis='x', rotation=90)

# Adjusting layout for readability
plt.tight_layout()

plt.show()


#Subplot3

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 16))

# Plot for DiscountOffer
sns.countplot(x='DiscountOffer', hue='category_by_Gender', data=myntra_filt,
              order=myntra_filt['DiscountOffer'].value_counts().index[0:20], ax=axes[0])
axes[0].set_title("Discount Offers by Gender", fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
axes[0].set_xlabel("Discount Offer", fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[0].set_ylabel("Count", fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90, fontsize=14)
for cont in axes[0].containers:
    axes[0].bar_label(cont, fontsize=14, rotation=90, padding=10)

# Plot for SizeOption
sns.countplot(x='SizeOption', hue='category_by_Gender', data=myntra_filt,
              order=myntra_filt['SizeOption'].value_counts().index[0:10], ax=axes[1])
axes[1].set_title("Size Options by Gender", fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
axes[1].set_xlabel("Size Option", fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[1].set_ylabel("Count", fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, fontsize=14)
for cont in axes[1].containers:
    axes[1].bar_label(cont, fontsize=14)

plt.tight_layout()
plt.show()

#Subplot4

avg_price_by_category = myntra_filt.groupby(['Category'])['OriginalPrice (in Rs)'].mean().sort_values(ascending=False)
highest_avg_price_category = avg_price_by_category.index[0]
print("Category with highest average original price:", highest_avg_price_category)
print("Average original price by category: \n", avg_price_by_category)


avg_price_by_gencategory = myntra_filt.groupby(['category_by_Gender'])['OriginalPrice (in Rs)'].mean().sort_values(ascending=False)
highest_avg_price_gencategory = avg_price_by_gencategory.index[0]
print("Gender category with highest average original price:", highest_avg_price_gencategory)
print("Average original price by gender category: \n", avg_price_by_gencategory)


# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

# Plot for average price by category
avg_price_by_category.plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Average Original Price by Category',fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
axes[0].set_xlabel('Category',fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[0].set_ylabel('Average Price (in Rs)',fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[0].tick_params(axis='x', rotation=45)

# Plot for average price by gender category
avg_price_by_gencategory.plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Average Original Price by Gender Category',fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
axes[1].set_xlabel('Gender Category',fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[1].set_ylabel('Average Price (in Rs)',fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


#Subplot5

# # Creating subplots with 1 row and 2 columns
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
#
# # First subplot
# sns.barplot(x='Individual_Category_Colour', y='Final Price (in Rs)', data=myntra_filt, ax=axes[0])
# axes[0].set_title('Average Discount Price by Color')
# axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)  # Rotate x-axis labels if needed
#
# # Second subplot
# sns.barplot(x='Individual_Category_Colour', y='Ratings', data=myntra_filt, ax=axes[1])
# axes[1].set_title('Ratings Distribution by Color')
# axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)  # Rotate x-axis labels if needed
#
# # Adjust layout
# plt.tight_layout()
# plt.show()
#
# # Filter out 'Not specified' entries
# filtered_df_Category_Colour = myntra_filt[myntra_filt['Individual_Category_Colour'] != 'Not specified']
#
# # Count the frequency of each color and filter by count > 1000
# color_counts = filtered_df_Category_Colour['Individual_Category_Colour'].value_counts()
# color_counts = color_counts[color_counts > 1000]
#
# # Create a pie chart
# plt.figure(figsize=(20, 20))
# plt.pie(color_counts, labels=color_counts.index, autopct='%1.1f%%', startangle=140)
# plt.title('Distribution of Colors in Individual Category (Excluding Not Specified)')
# plt.show()


# Filtering out 'Not specified' entries
filtered_df_Category_Colour = myntra_filt[myntra_filt['Individual_Category_Colour'] != 'Not specified']

# Count the frequency of each color and filter by count > 1000
color_counts = filtered_df_Category_Colour['Individual_Category_Colour'].value_counts()
color_counts = color_counts[color_counts > 1700]

#print(color_counts)

# Creating subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20), gridspec_kw={'height_ratios': [1, 2], 'width_ratios': [1, 1]})

# First subplot - Bar plot for Average Discount Price by Color
sns.barplot(x='Individual_Category_Colour', y='Final Price (in Rs)', data=myntra_filt, ax=axes[0, 0], errorbar=None)
axes[0, 0].set_title('Average Discount Price by Color',fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
axes[0,0].set_xlabel('Individual Category Colour',fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[0,0].set_ylabel('DicountPrice (in Rs)',fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

# Second subplot - Bar plot for Ratings Distribution by Color
sns.barplot(x='Individual_Category_Colour', y='Ratings', data=myntra_filt, ax=axes[0, 1], errorbar=None)
axes[0, 1].set_title('Ratings Distribution by Color',fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
axes[0,1].set_xlabel('Individual Category Colour',fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[0,1].set_ylabel('Ratings',fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

# Third subplot - Pie chart
axes[1, 0].remove()  # Remove the extra subplot
axes[1, 1].remove()  # Remove the extra subplot
ax_pie = fig.add_subplot(212)  # Add a new subplot for the pie chart
ax_pie.pie(color_counts, labels=color_counts.index, autopct='%1.1f%%', startangle=140)
ax_pie.set_title('Distribution of Colors in Individual Category (Excluding Not Specified)',fontdict={'family': 'serif', 'color': 'blue', 'size': 20})
for text in ax_pie.texts:
    text.set_family('serif')
    text.set_color('darkred')
    text.set_size(15)
            # Adjust layout and display the plot
plt.tight_layout()
plt.show()












