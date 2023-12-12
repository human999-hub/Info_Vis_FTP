import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import time
from dash import Dash, dcc, html, Input, Output, no_update
import dash
from dash import dcc,html,dash_table,callback
#from dash_core_components import Loading  # For older versions
import warnings
from dash.dcc import Loading
from dash.dependencies import Input,Output, State
import plotly.express as px
import plotly.graph_objs as go
from scipy.stats import kstest, normaltest
import plotly.figure_factory as ff
import scipy.stats as stats
from dash.exceptions import PreventUpdate
from dash import callback_context
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.stats import shapiro



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

pd.options.display.float_format = "{:,.2f}".format

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

url = "/Users/hinjam/Library/CloudStorage/GoogleDrive-iharitha@vt.edu/My Drive/Information Visualization/LAB:Assignments/Myntra Fasion Clothing.csv"

#url = '/kaggle/input/myntra-fashion-dataset/Myntra Fasion Clothing.csv'

df = pd.read_csv(url)

print(df.describe())

#print(df)

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

print(myntra.head())

# Check for missing values using isna() and isnull()

missing_values_na = myntra.isna().sum().sum()
missing_values_null = myntra.isnull().sum().sum()

missing_before_tab0 = myntra.isna().sum().sum()
missing_before_tab01 = myntra.isnull().sum().sum()
duplicate_values_before = myntra.duplicated().sum()

print(f'Missing values using isna(): \n{missing_values_na}')
print(f'Missing values using isnull():\n{missing_values_null}')

myntra['Discount Amount'] = myntra['OriginalPrice (in Rs)']-myntra['Final Price (in Rs)']
#myntra['Discount Amount'].sort_values(ascending=False)

#CleanUp
myntra.dropna(inplace=True)


# Confirming that dataset is clean
missing_values_na_after = myntra.isna().sum().sum()
print(f'Missing values after dropping using isna(): {missing_values_na_after}')

missing_values_null_after = myntra.isnull().sum().sum()
print(f'Missing values after dropping using isnull(): {missing_values_null_after}')

missing_after_tab0 = myntra.isna().sum().sum()
missing_after_tab01 = myntra.isnull().sum().sum()
duplicate_values_after = myntra.duplicated().sum()

print('After Data Cleaning!')
print(myntra.shape[0])
csv_file_path = '/Users/hinjam/new_Dashboard_before_myntra.csv'
myntra.to_csv(csv_file_path, index=False)

# Identify numerical columns for the dropdown
numerical_columns = myntra.select_dtypes(include=np.number).columns.tolist()
print(numerical_columns)

app = dash.Dash('My FTP',suppress_callback_exceptions=True)
app.title = 'Myntra Fashion Analytics'

image_filename = '/Users/hinjam/Desktop/Myntra - logo.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

image_filename1 = '/Users/hinjam/Downloads/Myntra_Logo1.png' # replace with your own image
encoded_image1 = base64.b64encode(open(image_filename1, 'rb').read())

image_filename2 = '/Users/hinjam/Desktop/Dataset Statistics.png' # replace with your own image
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())

# app.layout = (
# html.Div([
# html.Title('Myntra Dashboard'),
#     dcc.Tabs(id='main-tabs', value='tab-1', children=[
#         dcc.Tab(label='Outlier Analysis', value='tab-1'),
#         dcc.Tab(label='Correlation Coefficients', value='tab-2'),
#         dcc.Tab(label='PCA Analysis', value='tab-3'),
#         dcc.Tab(label='Normality Test & Transformation', value='tab-4'),
#         #dcc.Tab(label='Calculations', value='tab-1'),
#         dcc.Tab(label='Figures and Graphs', value='tab-5')
#     ]),
#     html.Div(id='tabs-content'),
#     dcc.Store(id='stored-data'),
# ]),
#
# html.Footer([
#         html.Div([
#             "Copyright © 2023 Myntra Fashion Analytics. All rights reserved.",
#             html.Br(),
#             "Last updated on: " + datetime.now().strftime("%Y-%m-%d"),
#         ], style={'textAlign': 'center', 'color': 'grey'}),
#
# html.Div([
#             dcc.Link('Myntra', href='https://www.myntra.com/'),
#             " | ",
#             dcc.Link('Support', href='https://support.myntra.com/'),
#             " | ",
#             dcc.Link('Contact Us', href='https://www.myntra.com/contactus')
#         ], style={'textAlign': 'center', 'color': 'blue', 'marginTop': '10px'})
#     ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'})
#
#
# )

app.layout = html.Div([
    # Main content
    dcc.Tabs(id='main-tabs', value='tab-1', children=[
        dcc.Tab(label='About Project',value='tab-1'),
        dcc.Tab(label='PreProcessing',value='tab-0'),
        dcc.Tab(label='Outlier Analysis', value='tab-2'),
        dcc.Tab(label='Correlation Coefficients', value='tab-3'),
        dcc.Tab(label='PCA Analysis', value='tab-4'),
        dcc.Tab(label='Normality Test & Transformation', value='tab-5'),
        dcc.Tab(label='Figures and Graphs', value='tab-6')
    ]),
    html.Div(id='tabs-content'),
    dcc.Store(id='stored-data'),

    # Footer
    html.Footer([
        html.Div([
            "Copyright © 2023 Myntra Fashion Analytics. All rights reserved.",
            html.Br(),
            "Last updated on: " + datetime.now().strftime("%Y-%m-%d"),
        ], style={'textAlign': 'center', 'color': 'grey'}),

        html.Div([
            dcc.Link('Myntra', href='https://www.myntra.com/'),
            " | ",
            dcc.Link('Support', href='https://support.myntra.com/'),
            " | ",
            dcc.Link('Contact Us', href='https://www.myntra.com/contactus')
        ], style={'textAlign': 'center', 'color': 'blue', 'marginTop': '10px'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'})
],style={'backgroundColor': '#F5F5F5'})

# Callback for Switching Tabs
@app.callback(
    Output('tabs-content', 'children'),
    [Input('main-tabs', 'value')]
)
def render_tab_content(tab):
    if tab=='tab-1':
        return tab1_layout
    if tab=='tab-0':
        return tab0_layout
    if tab == 'tab-2':
        return tab2_layout
    elif tab == 'tab-3':
        return tab3_layout
    elif tab == 'tab-4':
        return tab4_layout
    elif tab == 'tab-5':
        return tab5_layout
    elif tab == 'tab-6':
        return tab6_layout


tab1_layout = html.Div([
        html.H1("Myntra Fashion Dataset",style={
        'textAlign': 'center',
        'width': '60%',  # Adjust this to control the width of the paragraph
        'margin': 'auto',  # Centers the paragraph container
        'whiteSpace': 'pre-wrap',  # Allows the text to wrap
        'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
    }),
        html.Br(),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image1.decode()), style={'height': '300px','marginLeft': '420px'}),
        html.H3("Dataset Description",style={
        'textAlign': 'center',
        'width': '60%',  # Adjust this to control the width of the paragraph
        'margin': 'auto',  # Centers the paragraph container
        'whiteSpace': 'pre-wrap',  # Allows the text to wrap
        'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
    }),
    html.Br(),
        html.P("The Myntra dataset consists of over 50,000 observations and 13 features. The project investigates the Myntra fashion product dataset to gain insights into customer ratings, pricing and product "
               "characteristics in the online fashion retail sector.",style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),
       html.Br(),

html.P([
        "The dataset can be found at: ",
        html.A("Kaggle Myntra Dataset",
               href="https://www.kaggle.com/datasets/manishmathias/myntra-fashion-dataset",
               target="_blank",  # Opens the link in a new tab
               style={'color': '#007BFF', 'textDecoration': 'none'}  # Style for the hyperlink
        )
    ], style={
        'textAlign': 'center',
        'width': '60%',  # Adjust this to control the width of the paragraph
        'margin': 'auto',  # Centers the paragraph container
        'whiteSpace': 'pre-wrap',  # Allows the text to wrap
        'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
    }),
        html.Br(),
        html.H3("Dataset's Major Features",style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),
html.Br(),
        html.P([html.Span("BrandName: ", style={'fontWeight': 'bold'}),
               "The name of the manufacturer or designer of the product, categorizing items by the company that produced them."
               ],style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),
        html.P([html.Span("Category: ", style={'fontWeight': 'bold'}),
               "A broad classification that groups products into general types such as 'Bottom Wear' or 'Topwear,' reflecting the overarching segment each item belongs to."
               ],style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),

        html.P([html.Span("Individual category: ", style={'fontWeight': 'bold'}),
               "A more detailed categorization that specifies the exact type of product, such as 'jeans' or 'shirts,' providing a finer classification within the broader category."
               ],style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),
        html.P([html.Span("Category by Gender:", style={'fontWeight': 'bold'}),
               "A designation of the intended gender demographic for the product, such as 'Men' or 'Women,' used to sort items based on gender suitability."
               ],style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),
        html.P([html.Span("Discount Offer:", style={'fontWeight': 'bold'}),
               "Promotional information displayed as a percentage or other value, indicating the price reduction from the original cost"
               ],style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),
html.P([html.Span("Size Option: ", style={'fontWeight': 'bold'}),
               "The range of available sizes for the product, represented categorically (e.g., S, M, L, XL) and not intended to be quantitatively analyzed."
               ],style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),

html.P([html.Span("Individual Category Colour: ", style={'fontWeight': 'bold'}),
               "The primary color noted in the product description, acting as a categorical attribute to describe the dominant color of the item"
               ],style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),
html.P([html.Span("Original Price: ", style={'fontWeight': 'bold'}),
               "The Original price of the product without any discounts."
               ],style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),
html.P([html.Span("Final Price: ", style={'fontWeight': 'bold'}),
               "The price of the product after respective discount applied"
               ],style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),
html.P([html.Span("Discount Amount: ", style={'fontWeight': 'bold'}),
               "The Discount Amount applicable to the product"
               ],style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),
html.P([html.Span("Ratings: ", style={'fontWeight': 'bold'}),
               "The average customer rating for the product on a scale, usually from 1 to 5."
               ],style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),
html.P([html.Span("Reviews: ", style={'fontWeight': 'bold'}),
               "The number of reviews the product has received."
               ],style={
               'textAlign': 'center',
               'width': '60%',  # Adjust this to control the width of the paragraph
               'margin': 'auto',  # Centers the paragraph container
               'whiteSpace': 'pre-wrap',  # Allows the text to wrap
               'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
           }),
html.Br(),
 html.H3("Dataset Statistics",style={
        'textAlign': 'center',
        'width': '60%',  # Adjust this to control the width of the paragraph
        'margin': 'auto',  # Centers the paragraph container
        'whiteSpace': 'pre-wrap',  # Allows the text to wrap
        'overflowWrap': 'break-word',  # Breaks the word to the next line if needed
    }),
    html.Br(),
    html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode()),
             style={'height': '300px',
                    'marginLeft': '100px'
                    }),
])

tab0_layout = html.Div([
            html.H3("Dataset Information"),
            dcc.RadioItems(
                id='dataset-info-selection',
                options=[
                    {'label': 'Before preprocessing', 'value': 'before'},
                    {'label': 'After preprocessing', 'value': 'after'}
                ],
                value='before'
            ),
            html.Br(),
            html.P(id='na-values-info'),
            html.P(id='duplicate-values-info'),
            html.Button("Download Cleaned Dataset", id="btn-download-cleaned"),
            dcc.Download(id='download-dataframe-csv-first')
])

tab2_layout = html.Div([
html.Header([
   html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'height': '80px', 'marginRight': '15px'}),
    # Header text component with styling
    html.H1('Myntra Fashion Analytics Dashboard', style={'lineHeight': '80px', 'margin': '0'}),
], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '10px 0'}),

html.H4("Statistical Calculations and Analysis",style={'marginLeft': '450px'}),
html.H4("Note: For cleaned Dataset radio button to work in other tabs, Please remove outliers first",style={'color': 'red'}),
    html.Div([
        html.H4("Outlier Analysis",style={'margin': 'center'}),
        html.Label('Choose a Feature:'),
        dcc.Dropdown(
            id='outlier-column-dropdown',
            options=[{'label': col, 'value': col} for col in numerical_columns],
            value=numerical_columns[0] if numerical_columns else None
        ),
        html.Br(),
        html.Button('Perform Outlier Analysis', id='outlier-analysis-button', n_clicks=0),
        html.Div(id='outlier-analysis-output'),
        html.Img(id='boxplot-image'),
        html.Br(),
        html.Button('Remove Outliers', id='remove-outliers-button', n_clicks=0),
        html.Br(),
        html.Img(id='boxplot-image-removed'),
        html.Div(id='non-outliers-data-table'),
        #dcc.Store(id='stored-data'),
    ])
])

tab4_layout = html.Div([
html.Header([
   html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'height': '80px', 'marginRight': '15px'}),
    # Header text component with styling
    html.H1('Myntra Fashion Analytics Dashboard', style={'lineHeight': '80px', 'margin': '0'}),
], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '10px 0'}),

html.H4("Statistical Calculations and Analysis",style={'marginLeft': '450px'}),
    html.Div([
html.H4("PCA Analysis"),
        dcc.Dropdown(
            id='pca-feature-dropdown',
            options=[{'label': col, 'value': col} for col in numerical_columns],
            value=numerical_columns[:2],  # Default select first two columns
            multi=True
        ),
        html.Button('Perform PCA', id='pca-button', n_clicks=0),
        html.Div(id='pca-output'),
        dcc.Graph(id='pca-scree-plot'),  # Scree plot for PCA
        html.Div(id='pca-output-1'),
    ])
])
# tab4_layout = html.Div([
# html.Header([
#    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'height': '80px', 'marginRight': '15px'}),
#     # Header text component with styling
#     html.H1('Myntra Fashion Analytics Dashboard', style={'lineHeight': '80px', 'margin': '0'}),
# ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '10px 0'}),
#
# html.H4("Statistical Calculations and Analysis"),
# html.Div([
# html.H4("Normality Test"),
# dcc.RadioItems(
#         id='dataset-selection',
#         options=[
#             {'label': 'Original Dataset', 'value': 'original'},
#             {'label': 'Cleaned Dataset', 'value': 'cleaned'}
#         ],
#         value='original'
#     ),
#     dcc.Dropdown(id='feature-dropdown', options=[], value=None),
#     html.Div(id='stored-dataset', style={'display': 'none'}),
# dcc.Checklist(
#         id='test-selection',
#         options=[
#             {'label': 'KS Test', 'value': 'KS'},
#             {'label': 'DAK Test', 'value': 'DAK'}
#         ],
#         value=[],inline = True
#     ),
# dcc.RadioItems(
#         id='transformation-selection',
#         options=[
#             {'label': 'No Transformation', 'value': 'no_transform'},
#             {'label': 'Transformed Test', 'value': 'transform'}
#         ],
#         value='no_transform',
#         inline=True
#     ),
#     dcc.Checklist(
#         id='plot-selection',
#         options=[
#             {'label': 'Include Plots', 'value': 'include_plots'}
#         ],
#         value=[],
#         inline=True
#     ),
# html.Div(id='ks-input-container', style={'display': 'none'}, children=[
#     #html.Textarea('Enter a title for KS Test:'),
#     #dcc.Input(id='input-title-ks', type='text', placeholder='KS Test Title...')
# dcc.Textarea(
#         id='input-title-ks',
#         value='',
#         placeholder='Enter KS Test Title...',
#         style={'width': '20%', 'height': 20}
#     ),
# ]),
# html.Div(id='dak-input-container', style={'display': 'none'}, children=[
#     html.Label('Enter a title for DAK Test:'),
#     dcc.Input(id='input-title-dak', type='text', placeholder='DAK Test Title...')
# ]),
#
# html.Div(id='ks-transform-input-container', style={'display': 'none'}, children=[
#         html.Label('Enter a title for Transformed KS Test:'),
#         dcc.Input(id='input-title-ks-transform', type='text', placeholder='Transformed KS Test Title')
#     ]),
#     html.Div(id='dak-transform-input-container', style={'display': 'none'}, children=[
#         html.Label('Enter a title for Transformed DAK Test:'),
#         dcc.Input(id='input-title-dak-transform', type='text', placeholder='Transformed DAK Test Title')
#     ]),
#
#     html.Button('Submit', id='submit-button', n_clicks=0),
#     html.Div(id='test-results'),
# ])
# ])

tab5_layout = html.Div([
html.Header([
   html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'height': '80px', 'marginRight': '15px'}),
    # Header text component with styling
    html.H1('Myntra Fashion Analytics Dashboard', style={'lineHeight': '80px', 'margin': '0'}),
], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '10px 0'}),

html.H4("Statistical Calculations and Analysis",style={'marginLeft': '450px'}),
html.Div([
html.H4("Normality Test"),
dcc.RadioItems(
        id='dataset-selection',
        options=[
            {'label': 'Original Dataset', 'value': 'original'},
            {'label': 'Cleaned Dataset', 'value': 'cleaned'}
        ],
        value='original'
    ),
    dcc.Dropdown(
        id='feature-dropdown',
        options=[],  # Options will be populated in the callback
        value=None,
        multi=True  # Enable multi-selection
    ),
    html.Div(id='stored-dataset', style={'display': 'none'}),
dcc.Checklist(
        id='test-selection',
        options=[
            {'label': 'Kolmogorov–Smirnov Test(K-S Test)', 'value': 'KS'},
            {'label': 'DAgostinos K-squared test', 'value': 'DAK'},
            {'label': 'Shapiro–Wilk test', 'value': 'SW'},

        ],
        value=[],inline = True
    ),
dcc.RadioItems(
        id='transformation-selection',
        options=[
            {'label': 'No Transformation', 'value': 'no_transform'},
            {'label': 'Transformed Test', 'value': 'transform'}
        ],
        value='no_transform',
        inline=True
    ),
    dcc.Checklist(
        id='plot-selection',
        options=[
            {'label': 'Include Plots', 'value': 'include_plots'}
        ],
        value=[],
        inline=True
    ),
html.Div(id='ks-input-container', style={'display': 'none'}, children=[
    #html.Textarea('Enter a title for KS Test:'),
    #dcc.Input(id='input-title-ks', type='text', placeholder='KS Test Title...')
dcc.Textarea(
        id='input-title-ks',
        value='',
        placeholder='Enter Kolmogorov–Smirnov Test Title...',
        style={'width': '20%', 'height': 20}
    ),
]),
html.Div(id='dak-input-container', style={'display': 'none'}, children=[
    html.Label('Enter title for DAgostinos K-squared Test:'),
    dcc.Input(id='input-title-dak', type='text', placeholder=' DAgostinos K-squared Test Title...')
]),

html.Div(id='sw-input-container', style={'display': 'none'}, children=[
    html.Label('Enter a title for Shapiro-Wilk Test:'),
    dcc.Input(id='input-title-sw', type='text', placeholder='Shapiro-Wilk Test Title...')
]),

html.Div(id='ks-transform-input-container', style={'display': 'none'}, children=[
        html.Label('Enter a title for Transformed KS Test:'),
        dcc.Input(id='input-title-ks-transform', type='text', placeholder='Transformed Kolmogorov–Smirnov Test Title')
    ]),
    html.Div(id='dak-transform-input-container', style={'display': 'none'}, children=[
        html.Label('Enter a title for Transformed DAK Test:'),
        dcc.Input(id='input-title-dak-transform', type='text', placeholder='Transformed DAgostinos K-squared Test Title')
    ]),

html.Div(id='sw-transform-input-container', style={'display': 'none'}, children=[
        html.Label('Enter a title for Transformed Shapiro-Wilk Test:'),
        dcc.Input(id='input-title-sw-transform', type='text', placeholder='Transformed Shapiro-Wilk Test Title')
    ]),
    # html.Div([
    #     dcc.Checklist(
    #         id='store-transformed-data',
    #         options=[
    #             {'label': 'Use Transformed Data for Further Plotting', 'value': 'store'}
    #         ],
    #         value=[]
    #     )
    # ])
html.Div([
    dcc.Checklist(
        id='store-transformed-data',
        options=[
            {'label': 'Use Transformed Data for Further Plotting (Note: If this is used, Please note the scaling of axis,data changes for respective columns selected)', 'value': 'store'}
        ],
        value=[]
    )
], id='store-transformed-data-container', style={'display': 'none'})  # Initially hidden
    ,
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='test-results'),
])
])

tab3_layout = html.Div([
html.Header([
   html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'height': '80px', 'marginRight': '15px'}),
    # Header text component with styling
    html.H1('Myntra Fashion Analytics Dashboard', style={'lineHeight': '80px', 'margin': '0'}),
], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'padding': '10px 0'}),

html.H4("Statistical Calculations and Analysis",style={'marginLeft': '450px'}),
html.Div([
    html.H4("Correlation Coefficients Table"),
    dash_table.DataTable(id='correlation-table')
]),
html.Div([

    html.H4("Heatmap & Scatter Plot Matrix"),
    dcc.RadioItems(
        id='heatmap-dataset-selection',
        options=[
            {'label': 'Original Dataset', 'value': 'original'},
            {'label': 'Cleaned Dataset', 'value': 'cleaned'}
        ],
        value='original'
    ),
    dcc.Graph(id='correlation-heatmap'),
    dcc.Graph(id='scatter-plot-matrix'),


dcc.Tooltip(id='scatter-tooltip')
])
])

# Callback for Outlier Analysis
# @app.callback(
#     [Output('outlier-analysis-output', 'children'),
#      Output('non-outliers-data-table', 'children'),
#      Output('stored-data', 'data'),
#      Output('boxplot-image', 'src')],
#     [Input('outlier-analysis-button', 'n_clicks'),
#      Input('remove-outliers-button', 'n_clicks')],
#     [State('outlier-column-dropdown', 'value')]
# )
# def perform_outlier_analysis(outlier_n_clicks, remove_n_clicks, selected_column):
#     ctx = dash.callback_context
#
#     if not ctx.triggered or not selected_column:
#         return html.Div(), html.Div(), None
#
#     button_id = ctx.triggered[0]['prop_id'].split('.')[0]
#
#     column_data = myntra[selected_column]
#     Q1, Q3 = column_data.quantile([0.25, 0.75])
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
#
#     if button_id == 'outlier-analysis-button':
#         fig, ax = plt.subplots()
#         ax.boxplot(myntra[selected_column])
#         ax.set_title('Boxplot of ' + selected_column)
#         ax.set_ylabel('Value')
#         ax.set_xticks([1], [selected_column])
#
#         # Highlighting the outlier thresholds
#         ax.axhline(y=lower_bound, color='r', linestyle='--', label=f'Lower Bound = {lower_bound:.2f}')
#         ax.axhline(y=upper_bound, color='g', linestyle='--', label=f'Upper Bound = {upper_bound:.2f}')
#         ax.legend()
#         ax.grid()
#
#         # Convert the plot to a PNG image in memory
#         buffer = BytesIO()
#         fig.savefig(buffer, format='png')
#         buffer.seek(0)
#         image_png = buffer.getvalue()
#         buffer.close()
#
#         # Encode the PNG image to base64
#         encoded_image = base64.b64encode(image_png)
#         encoded_image = 'data:image/png;base64,' + encoded_image.decode()
#         # Display the analysis results
#         return html.Div([
#             html.P(f"Selected Column: {selected_column}"),
#             html.P(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}"),
#             html.P(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}"),
#             html.P(f"Number of Outliers: {len(outliers)}"),
#             html.P(f"Max Outlier Value: {outliers.max()}" if len(outliers) > 0 else ""),
#             html.P(f"Min Outlier Value: {outliers.min()}" if len(outliers) > 0 else "")
#         ]), html.Div(), None,encoded_image
#
#     elif button_id == 'remove-outliers-button':
#         # Remove outliers from the DataFrame and create a new DataFrame
#         myntra_no_outliers = myntra[~myntra[selected_column].isin(outliers)]
#         stored_data = myntra_no_outliers.to_json(date_format='iso', orient='split')
#         # print("Stored Data:", stored_data)
#         #return stored_data
#
#         table = dash_table.DataTable(
#             data=myntra_no_outliers.head().to_dict('records'),
#             columns=[{'name': i, 'id': i} for i in myntra_no_outliers.columns]
#         )
#         return html.Div([
#             html.P(f"Outliers removed from column: {selected_column}"),
#             html.P(f"Remaining rows in dataset: {len(myntra_no_outliers)}")
#         ]), table, stored_data
#
#     return html.Div(), html.Div(), None,None

# Callback to update dataset information
@app.callback(
    [Output('na-values-info', 'children'),
     Output('duplicate-values-info', 'children')],
    [Input('dataset-info-selection', 'value')]
)
def update_dataset_info(selected_option):
    if selected_option == 'before':
        na_info = f"Number of null or NA values in dataset (before preprocessing): {missing_before_tab0}"
        null_info = f"Number of null values in dataset (before preprocessing): {missing_before_tab01}"
    else:
        na_info = f"Number of NA values in dataset (after preprocessing): {missing_after_tab0}"
        null_info = f"Number of null values in dataset (after preprocessing): {missing_after_tab01}"
    return na_info, null_info

# Callback for download button
@app.callback(
    Output("download-dataframe-csv-first", "data"),
    Input("btn-download-cleaned", "n_clicks"),
    prevent_initial_call=True
)
def download_cleaned_data(n_clicks):
    if n_clicks:
        return dcc.send_data_frame(myntra.to_csv, "cleaned_myntra_dataset.csv", index=False)



@app.callback(
    [Output('outlier-analysis-output', 'children'),
     Output('non-outliers-data-table', 'children'),
     Output('stored-data', 'data'),
     Output('boxplot-image', 'src'),
     Output('boxplot-image-removed', 'src')],
    [Input('outlier-analysis-button', 'n_clicks'),
     Input('remove-outliers-button', 'n_clicks')],
    [State('outlier-column-dropdown', 'value')]
)
def perform_outlier_analysis(outlier_n_clicks, remove_n_clicks, selected_column):
    ctx = dash.callback_context

    if not ctx.triggered or not selected_column:
        return html.Div(), html.Div(), dash.no_update, dash.no_update, dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    column_data = myntra[selected_column]
    Q1, Q3 = column_data.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]

    if button_id == 'outlier-analysis-button':
        fig, ax = plt.subplots()
        ax.boxplot(column_data)
        ax.set_title('Boxplot of ' + selected_column, fontdict={'fontsize': 20, 'color': 'blue', 'fontname': 'serif'})
        ax.set_ylabel('Value', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
        ax.set_xticks([1], [selected_column], fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
        ax.axhline(y=lower_bound, color='r', linestyle='--', label=f'Lower Bound = {lower_bound:.2f}')
        ax.axhline(y=upper_bound, color='g', linestyle='--', label=f'Upper Bound = {upper_bound:.2f}')
        ax.legend()
        ax.grid()
        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        encoded_image = base64.b64encode(image_png)
        encoded_image = 'data:image/png;base64,' + encoded_image.decode()

        return html.Div([
            html.P(f"Selected Column: {selected_column}"),
            html.P(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}"),
            html.P(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}"),
            html.P(f"Number of Outliers: {len(outliers)}"),
            html.P(f"Max Outlier Value: {outliers.max()}" if len(outliers) > 0 else ""),
            html.P(f"Min Outlier Value: {outliers.min()}" if len(outliers) > 0 else "")
        ]), dash.no_update, dash.no_update, encoded_image, dash.no_update

    elif button_id == 'remove-outliers-button':
        myntra_no_outliers = myntra[~myntra[selected_column].isin(outliers)]
        stored_data = myntra_no_outliers.to_json(date_format='iso', orient='split')

        table = dash_table.DataTable(
            data=myntra_no_outliers.head().to_dict('records'),
            columns=[{'name': i, 'id': i} for i in myntra_no_outliers.columns]
        )

        fig, ax = plt.subplots()
        ax.boxplot(myntra_no_outliers[selected_column])
        ax.set_title('Boxplot of ' + selected_column + 'Post Removal', fontdict={'fontsize': 15, 'color': 'blue', 'fontname': 'serif'})
        ax.set_ylabel('Value', fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
        ax.set_xticks([1], [selected_column], fontdict={'fontsize': 15, 'color': 'darkred', 'fontname': 'serif'})
        # ax.axhline(y=lower_bound, color='r', linestyle='--', label='Lower Bound')
        # ax.axhline(y=upper_bound, color='g', linestyle='--', label='Upper Bound')
        # ax.legend()
        ax.grid()
        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        encoded_image_removed = base64.b64encode(image_png)
        encoded_image_removed = 'data:image/png;base64,' + encoded_image_removed.decode()

        return (dash.no_update, table, stored_data, dash.no_update, encoded_image_removed)

    return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update)




# Corrected PCA Analysis Callback
@app.callback(
    [Output('pca-output', 'children'),
     Output('pca-scree-plot', 'figure'),
     Output('pca-output-1', 'children')],
    [Input('pca-button', 'n_clicks')],
    [State('pca-feature-dropdown', 'value'),
     State('stored-data', 'data')]
)
def perform_pca(n_clicks, selected_features,stored_data):
    # print(f"Button Clicks: {n_clicks}, Selected Features: {selected_features}, Stored Data: {stored_data}")
    if n_clicks > 0 and stored_data and selected_features:
        # Standardize the features
        df_no_outliers = pd.read_json(stored_data, orient='split')
        df_selected = df_no_outliers[selected_features]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_selected)

        # Perform PCA
        pca = PCA()
        pca.fit(scaled_data)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

        # Singular values and condition number for the original feature space
        U, S, VT = np.linalg.svd(scaled_data, full_matrices=False)

        # Scree plot
        # scree_plot = px.line(
        #     x=np.arange(1, len(explained_variance_ratio) + 1),
        #     y=cumulative_variance,
        #     labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'},
        #     title='Scree Plot for PCA'
        # )
        scree_plot = px.line(
            x=np.arange(1, len(explained_variance_ratio) + 1),
            y=cumulative_variance * 100,  # Multiply by 100 to convert to percentage
            labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance (%)'},
            title='Scree Plot for PCA'
        )

        # Adding vertical line
        scree_plot.add_vline(x=n_components_95, line_dash="dash", line_color="red",
                             annotation_text=f"Optimum Number of Features: {n_components_95}",
                             annotation_position="bottom right")

        # Adding horizontal line
        scree_plot.add_hline(y=95, line_dash="dash", line_color="black",
                             annotation_text="95% Explained Variance",
                             annotation_position="bottom right")

        scree_plot.update_layout(
            xaxis_title='Number of Components',
            yaxis_title='Cumulative Explained Variance (%)',
            showlegend=False,
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightPink'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightBlue'),
            margin=dict(l=40, r=40, t=40, b=30),
            font=dict(family='serif', size=20),
            title_font_color='blue',
            xaxis_title_font=dict(family='serif', size=15, color='darkred'),
            yaxis_title_font=dict(family='serif', size=15, color='darkred')

        )

        # scree_plot.show()

        # PCA for reduced feature space
        pca_reduced = PCA(n_components=0.95, svd_solver='full')
        reduced_data = pca_reduced.fit_transform(scaled_data)
        U_reduced, S_reduced, VT_reduced = np.linalg.svd(reduced_data, full_matrices=False)
        reduced_explained_variance_ratio = pca_reduced.explained_variance_ratio_

        # Prepare the PCA output
        pca_output = html.Div([
            html.P(f"Singular Values for Original Feature Space: {S}"),
            html.P(f"Condition Number for Original Feature Space: {np.linalg.cond(scaled_data)}"),
            html.P(f"Explained Variance Ratio: {explained_variance_ratio}"),
            html.P(f"Cumulative Explained Variance: {cumulative_variance}"),
            html.P(f"Number of features to be removed: {scaled_data.shape[1] - n_components_95}"),
        ])

        # Output for reduced feature space
        pca_output_1 = html.Div([
            html.P(f"Explained Variance Ratio (Reduced Feature Space): {reduced_explained_variance_ratio}"),
            html.P(f"Singular Values for Reduced Feature Space: {S_reduced}"),
            html.P(f"Condition Number for Reduced Feature Space: {np.linalg.cond(reduced_data)}"),
        ])

        return pca_output, scree_plot, pca_output_1
    else:
        return html.Div(["No data or features selected for PCA."]), {}, html.Div()

    return html.Div(), {}, html.Div()

@app.callback(
    [Output('ks-input-container', 'style'),
     Output('dak-input-container', 'style'),
     Output('sw-input-container', 'style'),
     Output('ks-transform-input-container', 'style'),
     Output('dak-transform-input-container', 'style'),
     Output('sw-transform-input-container', 'style')],
    [Input('test-selection', 'value'),
     Input('transformation-selection', 'value')]
)
def toggle_input_fields(selected_tests, transformation_selection):
    show_ks = {'display': 'block'} if 'KS' in selected_tests else {'display': 'none'}
    show_dak = {'display': 'block'} if 'DAK' in selected_tests else {'display': 'none'}
    show_sw = {'display': 'block'} if 'SW' in selected_tests else {'display': 'none'}
    show_ks_transform = show_ks if transformation_selection == 'transform' else {'display': 'none'}
    show_dak_transform = show_dak if transformation_selection == 'transform' else {'display': 'none'}
    show_sw_transform = show_sw if transformation_selection == 'transform' else {'display': 'none'}
    return show_ks, show_dak,show_sw, show_ks_transform, show_dak_transform,show_sw_transform

@app.callback(
    Output('store-transformed-data-container', 'style'),  # Assuming this is the container div ID
    [Input('transformation-selection', 'value')]
)
def toggle_store_transformed_data_option(transformation_selection):
    if transformation_selection == 'transform':
        return {'display': 'block'}  # Show the checkbox
    else:
        return {'display': 'none'}  # Hide the checkbox


normality_status = ""
def ks_test(x, title, feature_name):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    if p > 0.01:
        normality_status = f'{title} dataset looks Normal with 99% accuracy'
    else:
        normality_status = f'{title} dataset looks Not Normal with 99% accuracy'
    #return html.Div(f'K-S test: {title} : statistics= {stats:.2f}, p-value = {p:.2f} (for {feature_name})')
    return html.Div([
        html.P(f'Kolmogorov–Smirnov test: {title}'),
        html.P(f'Statistics = {stats:.2f}, p-value = {p:.2f} (for {feature_name})'),
        html.P(normality_status)
    ])

def da_k_squared_test(x, title, feature_name):
    stats, p = normaltest(x)
    if p > 0.01:
        normality_status = f'{title} dataset looks Normal with 99% accuracy'
    else:
        normality_status = f'{title} dataset looks Not Normal with 99% accuracy'
    #return html.Div(f'DA K-squared test: {title} : statistics= {stats:.2f}, p-value = {p:.2f} (for {feature_name})')
    return html.Div([
        html.P(f'DAgostinos K-squared test: {title}'),
        html.P(f'Statistics = {stats:.2f}, p-value = {p:.2f} (for {feature_name})'),
        html.P(normality_status)
    ])


def shapiro_test(x,title,feature_name):
    stats, p = shapiro(x)
    print('=' * 50)
    if p > 0.01:
        normality_status = f'{title} dataset looks Normal with 99% accuracy'
    else:
        normality_status = f'{title} dataset looks Not Normal with 99% accuracy'
   # return html.Div(f'Shapiro test: {title} : statistics= {stats:.2f}, p-value = {p:.2f} (for {feature_name})')
    return html.Div([
        html.P(f'Shapiro test: {title}'),
        html.P(f'Statistics = {stats:.2f}, p-value = {p:.2f} (for {feature_name})'),
        html.P(normality_status)
    ])





@app.callback(
    Output('feature-dropdown', 'options'),
    [Input('dataset-selection', 'value')],
[State('stored-data', 'data')]

)
def update_features(selected_dataset,stored_data):
    if selected_dataset == 'original':
        dataset = myntra
    elif stored_data:  # Ensure stored_data_json is not None
        dataset = pd.read_json(stored_data, orient='split')
    else:
        return []  # Return empty list if no data is available

    # Filter only numerical columns
    numerical_cols = dataset.select_dtypes(include=[np.number]).columns
    return [{'label': col, 'value': col} for col in numerical_cols]
def generate_plots(dataset, selected_features, include_transformed):
    plots = []
    for feature in selected_features:
        feature_data = dataset[feature]
        # Original Data Plots
        plots.append(create_histogram(feature_data, f'Histogram of {feature}'))
        plots.append(create_qq_plot(feature_data, f'QQ-plot of {feature}'))

        # Transformed Data Plots
        if include_transformed and np.all(feature_data > 0):
            transformed_data, _ = stats.boxcox(feature_data)
            plots.append(create_histogram(transformed_data, f'Histogram of Transformed {feature}'))
            plots.append(create_qq_plot(transformed_data, f'QQ-plot of Transformed {feature}'))

    return plots

def create_histogram(data, title):
    hist = go.Figure(data=[go.Histogram(x=data, nbinsx=30)])
    hist.update_layout(
        title=title,
        xaxis_title_text='Value',
        yaxis_title_text='Count',
        bargap=0.2,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        font=dict(family='serif', size=20),
        title_font_color='blue',
        xaxis_title_font=dict(family='serif', size=15, color='darkred'),
        yaxis_title_font=dict(family='serif', size=15, color='darkred')

    )
    return dcc.Graph(figure=hist)

def create_qq_plot(data, title):
    qq_plot = go.Figure(data=[go.Scatter(x=np.sort(stats.norm.rvs(size=len(data))),
                                         y=np.sort(data), mode='markers')])
    qq_plot.update_layout(
        title=title,
        xaxis_title_text='Theoretical Quantiles',
        yaxis_title_text='Ordered Values',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        font=dict(family='serif', size=20),
        title_font_color='blue',
        xaxis_title_font=dict(family='serif', size=15, color='darkred'),
        yaxis_title_font=dict(family='serif', size=15, color='darkred')
    )
    return dcc.Graph(figure=qq_plot)


@app.callback(
    [Output('test-results', 'children'),
            Output('stored-data', 'data', allow_duplicate=True)],
    [Input('submit-button', 'n_clicks'),
     #Input('input-title-ks','children'),
     ],
    [State('feature-dropdown', 'value'),
     State('test-selection', 'value'),
     State('input-title-ks', 'value'),
     State('input-title-dak', 'value'),
State('input-title-sw', 'value'),
     State('input-title-ks-transform', 'value'),
     State('input-title-dak-transform', 'value'),
State('input-title-sw-transform', 'value'),
     State('transformation-selection', 'value'),
     State('plot-selection', 'value'),
     State('dataset-selection', 'value'),
     #State('use-transformed-data', 'value'),
     #State('numerical-columns-dropdown', 'value'),
     State('stored-data', 'data'),
     State('store-transformed-data', 'value')],
     prevent_initial_call=True
)
def perform_tests(n_clicks, selected_features, selected_tests, ks_title, dak_title,sw_title, ks_transform_title, dak_transform_title,sw_transform_title, transform_selection, plot_selection, selected_dataset, stored_data, store_transformed):
    if not n_clicks or not selected_features:
        raise PreventUpdate

    if selected_dataset == 'original':
        dataset = myntra  # Replace 'myntra' with your actual dataset variable
    elif stored_data:
        dataset = pd.read_json(stored_data, orient='split')
    else:
        return html.Div('No dataset available.')

    results = []
    transformation_applied = False
    for feature_name in selected_features:
        if feature_name not in dataset:
            continue  # Skip if the feature is not in the dataset

        feature_data = dataset[feature_name]

        if 'KS' in selected_tests:
            results.append(ks_test(feature_data, ks_title, feature_name))

        if 'DAK' in selected_tests:
            results.append(da_k_squared_test(feature_data, dak_title, feature_name))

        if 'SW' in selected_tests:
            results.append(shapiro_test(feature_data, sw_title, feature_name))

        if transform_selection == 'transform':
            if np.all(feature_data > 0):
                transformed_data, best_lambda = stats.boxcox(feature_data)
                if 'KS' in selected_tests:
                    results.append(ks_test(transformed_data, ks_transform_title, feature_name))
                if 'DAK' in selected_tests:
                    results.append(da_k_squared_test(transformed_data, dak_transform_title, feature_name))
                if 'SW' in selected_tests:
                    results.append(shapiro_test(transformed_data, sw_transform_title, feature_name))
            else:
                results.append(html.Div(f'Cannot apply Box-Cox transformation on {feature_name} due to non-positive values'))

    # if 'include_plots' in plot_selection:
    #     # Assuming generate_plots function handles multiple features correctly
    #     results.extend(generate_plots(dataset[selected_features]))
    include_transformed_plots = 'include_plots' in plot_selection and transform_selection == 'transform'
    if include_transformed_plots:
        results.extend(generate_plots(dataset, selected_features, True))

    #transformed_dataset = dataset.copy()
    # Store transformed data if opted
    # Apply transformations
    # if stored_data:
    #     stored_data = pd.read_json(stored_data, orient='split')
    # else:
    #     stored_data = pd.DataFrame()  # or some default value
    if 'store' in store_transformed and transform_selection == 'transform':
        for col in selected_features:
            #if col in stored_data.columns and np.all(stored_data[col] > 0):
            if np.all(dataset[col] > 0):
                transformed_data, _ = stats.boxcox(dataset[col])
                dataset[col] = transformed_data
                transformation_applied = True
            else:
                results.append(html.Div(f'Cannot apply Box-Cox transformation on {col} due to non-positive values'))
    if transformation_applied:
        success_message = "Transformed data stored successfully."
        results.append(html.Div(success_message))
        print("Transformed Data:", dataset.head())
        if selected_dataset == 'cleaned' and transformation_applied:
            stored_data = dataset.to_json(orient='split')
        print("Transformed Data:", pd.read_json(stored_data, orient='split').head())
    return results,stored_data



@app.callback(
    [Output('correlation-heatmap', 'figure'),
     Output('scatter-plot-matrix', 'figure')],
    [
     Input('heatmap-dataset-selection', 'value'),
     State('stored-data', 'data')
     ]
)
def update_heatmap_scatter_matrix(selected_dataset,stored_data_json):
    if selected_dataset == 'original':
        dataset = myntra
    elif stored_data_json:
        dataset = pd.read_json(stored_data_json, orient='split')
        print("corr_stored_data")
        print(dataset)
    else:
        return {}, {}

    # Select only numeric columns for correlation
    numeric_dataset = dataset.select_dtypes(include=[np.number])

    # Check if the numeric dataset is empty
    if numeric_dataset.empty:
        return {}, {}

    # Calculate the Pearson correlation coefficients
    correlation_matrix = numeric_dataset.corr()

    # Create the heatmap
    # heatmap_figure = ff.create_annotated_heatmap(
    #     z=correlation_matrix.to_numpy(),
    #     x=list(correlation_matrix.columns),
    #     y=list(correlation_matrix.index),
    #
    #     colorscale='RdBu',
    #     annotation_text=correlation_matrix.round(2).astype(str).values,
    #     showscale=True
    # )
    # Update the column names for x-axis labels if needed
    new_x_labels = ['Final Price (in Rs)' if 'FinalPrice' in col else col for col in correlation_matrix.columns]

    # Update the index names for y-axis labels if needed
    new_y_labels = ['OriginalPrice (in Rs)' if 'OriginalPrice' in idx else idx for idx in correlation_matrix.index]
    heatmap_figure = ff.create_annotated_heatmap(
        z=correlation_matrix.to_numpy(),
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index),
        colorscale='RdBu',
        annotation_text=correlation_matrix.round(2).astype(str).values,
        showscale=True
    )
    heatmap_figure.update_layout(
        xaxis=dict(
            tickfont=dict(color='darkred',size=15,           # Sets the font size
            family='serif')  # Change 'red' to your desired color
        ),
    yaxis = dict(
        tickfont=dict(color='darkred', size=15,  # Sets the font size
                      family='serif')  # Change 'red' to your desired color
    )
    )



    # Create the scatter plot matrix
    # scatter_matrix_figure = px.scatter_matrix(
    #     dataset,
    #     dimensions=correlation_matrix.columns,
    #     title="Scatter Plot Matrix"
    # )
    # scatter_matrix_figure.update_layout(
    #     height=1200,
    #     width=1200,
    #     xaxis_tickangle=-45,
    #     yaxis_tickangle=-45,
    #     #xaxis_title_font=dict(family='serif', size=15, color='darkred'),
    #     #yaxis_title_font=dict(family='serif', size=15, color='darkred'),
    #     font=dict(family='serif', size=20),
    #     title_font_color='blue',
    #
    # )
    scatter_matrix_figure = px.scatter_matrix(
        dataset,
        dimensions=correlation_matrix.columns,
        title="Scatter Plot Matrix"
    )

    scatter_matrix_figure.update_layout(
        height=1200,
        width=1200,
        font=dict(family='serif', size=20),
        title_font_color='blue',
    )
    scatter_matrix_figure.update_layout(
        xaxis=dict(
            title='FinalPrice (Rs)',
            title_font=dict(family='serif', size=15, color='darkred')
        ),
        xaxis2=dict(
            title='OriginalPrice (in Rs)',
            title_font=dict(family='serif', size=15, color='darkred')
        ),
        xaxis3=dict(
            title='Ratings',
            title_font=dict(family='serif', size=15, color='darkred')
        ),
        xaxis4=dict(
            title='Reviews',
            title_font=dict(family='serif', size=15, color='darkred')
        ),
        xaxis5=dict(
            title='Discount Amount',
            title_font=dict(family='serif', size=15, color='darkred')
        ),
        xaxis6=dict(
            title='Discount Depth (%)',
            title_font=dict(family='serif', size=15, color='darkred')
        ),
    )
    scatter_matrix_figure.update_layout(
        yaxis=dict(
            title='FinalPrice (Rs)',
            title_font=dict(family='serif', size=15, color='darkred')
        ),
        yaxis2=dict(
            title='OriginalPrice (in Rs)',
            title_font=dict(family='serif', size=15, color='darkred')
        ),
        yaxis3=dict(
            title='Ratings',
            title_font=dict(family='serif', size=15, color='darkred')
        ),
        yaxis4=dict(
            title='Reviews',
            title_font=dict(family='serif', size=15, color='darkred')
        ),
        yaxis5=dict(
            title='Discount Amount',
            title_font=dict(family='serif', size=15, color='darkred')
        ),
        yaxis6=dict(
            title='Discount Depth (%)',
            title_font=dict(family='serif', size=15, color='darkred')
        ),
    )

    return heatmap_figure, scatter_matrix_figure

@app.callback(
    Output('scatter-tooltip', 'children'),
    Output('scatter-tooltip', 'show'),
    Output('scatter-tooltip', 'bbox'),
    [
    Input('scatter-plot-matrix', 'hoverData'),
    Input('heatmap-dataset-selection', 'value'),
    Input('stored-data', 'data')],
    [State('scatter-plot-matrix', 'figure'),
     ]
)
# def update_tooltip(hoverData, selected_dataset,stofigure):
#     if hoverData is None:
#         return '', False, {}
#
#     # Extract the index of the hovered point
#     point_index = hoverData['points'][0]['pointIndex']
#     hovered_point_data = myntra.iloc[point_index]
#
#     # Create the tooltip content
#     tooltip_content = html.Div([
#         html.P(f"Brand: {hovered_point_data['BrandName']}"),
#         html.P(f"Category: {hovered_point_data['Category']}"),
#         html.P(f"Original Price: {hovered_point_data['OriginalPrice (in Rs)']}"),
#         html.P(f"Discount Price: {hovered_point_data['Final Price (in Rs)']}"),
#         html.P(f"Discount Amount: {hovered_point_data['Discount Amount']}"),
#         html.P(f"Ratings: {hovered_point_data['Ratings']}")
#     ])
#
#     # Get the bounding box to position the tooltip
#     bbox = hoverData['points'][0]['bbox']
#
#     return tooltip_content, True, bbox

def update_tooltip(hoverData, selected_dataset, stored_data_json,scatter_plot_figure):
    if hoverData is None:
        return '', False, {}

    #Select the dataset based on user choice
    if selected_dataset == 'original':
        dataset = myntra
    elif stored_data_json:
        dataset = pd.read_json(stored_data_json, orient='split')
    else:
        return '', False, {}

    # Extract the index of the hovered point
    point_index = hoverData['points'][0]['pointIndex']
    hovered_point_data = dataset.iloc[point_index]

    # Create the tooltip content
    tooltip_content = html.Div([
        html.P(f"Brand: {hovered_point_data['BrandName']}"),
        html.P(f"Category: {hovered_point_data['Category']}"),
        html.P(f"Original Price: {hovered_point_data['OriginalPrice (in Rs)']}"),
        html.P(f"Discount Price: {hovered_point_data['Final Price (in Rs)']}"),
        html.P(f"Discount Amount: {hovered_point_data['Discount Amount']}"),
        html.P(f"Ratings: {hovered_point_data['Ratings']}")
    ])

    # Get the bounding box to position the tooltip
    bbox = hoverData['points'][0]['bbox']

    return tooltip_content, True, bbox

@app.callback(
    Output('correlation-table', 'data'),
    Output('correlation-table', 'columns'),
    [Input('heatmap-dataset-selection', 'value'),
     State('stored-data', 'data')]
)
def update_table(selected_dataset, stored_data_json):
    if selected_dataset == 'original':
        dataset = myntra  # Assuming 'myntra' is your original dataset
    elif stored_data_json:
        dataset = pd.read_json(stored_data_json, orient='split')
    else:
        return [], []

    # Remove 'Discount Depth (%)' column from the dataset
    if 'Discount Depth (%)' in dataset.columns:
        dataset = dataset.drop(columns=['Discount Depth (%)'])

    # Select only numeric columns for correlation
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    correlation_matrix = dataset[numeric_cols].corr()

    # Round the correlation matrix to two decimal places
    correlation_matrix_rounded = correlation_matrix.round(2).reset_index()

    # Define the table columns
    columns = [{"name": i, "id": i} for i in correlation_matrix_rounded.columns]

    # Convert the rounded correlation matrix to a dictionary for the DataTable
    data = correlation_matrix_rounded.to_dict('records')

    return data, columns



#Layout2

#grouped_bar_data = myntra.groupby('Category')['Discount Amount'].mean().reset_index()
#grouped_tab_data = myntra.groupby(['Category', 'Individual_category'])['Final Price (in Rs)'].mean().reset_index()
#print(grouped_tab_data)
# List of selected brands
selected_brands = [
    "VIMAL", "Go Colors", "max", "Dollar", "Qurvii", "Nanda Silk Mills",
    "Manyavar", "SOUNDARYA", "Indian Terrain", "URBANIC", "Being Human",
    "RAMRAJ COTTON", "Fabindia", "Ethnix by Raymond", "CAVALLO by Linen Club",
    "Linen Club", "Oxemberg", "Armaan Ethnic", "Hastakala", "KLM Fashion Mall",
    "Women Republic", "taruni", "PINKVILLE JAIPUR", "Zanani INDIA"
]

# myntra['Ratings'] = pd.to_numeric(myntra['Ratings'], errors='coerce')
# grouped_avg_ratings = myntra.groupby(['BrandName', 'category_by_Gender'])['Ratings'].mean().reset_index()

# Bar plot figure
# bar_fig = px.bar(grouped_bar_data, x='Category', y='Discount Amount', title='Average Discount Amount for Each Category')
# bar_fig.update_layout(title_x=0.5)

# Calculate the count of SizeOptions for each Category
# category_size_counts = myntra.groupby('Category')['SizeOption'].apply(lambda x: ', '.join(x)).reset_index()
# category_size_counts['SizeOption'] = category_size_counts['SizeOption'].str.split(', ').apply(len)

# # Define bins
# bins = [0, 10, 20, 50, 100, 500, 1000, 5000, 10000]
#
# # Create a new column 'Review_Bins' to store the binned data
# myntra['Reviews'] = pd.to_numeric(myntra['Reviews'], errors='coerce')
# myntra['Review_Bins'] = pd.cut(myntra['Reviews'], bins, labels=["0-10", "11-20", "21-50", "51-100", "101-500", "501-1000", "1001-5000", "5001-10000"])
#
# # Create a countplot
# countplot_data = myntra['Review_Bins'].value_counts().reset_index()
# countplot_data.columns = ['Review_Bins', 'Count']

# Filter the dataset for rows where OriginalPrice is below 5,000
filtered_myntra = myntra[myntra["OriginalPrice (in Rs)"] < 5000]

# Assuming 'df' is your DataFrame containing the dataset
myntra['Discount Depth (%)'] = ((myntra['OriginalPrice (in Rs)'] - myntra['Final Price (in Rs)']) / myntra['OriginalPrice (in Rs)']) * 100
print('Discount Depth (%) - myntra')
print(myntra.head())


tab6_layout = html.Div([

#Download

    # html.Div(className='row', children='Myntra Fashion',
    #          style={'textAlign': 'center', 'color': 'blue', 'fontSize': 30}),
    html.Button("Download Data", id="btn_download",
                style={
                    'position': 'absolute',
                    'top': '10px',  # Adjust the distance from the top as needed
                    'right': '10px',  # Adjust the distance from the right as needed
                    'zIndex': '1000'  # Ensure the button is above other elements
                }),

    # Hidden component that will trigger the download
    dcc.Download(id="download-dataframe-csv"),

    #Image & Title

html.Header([

html.Figure([
    html.Img(
        src='data:image/png;base64,{}'.format(encoded_image.decode()),
        style={'height': '80px', 'marginBottom': '2px'}
    ),
    html.H1(
        'Myntra Fashion Analytics Dashboard',
        style={'lineHeight': '30px', 'margin': '0', 'textAlign': 'center'}
    ),
    html.Figcaption(
        'Visualizing Fashion Data and Trends',
        style={'textAlign': 'center'}
    )
], style={
    'display': 'flex',
    'flexDirection': 'column',
    'alignItems': 'center',
    'justifyContent': 'center',
    #'padding': '5px 0'
})
    ]),

    html.Br(),
    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            # dash_table.DataTable(data=grouped_tab_data.to_dict('records'), page_size=11,page_action='native',
            #                      style_table={'overflowX': 'auto'})
# dash_table.DataTable(
#     data=grouped_tab_data.to_dict('records'),
#     columns=[{'name': i, 'id': i} for i in grouped_tab_data.columns],
#     page_size=11,
#     page_action='native'
# )

#stored_data
html.H4("Table: Average Final Price for Each Individual Category",style={'marginLeft':'400px'}),
html.Br(),
dash_table.DataTable(id='data-table-in-tab2',page_size=11,page_action='native'),

        ]),
        html.Br(),
 #Lineplot

#         html.Div(className='row', children=[
#             dcc.Dropdown(
#                 options=[{'label': category, 'value': category} for category in myntra['Category'].unique()],
#                 value=myntra['Category'].iloc[0],
#                 id='category-dropdown'
#             )
#         ]),
#         # html.Div(className='six columns', children=[
#         #     dcc.Graph(figure={}, id='line-chart-final')
#         # ])
#
# html.Div(className='six columns', children=[
#     Loading(
#         id="loading-1",
#         type="default",
#         children=dcc.Graph(figure={}, id='line-chart-final')
#     )
# ])
#     ]),

#stored_data

html.Div(className='row', children=[
        dcc.Dropdown(
            id='category-dropdown',
            # The options will be set dynamically via callback
        )
    ]),

html.Div(className='six columns', children=[
    Loading(
        id="loading-1",
        type="default",
        children=dcc.Graph(figure={}, id='line-chart-final')
    )
])
]),
#Bar plot

    # html.Div(className='row', children=[
    #     html.Div(className='twelve columns', children=[
    #         dcc.Graph(figure=bar_fig, id='bar-chart-category')
    #     ])
    # ]),

html.Div([
        dcc.RangeSlider(
            id='price-range-slider',
            # min=myntra['Final Price (in Rs)'].min(),
            # max=myntra['Final Price (in Rs)'].max(),
min=0,
max=5000,

            step=500,
marks={i: str(i) for i in range(0, 5001, 500)},  # Marks at intervals of 5000
            value=[0, 5000],  # Initial selected range covering the full extent
        ),
    ], style={'padding': '20px'}),

    html.Div([
        dcc.Graph(id='category-bar-chart')
    ]),


# Bar plot - stacked
#     html.Div(className='row', children=[
#         dcc.Checklist(
#             options=[{'label': brand, 'value': brand} for brand in selected_brands],
#             value=selected_brands,
#             id='brand-checklist',
#             #inline=True
#         )
#     ]),

html.Div(className='row', children=[
    dcc.Checklist(
        options=[{'label': brand, 'value': brand} for brand in selected_brands],
        value=selected_brands,
        id='brand-checklist',
        style={
            'display': 'grid',
            'grid-template-columns': '1fr 1fr 1fr 1fr 1fr',  # Creates three equal columns
            'grid-gap': '10px'  # Optional: Adjust the space between the checkboxes
        }
    )
]),

    html.Div(className='row', children=[
        # For Brand Gender Bar Chart
        html.Div(className='six columns', children=[
            dcc.Graph(figure={}, id='brand-gender-bar-chart')
        ]),

#Bar plot - Group

        # For Brand Gender Rating Chart
        html.Div(className='six columns', children=[
            dcc.Graph(figure={}, id='brand-gender-rating-chart')
        ])
    ]),
    # Countplot
    #RadioItems to select a category
    # dcc.RadioItems(
    #     id='category-radio',
    #     options=[{'label': category, 'value': category} for category in myntra['Category'].unique()],
    #     value=myntra['Category'].iloc[0],
    #     labelStyle={'display': 'block'}
    # ),
    #dcc.Graph(id='countplot-graph'),

#Original

# html.H4("Count of Products by Review Bins", style={'textAlign': 'center'}),
#     dcc.Graph(
#         figure=px.bar(countplot_data, x='Review_Bins', y='Count', labels={'Review_Bins': 'Review Bins', 'Count': 'Number of Products'}),
#     ),

#stored_data

#html.H4("Count of Products by Review Bins", style={'textAlign': 'center'}),
dcc.Graph(
    id='review-countplot',
    # The figure will be set via the callback
),

#Pie Chart

html.H4("Size Options for Categories and Gender"),

html.Div([
    html.Div([
        html.H4("Select a Category:"),
        dcc.RadioItems(
            id='category-radio1',
            #options=[{'label': cat, 'value': cat} for cat in myntra['Category'].unique()],
            #value='Bottom Wear',
            labelStyle={'display': 'block'}
        ),
    ], className='six columns'),

    html.Div([
        html.H4("Select a Gender:"),
        dcc.RadioItems(
            id='gender-radio',
            #options=[{'label': gender, 'value': gender} for gender in myntra['category_by_Gender'].unique()],
            value='Men',
            labelStyle={'display': 'block'}
        ),
    ], className='six columns'),
], className='row'),

dcc.Graph(id='size-pie-chart'),

    #html.H4("Distribution of Original Prices Below 5,000"),
    html.Div([
    dcc.Graph(
        id='original-price-distplot',
        config={'displayModeBar': False},  # Hide the plotly toolbar
    ),
#dcc.Store(id='stored-data'),
]),

    #Line Plot Slider Test

# html.Div([
#     dcc.Markdown("### Price Range Filter"),
#     dcc.RangeSlider(
#         id='price-range-slider',
#         min=0,  # Set the minimum value of the range
#         max=5000,  # Set the maximum value of the range
#         step=100,  # Set the step increment
#         marks={0: '0', 1000: '1000', 2000: '2000', 3000: '3000', 4000: '4000', 5000: '5000'},  # Custom marks
#         value=[0, 5000],  # Initial selected range
#     ),
# ]),
#
# html.Div(className='six columns', children=[
#     Loading(
#         id="loading-1",
#         type="default",
#         children=dcc.Graph(id='line-chart-final')  # Use the same ID as in the callback
#     )
# ])

# html.Div([
#     html.H4("Interactive Scatter Plot Matrix"),
#     html.Label("Select a range of Ratings:"),
#     dcc.Slider(
#         id='ratings-slider',
#         min=myntra['Ratings'].min(),
#         max=myntra['Ratings'].max(),
#         value=myntra['Ratings'].max(),
#         step=0.1,
#         marks={str(rating): str(rating) for rating in range(int(myntra['Ratings'].min()), int(myntra['Ratings'].max())+1)},
#         included=False
#     ),
#     dcc.Graph(id='scatter-plot-matrix-slider')
# ]),

#Original Slider Code

# html.Div([
#     html.H4("Interactive Scatter Plot Matrix"),
#     html.Label("Select a range of Discount Depth (%):"),
#     dcc.Slider(
#         id='discount-depth-slider',
#         min=myntra['Discount Depth (%)'].min(),
#         max=myntra['Discount Depth (%)'].max(),
#         value=myntra['Discount Depth (%)'].min(),
#         step=0.1,
#         marks={str(int(i)): str(int(i)) for i in range(int(myntra['Discount Depth (%)'].min()), int(myntra['Discount Depth (%)'].max())+1, 5)},  # Mark every 5%
#         included=False
#     ),
#     dcc.Graph(id='scatter-plot-matrix-slider1')
# ])

html.Div([
    html.H4("Interactive Scatter Plot Matrix"),
    html.Label("Select a range of Discount Depth (%):"),
    dcc.Slider(
        id='discount-depth-slider',
        min=0,  # Temporary placeholder values
        max=100,  # Temporary placeholder values
        value=20,  # Temporary placeholder value
        step=0.1,
        marks={str(i): str(i) for i in range(0, 101, 5)},  # Mark every 5%
        included=False
    ),
    dcc.Graph(id='scatter-plot-matrix-slider1')
]),

# html.Div([
#
#     # Plot placeholders
#    # dcc.Graph(id='avg-discount-price-color'),
#    # dcc.Graph(id='ratings-distribution-color'),
#
#     dcc.Graph(id='combined-chart'),
#     dcc.Graph(id='color-distribution-pie'),
#
#  ])
html.Div([
        html.H4("Combined Chart: Average Discount Amount and Ratings Distribution by Colour"),
        html.P("Select Distribution for Ratings:",style={'margin-left':'800px'}),
        dcc.RadioItems(
            id='distribution-type',
            options=[
                {'label': 'Bar', 'value': 'bar'},
                {'label': 'Box', 'value': 'box'},
                {'label': 'Violin', 'value': 'violin'}
            ],
            value='bar',  # Default value
            inline=True,style={'margin-left':'800px'}
        ),
        dcc.Graph(id='combined-chart'),

        dcc.Graph(id='color-distribution-pie')
    ]),
])

#Download callback

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_download", "n_clicks"),
    #Input('stored-data', 'data'),
    prevent_initial_call=True,
)
def download_data(n_clicks):
    if n_clicks:
        # Convert DataFrame to CSV string
        #reused_df = pd.read_json(stored_data, orient='split')
        return dcc.send_data_frame(myntra.to_csv, "myntra_dataset.csv", index=False)

#stored_data

@app.callback(
    Output('data-table-in-tab2', 'data'),  # This matches the ID of the DataTable
    Input('stored-data', 'data')  # This is triggered when `stored-data` changes
)
def update_table_from_stored_data(stored_data):
    if not stored_data:
        raise PreventUpdate
    reused_df = pd.read_json(stored_data, orient='split')
    grouped_data = reused_df.groupby(['Category', 'Individual_category'])['Final Price (in Rs)'].mean().reset_index()
    grouped_data['Final Price (in Rs)'] = grouped_data['Final Price (in Rs)'].round(2)
    return grouped_data.to_dict('records')


#Line Plot callback

# @app.callback(
#     Output(component_id='line-chart-final', component_property='figure'),
#     Input(component_id='category-dropdown', component_property='value')
# )
# def update_graph(selected_category):
#     print("Callback triggered")  # Debug print
#     time.sleep(2)
#     # Filter by the selected category
#     filtered_data = myntra[myntra['Category'] == selected_category]
#
#     # Group by 'Individual_category' and calculate the average discount price
#     grouped_data = filtered_data.groupby('Individual_category')['Final Price (in Rs)'].mean().reset_index()
#
#     # Create the figure using plotly express
#     fig = px.line(grouped_data, x='Individual_category', y='Final Price (in Rs)',
#                   title=f'Average Price After Discount for {selected_category}')
#     fig.update_layout(title_x=0.5)
#     return fig

#Stored_data

@app.callback(
    Output('category-dropdown', 'options'),
    Input('stored-data', 'data')
)
def set_dropdown_options(stored_data):
    if not stored_data:
        raise PreventUpdate
    df = pd.read_json(stored_data, orient='split')
    return [{'label': category, 'value': category} for category in df['Category'].unique()]

@app.callback(
    Output('line-chart-final', 'figure'),
    [Input('category-dropdown', 'value'),
     Input('stored-data', 'data')]
)
def update_line_chart(selected_category, stored_data):
    print('Line plot callback triggered')
    time.sleep(2)
    if not stored_data or not selected_category:
        raise PreventUpdate
    df = pd.read_json(stored_data, orient='split')

    # Filter the DataFrame based on the selected category
    filtered_data = df[df['Category'] == selected_category]

    # Group by 'Individual_category' and calculate the average discount price
    grouped_data = filtered_data.groupby('Individual_category')['Final Price (in Rs)'].mean().reset_index()

    fig = px.line(
        grouped_data,
        x='Individual_category',  # Replace with your actual column name
        y='Final Price (in Rs)',  # Replace with your actual column name
        title=f'Line Plot for {selected_category}'
    )
    fig.update_layout(title_x=0.5,
                      font=dict(family='serif', size=20),
                      title_font_color='blue',
                      xaxis_title_font=dict(family='serif', size=15, color='darkred'),
                      yaxis_title_font=dict(family='serif', size=15, color='darkred'))
    return fig

#Line Plot - callback Test
# desired_categories = ['jeans', 'tshirts', 'trackpants','trousers','kurtas','kurta-sets','tops','shirts','sarees','sweatshirts']
# @app.callback(
#     Output('line-chart-final', 'figure'),  # Replace 'line-chart-final' with your actual chart component ID
#     Input('price-range-slider', 'value')
# )
# def update_line_chart(selected_price_range):
#     min_price, max_price = selected_price_range
#
#     # Filter your data based on the selected price range
#     filtered_data = myntra[(myntra['Discount Amount'] >= min_price) &
#                            (myntra['Discount Amount'] <= max_price)]
#
#     # Further filter to include only the desired categories
#     filtered_data = filtered_data[filtered_data['Individual_category'].isin(desired_categories)]
#
#     # Calculate the average Discount Amount for each category
#     avg_price_data = filtered_data.groupby('Individual_category')['Final Price (in Rs)'].mean().reset_index()
#
#     # Create a new line chart based on the average price data
#     fig = px.line(avg_price_data, x='Individual_category', y='Final Price (in Rs)',
#                   title='Average Price After Discount by Category')
#
#     return fig

#Barchart callback

# @app.callback(
#     Output('category-bar-chart', 'figure'),
#     Input('price-range-slider', 'value')
# )
# def update_graph(selected_price_range):
#     min_price, max_price = selected_price_range
#     filtered_df = myntra[(myntra['Final Price (in Rs)'] >= min_price) & (myntra['Final Price (in Rs)'] <= max_price)]
#     print('Bar Filtered df')
#     print(filtered_df)
#
#     avg_price_data = filtered_df.groupby('Individual_category')['Final Price (in Rs)'].mean().reset_index()
#     print('Avg Individual Category price')
#     print(avg_price_data)
#     #print(filtered_df)
#     fig = px.bar(avg_price_data, x='Individual_category', y='Final Price (in Rs)', title='Average Discount Amount per Category')
#     return fig

#stored_data
@app.callback(
    Output('category-bar-chart', 'figure'),  # Change this to the ID of your bar chart component in tab2
    Input('stored-data', 'data'),  # Again, assuming 'stored-data' is the ID of your dcc.Store component
    Input('price-range-slider', 'value')  # Assuming this is the input that drives changes in your bar chart
)
def update_graph(stored_data_json, selected_price_range):
    if stored_data_json is None:
        raise PreventUpdate

    min_price, max_price = selected_price_range
    stored_data_df = pd.read_json(stored_data_json, orient='split')
    filtered_df = stored_data_df[(stored_data_df['Final Price (in Rs)'] >= min_price) &
                                 (stored_data_df['Final Price (in Rs)'] <= max_price)]
    avg_price_data = filtered_df.groupby('Individual_category')['Final Price (in Rs)'].mean().reset_index()
    #avg_price_data['Final Price (in Rs)'] = avg_price_data['Final Price (in Rs)'].round(2)

    # Resetting the default template
    import plotly.io as pio
    pio.templates.default = "plotly"  # or "simple_white"


    # Create the figure using Plotly Express or Graph Objects
    fig = px.bar(avg_price_data, x='Individual_category', y='Final Price (in Rs)',
                 title='Average Discount Amount per Category',template=None)
    fig.update_layout(title_x=0.5,
                      font=dict(family='serif', size=20),
                      title_font_color='blue',
                      xaxis_title_font=dict(family='serif', size=15, color='darkred'),
                      yaxis_title_font=dict(family='serif', size=15, color='darkred'))
    return fig

#Bar Plot (stack) callback


# @app.callback(
#     Output(component_id='brand-gender-bar-chart', component_property='figure'),
#     Input(component_id='brand-checklist', component_property='value')
# )
# def update_brand_gender_chart(selected_brands_list):
#     filtered_df = myntra[myntra['BrandName'].isin(selected_brands_list)]
#     grouped_data = filtered_df.groupby(['BrandName', 'category_by_Gender']).size().reset_index(name='Count')
#     fig = px.bar(grouped_data, x='BrandName', y='Count', color='category_by_Gender',
#                  title="Brand Preference by Gender for Selected Brands",
#                  labels={'Count': 'Number of Products'},
#                  color_discrete_map={'Men': 'blue', 'Women': 'pink'},
#                  height=600,
#                  width=1000)
#     fig.update_layout(title_x=0.5)
#     return fig

#stored_data

@app.callback(
    Output('brand-gender-bar-chart', 'figure'),
    [Input('brand-checklist', 'value'),
     Input('stored-data', 'data')]  # Adding stored-data as an input
)
def update_brand_gender_chart(selected_brands_list, stored_data_json):
    if not stored_data_json:
        raise PreventUpdate

    # Convert stored JSON data to DataFrame
    df = pd.read_json(stored_data_json, orient='split')

    # Filtering the DataFrame based on selected brands
    filtered_df = df[df['BrandName'].isin(selected_brands_list)]

    # Grouping data
    grouped_data = filtered_df.groupby(['BrandName', 'category_by_Gender']).size().reset_index(name='Count')

    # Resetting the default template
    import plotly.io as pio
    pio.templates.default = "plotly"  # or "simple_white"

    # Creating the bar chart
    fig = px.bar(grouped_data, x='BrandName', y='Count', color='category_by_Gender',
                 title="Brand Preference by Gender for Selected Brands",
                 labels={'Count': 'Number of Products'},
                 color_discrete_map={'Men': 'blue', 'Women': 'pink'},
                 height=600, width=1000,
                 template=None

                 )
    fig.update_layout(title_x=0.5,
                      font=dict(family='serif', size=20),
                      title_font_color='blue',
                      xaxis_title_font=dict(family='serif', size=15, color='darkred'),
                      yaxis_title_font=dict(family='serif', size=15, color='darkred'))

    return fig

#Bar Plot (Group) callback

# @app.callback(
#     Output(component_id='brand-gender-rating-chart', component_property='figure'),
#     Input(component_id='brand-checklist', component_property='value')
# )
# def update_brand_gender_rating_chart(selected_brands_list):
#     # Filter data based on selected brands
#     filtered_df = grouped_avg_ratings[grouped_avg_ratings['BrandName'].isin(selected_brands_list)]
#
#     # Plot
#     fig = px.bar(filtered_df,
#                  x='BrandName',
#                  y='Ratings',
#                  color='category_by_Gender',
#                  barmode='group',
#                  title='Average Ratings by Brand and Gender for Selected Brands')
#     fig.update_layout(title_x=0.5)
#     return fig

#stored_data

@app.callback(
    Output('brand-gender-rating-chart', 'figure'),
    [Input('brand-checklist', 'value'),
     Input('stored-data', 'data')]  # Adding stored-data as an input
)
def update_brand_gender_rating_chart(selected_brands_list, stored_data_json):
    if not stored_data_json:
        raise PreventUpdate

    # Convert stored JSON data to DataFrame
    df = pd.read_json(stored_data_json, orient='split')

    # Convert 'Ratings' to numeric in case it's not
    df['Ratings'] = pd.to_numeric(df['Ratings'], errors='coerce')

    # Grouping by BrandName and category_by_Gender and calculating mean ratings
    grouped_avg_ratings = df.groupby(['BrandName', 'category_by_Gender'])['Ratings'].mean().reset_index()

    # Filter data based on selected brands
    filtered_df = grouped_avg_ratings[grouped_avg_ratings['BrandName'].isin(selected_brands_list)]
    import plotly.io as pio
    pio.templates.default = "plotly"

    # Creating the bar chart
    fig = px.bar(filtered_df,
                 x='BrandName',
                 y='Ratings',
                 color='category_by_Gender',
                 barmode='group',
                 title='Average Ratings by Brand and Gender for Selected Brands',
                 template=None)
    fig.update_layout(title_x=0.5,
                      font=dict(family='serif', size=20),
                      title_font_color='blue',
                      xaxis_title_font=dict(family='serif', size=15, color='darkred'),
                      yaxis_title_font=dict(family='serif', size=15, color='darkred'))

    return fig

#Count Plot callback

# @app.callback(
#     Output('countplot-graph', 'figure')
    #[Input('category-radio', 'value')]
#)
# def update_countplot(selected_category):
#     filtered_data = myntra[myntra['Category'] == selected_category]
#
#     # Aggregate the data for the bar chart
#     aggregated_data = filtered_data['Individual_category'].value_counts().reset_index()
#     aggregated_data.columns = ['Individual_category', 'Count']
#
#     # Use Plotly Express to create a bar chart (countplot)
#     fig = px.bar(
#         aggregated_data,
#         x='Individual_category',
#         y='Count',
#         title='Count of products available in each category'
#     )
#
#     fig.update_traces(marker_color='blue')  # Customize bar color
#     fig.update_layout(
#         title={'text': 'Count of products available in each category', 'x': 0.5, 'xanchor': 'center'},
#         title_font={'family': 'serif', 'color': 'blue', 'size': 20},
#         xaxis_title_text='Category',
#         xaxis_title_font={'family': 'serif', 'color': 'darkred', 'size': 15},
#         yaxis_title_text='Count',
#         yaxis_title_font={'family': 'serif', 'color': 'darkred', 'size': 15},
#         xaxis_tickangle=-90
#     )
#
#     return fig

#stored_data

@app.callback(
    Output('review-countplot', 'figure'),  # Update this ID to match your countplot Graph component in the layout
    Input('stored-data', 'data')
)
def update_review_countplot(stored_data_json):
    if not stored_data_json:
        raise PreventUpdate

    df = pd.read_json(stored_data_json, orient='split')

    # Define bins
    bins = [0, 10, 20, 50, 100, 500, 1000]
    # Convert 'Reviews' to numeric and bin the data
    #df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    df['Review_Bins'] = pd.cut(df['Reviews'], bins, labels=["0-10", "11-20", "21-50", "51-100", "101-500", "501-1000"],include_lowest=True)
    df['Review_Bins'] = df['Review_Bins'].astype(str)
    # Create countplot data
    countplot_data = df['Review_Bins'].value_counts().reset_index()
    print(countplot_data)
    countplot_data.columns = ['Review_Bins', 'Count']
    import plotly.io as pio
    pio.templates.default = "plotly"
    # Create the countplot figure
    fig = px.bar(countplot_data, x='Review_Bins', y='Count', labels={'Review_Bins': 'Review Bins', 'Count': 'Number of Products'},title = 'Count of Products by Review Bins',template=None)
    fig.update_layout(title_x=0.5,
                      font=dict(family='serif', size=20),
                      title_font_color='blue',
                      xaxis_title_font=dict(family='serif', size=15, color='darkred'),
                      yaxis_title_font=dict(family='serif', size=15, color='darkred'))
    return fig

#Pie chart

# @app.callback(
#     Output('size-pie-chart', 'figure'),
#     [Input('category-radio1', 'value'),
#      Input('gender-radio', 'value')]
# )
# def update_pie_chart(selected_category, selected_gender):
#     filtered_data = myntra[
#         (myntra['Category'] == selected_category) & (myntra['category_by_Gender'] == selected_gender)
#         ]
#     filtered_counts = filtered_data['SizeOption'].value_counts()
#
#     size_options_to_plot = ['28, 30, 32, 34, 36', 'Onesize']
#
#     # Ensure the desired size options are present in filtered_counts
#     for size_option in size_options_to_plot:
#         if size_option not in filtered_counts:
#             filtered_counts[size_option] = 0
#
#     # Now, filter only the desired size options to plot
#     filtered_counts = filtered_counts[size_options_to_plot]
#
#     fig = px.pie(
#         names=filtered_counts.index,
#         values=filtered_counts.values,
#         title=f'Size Options for {selected_category} ({selected_gender})'
#     )
#     return fig

#stored_data

#Pie Chart callback

@app.callback(
    [Output('category-radio1', 'options'),
     Output('gender-radio', 'options')],
    [Input('stored-data', 'data')]
)
def set_radio_options(stored_data_json):
    if not stored_data_json:
        raise PreventUpdate

    df = pd.read_json(stored_data_json, orient='split')

    # Generating options for the category and gender radio items
    category_options = [{'label': cat, 'value': cat} for cat in df['Category'].unique()]
    gender_options = [{'label': gender, 'value': gender} for gender in df['category_by_Gender'].unique()]

    return category_options, gender_options
# @app.callback(
#     Output('size-pie-chart', 'figure'),
#     [Input('category-radio1', 'value'),
#      Input('gender-radio', 'value'),
#      Input('stored-data', 'data')]
# )
# def update_pie_chart(selected_category, selected_gender, stored_data_json):
#     if not stored_data_json:
#         raise PreventUpdate
#
#     df = pd.read_json(stored_data_json, orient='split')
#
#     # Filter data based on the selected category and gender
#     filtered_data = df[
#         (df['Category'] == selected_category) & (df['category_by_Gender'] == selected_gender)
#     ]
#     filtered_counts = filtered_data['SizeOption'].value_counts()
#
#     # Define size options to plot and ensure they are present in the counts
#     size_options_to_plot = ['28, 30, 32, 34, 36', 'Onesize']
#     for size_option in size_options_to_plot:
#         if size_option not in filtered_counts:
#             filtered_counts[size_option] = 0
#
#     # Filter only the desired size options
#     filtered_counts = filtered_counts[size_options_to_plot]
#
#     # Create the pie chart figure
#     fig = px.pie(
#         names=filtered_counts.index,
#         values=filtered_counts.values,
#         title=f'Size Options for {selected_category} ({selected_gender})'
#     )
#     return fig

@app.callback(
    Output('size-pie-chart', 'figure'),
    [Input('category-radio1', 'value'),
     Input('gender-radio', 'value'),
     Input('stored-data', 'data')]
)
def update_pie_chart(selected_category, selected_gender, stored_data_json):
    if not stored_data_json:
        raise PreventUpdate

    df = pd.read_json(stored_data_json, orient='split')

    # Filter data based on the selected category and gender
    filtered_data = df[
        (df['Category'] == selected_category) & (df['category_by_Gender'] == selected_gender)
    ]
    filtered_counts = filtered_data['SizeOption'].value_counts()

    # Define size options to plot and ensure they are present in the counts
    size_options_to_plot = ['28, 30, 32, 34, 36', 'Onesize']
    for size_option in size_options_to_plot:
        if size_option not in filtered_counts:
            filtered_counts[size_option] = 0

    # Filter only the desired size options
    filtered_counts = filtered_counts[size_options_to_plot]
    import plotly.io as pio
    pio.templates.default = "plotly"

    # Create the pie chart figure
    fig = px.pie(
        names=filtered_counts.index,
        values=filtered_counts.values,
        title=f'Size Options for {selected_category} ({selected_gender})',template=None
    )

    # Update the layout for title
    fig.update_layout(
        title={
            'text': f"Size Options for {selected_category} ({selected_gender})",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'family': 'serif', 'color': 'blue', 'size': 20}
        }
    )

    return fig


#Displot

# Callback to update the displot
# @app.callback(
#     Output('original-price-distplot', 'figure'),
#     Input('original-price-distplot', 'relayoutData')
# )
# def update_original_price_hist(relayoutData):
#     # Create the histogram using plotly express
#     fig = px.histogram(filtered_myntra, x="OriginalPrice (in Rs)",
#                        nbins=50,
#                        #histnorm='count',
#                        title='Distribution of Original Prices Below 5,000')
#
#     return fig

# @app.callback(
#     Output('original-price-distplot', 'figure'),
#     Input('stored-data', 'data')
# )
# def update_original_price_hist(stored_data_json):
#     #print(stored_data_json)
#     if not stored_data_json:
#         raise PreventUpdate
#
#     df = pd.read_json(stored_data_json, orient='split')
#     print(df.shape[0])
#     print(df.head())
#     # Filter the dataset for rows where OriginalPrice is below 5,000
#     filtered_myntra = df[df["OriginalPrice (in Rs)"] < 5000]
#
#     # Create the histogram using plotly express
#     fig = px.histogram(filtered_myntra, x="OriginalPrice (in Rs)",
#                        nbins=50,
#                        title='Distribution of Original Prices Below 5,000')
#
#     fig.update_layout(
#         title={
#             'text': "Distribution of Original Prices Below 5,000",
#             'y': 0.9,
#             'x': 0.5,
#             'xanchor': 'center',
#             'yanchor': 'top',
#             'font': {'family': 'serif', 'color': 'blue', 'size': 20}
#         },
#         xaxis_title="Original Price (in Rs)",
#         yaxis_title="Count",
#         xaxis_title_font={'family': 'serif', 'color': 'darkred', 'size': 15},
#         yaxis_title_font={'family': 'serif', 'color': 'darkred', 'size': 15}
#     )
#
#     return fig

@app.callback(
    Output('original-price-distplot', 'figure'),
    Input('stored-data', 'data')
)
def update_original_price_hist(stored_data_json):
    if not stored_data_json:
        raise PreventUpdate

    df = pd.read_json(stored_data_json, orient='split')

    # Ensure the 'OriginalPrice (in Rs)' column is numeric
    df['OriginalPrice (in Rs)'] = pd.to_numeric(df['OriginalPrice (in Rs)'], errors='coerce')

    # Filter the dataset for rows where OriginalPrice is below 5,000
    filtered_prices = df[df["OriginalPrice (in Rs)"] < 5000]["OriginalPrice (in Rs)"].dropna()
    # import plotly.io as pio
    # pio.templates.default = "plotly"
    # Create the distribution plot
    fig = ff.create_distplot([filtered_prices], ['Original Price'],bin_size=50)

    # Update layout of the figure
    fig.update_layout(
        title={
            'text': "Distribution of Original Prices Below 5,000",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'family': 'serif', 'color': 'blue', 'size': 20}
        },
        xaxis_title="Original Price (in Rs)",
        yaxis_title="Density",
        xaxis_title_font={'family': 'serif', 'color': 'darkred', 'size': 15},
        yaxis_title_font={'family': 'serif', 'color': 'darkred', 'size': 15},
        bargap=0.1  # You can adjust the gap between bars if needed
    )

    return fig



# @app.callback(
#     [Output('rating-slider', 'min'),
#      Output('rating-slider', 'max'),
#      Output('rating-slider', 'marks'),
#      Output('rating-slider', 'value')],
#     [Input('stored-data', 'data')]
# )
# def update_slider(stored_data_json):
#     if not stored_data_json:
#         raise dash.exceptions.PreventUpdate
#
#     df = pd.read_json(stored_data_json, orient='split')
#
#     min_rating = df['Ratings'].min()
#     max_rating = df['Ratings'].max()
#     marks = {i: f'{i}' for i in range(int(min_rating), int(max_rating) + 1)}
#     value = min_rating
#
#     return min_rating, max_rating, marks, value
#
# @app.callback(
#     Output('scatter-plot-matrix-slider', 'figure'),
#     [Input('ratings-slider', 'value')]
# )
# def update_scatter_matrix(selected_rating):
#     # Filter the DataFrame based on the selected rating value
#     filtered_df = myntra[myntra['Ratings'] >= selected_rating]
#
#     # Update the scatter plot matrix figure with the filtered data
#     fig = px.scatter_matrix(
#         filtered_df,
#         dimensions=['Final Price (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews'],
#         title="Scatter Plot Matrix Filtered by Ratings"
#     )
#
#     return fig

# @app.callback(
#     Output('scatter-plot-matrix-slider1', 'figure'),
#     [Input('discount-depth-slider', 'value')]
# )
# def update_scatter_matrix(selected_discount_depth):
#     # Filter the DataFrame based on the selected discount depth value
#     filtered_df = myntra[myntra['Discount Depth (%)'] >= selected_discount_depth]
#
#     # Update the scatter plot matrix figure with the filtered data
#     fig = px.scatter_matrix(
#         filtered_df,
#         dimensions=['Final Price (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews'],
#         title="Scatter Plot Matrix Filtered by Discount Depth"
#     )
#
#     return fig

# Callback to set slider ranges
@app.callback(
    [Output('discount-depth-slider', 'min'),
     Output('discount-depth-slider', 'max'),
     Output('discount-depth-slider', 'marks')],
    [Input('stored-data', 'data')]
)
def set_slider_ranges(stored_data_json):
    if stored_data_json is None:
        raise PreventUpdate

    df = pd.read_json(stored_data_json, orient='split')
    df['Discount Depth (%)'] = ((df['OriginalPrice (in Rs)'] - df['Final Price (in Rs)']) / df[
        'OriginalPrice (in Rs)']) * 100

    min_discount = df['Discount Depth (%)'].min()
    max_discount = df['Discount Depth (%)'].max()
    marks = {str(int(i)): str(int(i)) for i in range(int(min_discount), int(max_discount) + 1, 5)}

    return min_discount, max_discount, marks


# Callback to update scatter plot matrix
# @app.callback(
#     Output('scatter-plot-matrix-slider1', 'figure'),
#     [Input('discount-depth-slider', 'value')],
#     [State('stored-data', 'data')]
# )
# def update_scatter_matrix(selected_discount_depth, stored_data_json):
#     if stored_data_json is None:
#         raise PreventUpdate
#
#     df = pd.read_json(stored_data_json, orient='split')
#     df['Discount Depth (%)'] = ((df['OriginalPrice (in Rs)'] - df['Final Price (in Rs)']) / df['OriginalPrice (in Rs)']) * 100
#     filtered_df = df[df['Discount Depth (%)'] >= selected_discount_depth]
#
#     fig = px.scatter_matrix(
#         filtered_df,
#         dimensions=['Final Price (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews'],
#         title="Scatter Plot Matrix Filtered by Discount Depth"
#     )
#
#     return fig

# Callback to update scatter plot matrix
@app.callback(
    Output('scatter-plot-matrix-slider1', 'figure'),
    [Input('discount-depth-slider', 'value')],
    [State('stored-data', 'data')]
)
def update_scatter_matrix(selected_discount_depth, stored_data_json):
    if stored_data_json is None:
        raise PreventUpdate

    df = pd.read_json(stored_data_json, orient='split')
    df['Discount Depth (%)'] = ((df['OriginalPrice (in Rs)'] - df['Final Price (in Rs)']) / df['OriginalPrice (in Rs)']) * 100
    filtered_df = df[df['Discount Depth (%)'] >= selected_discount_depth]

    fig = px.scatter_matrix(
        filtered_df,
        dimensions=['Final Price (in Rs)', 'OriginalPrice (in Rs)', 'Ratings', 'Reviews'],
        labels={'Final Price (in Rs)': 'Final Price (Rs)',
                'OriginalPrice (in Rs)': 'Original Price (Rs)',
                'Ratings': 'Ratings',
                'Reviews': 'Reviews'},
        title="Scatter Plot Matrix Filtered by Discount Depth"
    )

    # Update layout for title and axes
    fig.update_layout(
        title={
            'text': "Scatter Plot Matrix Filtered by Discount Depth",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'family': 'serif', 'color': 'blue', 'size': 20}
        },
        font=dict(family="serif", color="darkred", size=15),
        autosize=False,
        width=1200,  # Adjust width
        height=800,  # Adjust height
    )
    fig.update_xaxes(tickangle=45)
    return fig

# @app.callback(
#     Output('avg-discount-price-color', 'figure'),
#     [Input('stored-data', 'data')]
# )
# def update_avg_discount_price_color(stored_data_json):
#     if not stored_data_json:
#         raise PreventUpdate
#
#     df = pd.read_json(stored_data_json, orient='split')
#     avg_discounted_price_by_color = df.groupby('Individual_Category_Colour')['Discount Amount'].mean().reset_index()
#
#     fig = px.bar(avg_discounted_price_by_color, x='Individual_Category_Colour', y='Discount Amount',
#                  title='Average Discount Amount per Colour')
#     fig.update_layout(title_x=0.5,
#                       font=dict(family='serif', size=20),
#                       title_font_color='blue',
#                       xaxis_title_font=dict(family='serif', size=15, color='darkred'),
#                       yaxis_title_font=dict(family='serif', size=15, color='darkred'),
#                       # plot_bgcolor='lightblue',  # Ensure this is set to a color that contrasts with the bars
#                       # paper_bgcolor='lightblue',  # Same as above
#                       )
#
#     return fig
#
#
# @app.callback(
#     Output('ratings-distribution-color', 'figure'),
#     [Input('stored-data', 'data')]
# )
# def update_ratings_distribution_color(stored_data_json):
#     if not stored_data_json:
#         raise PreventUpdate
#
#     # Convert stored JSON data to DataFrame
#     df = pd.read_json(stored_data_json, orient='split')
#
#     # Convert 'Ratings' to numeric in case it's not
#     df['Ratings'] = pd.to_numeric(df['Ratings'], errors='coerce')
#
#     # Grouping by 'Individual_Category_Colour' and calculating mean ratings
#     avg_ratings_by_color = df.groupby('Individual_Category_Colour')['Ratings'].mean().reset_index()
#
#     # Creating the bar chart
#     fig = px.bar(avg_ratings_by_color, x='Individual_Category_Colour', y='Ratings', title='Average Ratings Distribution by Color')
#     fig.update_layout(title_x=0.5,
#                       font=dict(family='serif', size=20),
#                       title_font_color='blue',
#                       xaxis_title_font=dict(family='serif', size=15, color='darkred'),
#                       yaxis_title_font=dict(family='serif', size=15, color='darkred'),
#
#                       )
#
#     return fig




# @app.callback(
#     Output('combined-chart', 'figure'),
#     [Input('stored-data', 'data')]
# )
# def update_combined_chart(stored_data_json):
#     if not stored_data_json:
#         raise PreventUpdate
#
#     df = pd.read_json(stored_data_json, orient='split')
#
#     # Calculate average Discount Amount by color
#     avg_discounted_price_by_color = df.groupby('Individual_Category_Colour')['Discount Amount'].mean().reset_index()
#
#     # Calculate average ratings by color
#     df['Ratings'] = pd.to_numeric(df['Ratings'], errors='coerce')
#     avg_ratings_by_color = df.groupby('Individual_Category_Colour')['Ratings'].mean().reset_index()
#
#     # Create subplots
#     fig = make_subplots(rows=1, cols=2, subplot_titles=('Average Discount Amount per Colour', 'Average Ratings Distribution by Color'))
#
#     # Add bar for average Discount Amount
#     fig.add_trace(
#         go.Bar(x=avg_discounted_price_by_color['Individual_Category_Colour'], y=avg_discounted_price_by_color['Discount Amount']),
#         row=1, col=1
#     )
#
#     # Add bar for average ratings
#     fig.add_trace(
#         go.Bar(x=avg_ratings_by_color['Individual_Category_Colour'], y=avg_ratings_by_color['Ratings']),
#         row=1, col=2
#     )
#
#     # Update layout for the subplot figure
#     fig.update_layout(
#         title_text='Combined Chart: Average Discount Amount and Ratings Distribution by Colour',
#         title_x=0.5,
#         font=dict(family='serif', size=20),
#         title_font_color='blue',
#         showlegend=False
#     )
#
#     # Update x-axis and y-axis titles with specific styles
#     #fig.update_layout(row=1,col=1,title_font=dict(family='serif', size=20, color='blue'))
#     fig.update_xaxes(title_text='Colour', row=1, col=1, title_font=dict(family='serif', size=15, color='darkred'))
#     fig.update_yaxes(title_text='Discount Amount', row=1, col=1, title_font=dict(family='serif', size=15, color='darkred'))
#     fig.update_xaxes(title_text='Colour', row=1, col=2, title_font=dict(family='serif', size=15, color='darkred'))
#     fig.update_yaxes(title_text='Ratings', row=1, col=2, title_font=dict(family='serif', size=15, color='darkred'))
#     # Style the subplot titles
#     for i, annotation in enumerate(fig['layout']['annotations']):
#         annotation['font'] = dict(family='serif', size=20, color='blue')
#
#     return fig

@app.callback(
    Output('combined-chart', 'figure'),
    [Input('stored-data', 'data'),
     Input('distribution-type', 'value')]
)
def update_combined_chart(stored_data_json, distribution_type):
    if not stored_data_json:
        raise PreventUpdate

    df = pd.read_json(stored_data_json, orient='split')

    # Calculate average Discount Amount by color
    avg_discounted_price_by_color = df.groupby('Individual_Category_Colour')['Discount Amount'].mean().reset_index()

    # Calculate average ratings by color
    df['Ratings'] = pd.to_numeric(df['Ratings'], errors='coerce')
    #avg_ratings_by_color = df.groupby('Individual_Category_Colour')['Ratings','category_by_Gender'].mean().reset_index()
    avg_ratings_by_color = df.groupby('Individual_Category_Colour')['Ratings'].mean().reset_index()
    #print('Avg rating by color')
    #print(avg_ratings_by_color)
    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Average Discount Amount per Colour', 'Distribution of Ratings by Colour'))

    # Add bar for average Discount Amount
    fig.add_trace(

        go.Bar(x=avg_discounted_price_by_color['Individual_Category_Colour'],
               y=avg_discounted_price_by_color['Discount Amount'],
        text=avg_discounted_price_by_color['Discount Amount'].round(2),
        # Round to 2 decimal places for cleaner display
        textposition='outside'),
        row=1, col=1
    )

    # Depending on the selected distribution, create the corresponding plot for ratings
    if distribution_type == 'bar':
        fig.add_trace(
            go.Bar(x=avg_ratings_by_color['Individual_Category_Colour'], y=avg_ratings_by_color['Ratings']),
            row=1, col=2
        )
    elif distribution_type == 'box':
        #selected_colors = avg_ratings_by_color['Individual_Category_Colour'].head(2)

        fig.add_trace(
           # go.Box(x=df['category_by_Gender'], y=df['Ratings']),
            go.Box(x=df['Individual_Category_Colour'], y=df['Ratings']
                  ),
           #  px.histogram(avg_ratings_by_color,
           #       x="Individual_Category_Colour", y="Ratings",
           #      hover_data=avg_ratings_by_color.columns),
            row=1, col=2


        )
    elif distribution_type == 'violin':
        fig.add_trace(
            go.Violin(x=df['Individual_Category_Colour'], y=df['Ratings']
),
            row=1, col=2
        )

    # Style the subplot titles
    for i, annotation in enumerate(fig['layout']['annotations']):
        annotation['font'] = dict(family='serif', size=20, color='blue')

    # Update layout for the subplot figure
    fig.update_layout(
        title_x=0.5,
        font=dict(family='serif', size=20),
        title_font_color='blue',
        showlegend=False
    )

    # Update x-axis and y-axis titles with specific styles
    fig.update_xaxes(title_text='Colour', row=1, col=1, title_font=dict(family='serif', size=15, color='darkred'))
    fig.update_yaxes(title_text='Discount Amount', row=1, col=1, title_font=dict(family='serif', size=15, color='darkred'))
    fig.update_xaxes(title_text='Colour', row=1, col=2, title_font=dict(family='serif', size=15, color='darkred'))
    fig.update_yaxes(title_text='Ratings', row=1, col=2, title_font=dict(family='serif', size=15, color='darkred'))

    return fig


@app.callback(
    Output('color-distribution-pie', 'figure'),
    [Input('stored-data', 'data')]
)
def update_color_distribution_pie(stored_data_json):
    if not stored_data_json:
        raise PreventUpdate

    df = pd.read_json(stored_data_json, orient='split')
    color_counts = df[df['Individual_Category_Colour'] != 'Not specified']['Individual_Category_Colour'].value_counts()
    print(color_counts)
    fig = px.pie(color_counts, names=color_counts.index, values=color_counts.values)
    #fig.update_traces(texttemplate='%{percent:.1f}%')
    fig.update_layout(title_x=0.5,
                      title_text='Distribution of Colors in Individual Category (Excluding Not Specified)',
                      title_font={'family': 'serif', 'color': 'blue', 'size': 20})
    return fig
    # df = pd.read_json(stored_data_json, orient='split')
    # color_counts = df[df['Individual_Category_Colour'] != 'Not specified']['Individual_Category_Colour'].value_counts()
    # total = color_counts.sum()
    # # Create a new column with formatted percentages
    # formatted_percentages = [(count / total * 100).round(2) for count in color_counts]
    #
    # # Creating the pie chart
    # fig = px.pie(color_counts, names=color_counts.index, values=color_counts.values)
    #
    # # Update the text to be the formatted percentages and position it inside the chart
    # fig.update_traces(textinfo='text+%', text=formatted_percentages, textposition='inside')
    # #fig.update_traces(textinfo='percent', texttemplate='%{percent:.2f}%')
    #
    # # Updating the layout
    # fig.update_layout(
    #     title_x=0.5,
    #     title_text='Distribution of Colors in Individual Category (Excluding Not Specified)',
    #     title_font={'family': 'serif', 'color': 'blue', 'size': 20}
    # )
    #
    # # Show the figure
    # return fig

if __name__ == '__main__':
    app.run_server(debug=True)