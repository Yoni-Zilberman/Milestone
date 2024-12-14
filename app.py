import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import base64
import io

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Upload File"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select File')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Div(id='target-selection'),
    html.Div(id='categorical-selection'),
    html.Div([
        html.Div(id='avg-plot', style={'width': '48%', 'display': 'inline-block'}),
        html.Div(id='corr-plot', style={'width': '48%', 'display': 'inline-block'})
    ]),
    html.Div(id='feature-selection'),
    html.Button('Train', id='train-button', n_clicks=0),
    html.Div(id='training-output'),
    dcc.Input(id='prediction-input', type='text', placeholder='Enter values (comma-separated)'),
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output'),
    dcc.Store(id='stored-data'),
    dcc.Store(id='stored-model')
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df
    except Exception as e:
        print(e)
        return None

@app.callback(
    [Output('output-data-upload', 'children'),
     Output('stored-data', 'data'),
     Output('target-selection', 'children'),
     Output('categorical-selection', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return [html.Div()] * 4
    
    df = parse_contents(contents, filename)
    if df is not None:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        target_selection = html.Div([
            html.H3("Select Target:"),
            dcc.Dropdown(
                id='target-dropdown',
                options=[{'label': col, 'value': col} for col in numeric_columns],
                value=numeric_columns[0]
            )
        ])
        
        categorical_selection = html.Div([
            dcc.RadioItems(
                id='categorical-radio',
                options=[{'label': col, 'value': col} for col in categorical_columns],
                value=categorical_columns[0]
            )
        ])
        
        return [
            html.Div([html.H5(f'Uploaded: {filename}')]),
            df.to_json(date_format='iso', orient='split'),
            target_selection,
            categorical_selection
        ]
    return [html.Div()] * 4

@app.callback(
    [Output('avg-plot', 'children'),
     Output('corr-plot', 'children')],
    [Input('categorical-radio', 'value'),
     Input('target-dropdown', 'value'),
     Input('stored-data', 'data')]
)
def update_graphs(selected_cat, target, stored_data):
    if stored_data is None:
        return [html.Div()] * 2
    
    df = pd.read_json(stored_data, orient='split')
    
    avg_fig = px.bar(
        df.groupby(selected_cat)[target].mean().reset_index(),
        x=selected_cat,
        y=target,
        title=f'Average {target} by {selected_cat}'
    )
    
    correlations = df.select_dtypes(include=['float64', 'int64']).corr()[target].abs()
    correlations = correlations[correlations.index != target]
    corr_fig = px.bar(
        x=correlations.index,
        y=correlations.values,
        title=f'Correlation Strength of Numerical Variables with {target}'
    )
    
    return [
        dcc.Graph(figure=avg_fig),
        dcc.Graph(figure=corr_fig)
    ]

@app.callback(
    Output('feature-selection', 'children'),
    [Input('stored-data', 'data')]
)
def update_feature_selection(stored_data):
    if stored_data is None:
        return html.Div()
    
    df = pd.read_json(stored_data, orient='split')
    return html.Div([
        html.H3("Select features for training:"),
        html.Div([
            dcc.Checklist(
                id='feature-checklist',
                options=[{'label': col, 'value': col} for col in df.columns],
                value=[],
                inline=True
            )
        ])
    ])

@app.callback(
    [Output('training-output', 'children'),
     Output('stored-model', 'data')],
    [Input('train-button', 'n_clicks')],
    [State('feature-checklist', 'value'),
     State('target-dropdown', 'value'),
     State('stored-data', 'data')]
)
def train_model(n_clicks, selected_features, target, stored_data):
    if n_clicks == 0 or not selected_features or stored_data is None:
        return html.Div(), None
    
    df = pd.read_json(stored_data, orient='split')
    X = df[selected_features]
    y = df[target]
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ] if len(categorical_features) > 0 else [
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    r2_score = model.score(X_test, y_test)
    
    return html.Div(f"The R2 score is: {r2_score:.2f}"), {
        'features': selected_features,
        'target': target,
        'model_trained': True
    }

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('prediction-input', 'value'),
     State('stored-model', 'data'),
     State('stored-data', 'data')]
)
def make_prediction(n_clicks, input_value, model_data, stored_data):
    if n_clicks == 0 or not input_value or not model_data or not stored_data:
        return html.Div()
    
    try:
        values = [x.strip().lower() if isinstance(x, str) else x 
                 for x in input_value.split(",")]
        
        if len(values) != len(model_data['features']):
            return html.Div("Please enter values for all selected features", style={'color': 'red'})
        
        df = pd.read_json(stored_data, orient='split')
        X = df[model_data['features']]
        y = df[model_data['target']]
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ] if len(categorical_features) > 0 else [
                ('num', StandardScaler(), numeric_features)
            ]
        )
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42))
        ])
        
        X_train = X
        y_train = y
        model.fit(X_train, y_train)
        
        input_df = pd.DataFrame([values], columns=model_data['features'])
        prediction = model.predict(input_df)[0]
        return html.Div(f"Predicted {model_data['target']}: {prediction:.2f}")
    
    except Exception as e:
        return html.Div(f"Error in prediction: {str(e)}", style={'color': 'red'})

if __name__ == '__main__':
    app.run_server(debug=True)
