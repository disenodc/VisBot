import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import openai
from openai import OpenAI


# Cargar la API key desde el entorno
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
)

st.set_page_config(
    page_title="VisBot - Visualization Recommender powered with Ai",
    page_icon="https://raw.githubusercontent.com/disenodc/VisBot/refs/heads/main/bot_2.ico",
)

# Function to read data from a file
def read_data(file_path):
    if file_path.name.endswith('.csv'):
        first_line = file_path.readline().decode('utf-8')
        file_path.seek(0)
        
        if '\t' in first_line:
            delimiter = '\t'
        elif ';' in first_line:
            delimiter = ';'
        else:
            delimiter = ','
            
        df = pd.read_csv(file_path, delimiter=delimiter)
    elif file_path.name.endswith('.txt'):
        first_line = file_path.readline().decode('utf-8')
        file_path.seek(0)
        
        if '\t' in first_line:
            delimiter = '\t'
        elif ';' in first_line:
            delimiter = ';'
        else:
            delimiter = ','
            
        df = pd.read_csv(file_path, delimiter=delimiter)
    elif file_path.name.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.name.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format.")
    
    return df

# Función para obtener recomendaciones de visualización de OpenAI utilizando GPT-4
def get_openai_recommendation(df):
    description = f"The data set has {df.shape[0]} rows and {df.shape[1]} columns. "
    for column in df.columns:
        description += f"The '{column}' column is of type {df[column].dtype}, "
        if pd.api.types.is_numeric_dtype(df[column]):
            description += f"and contains numerical values ranging from {df[column].min()} to {df[column].max()}. "
        
        elif isinstance(df[column].dtype, pd.CategoricalDtype) or df[column].nunique() < 10:
            description += f"and contains {df[column].nunique()} unique categories. "

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            description += f"and contains time data ranging from {df[column].min()} to {df[column].max()}. "

    response = client.chat.completions.create(
        model= "gpt-4-turbo",  # Cambiamos a gpt-4 o gpt-4-turbo
        messages=[
            {"role": "system",
            "content": "You are an expert assistant in data analysis specialized in visualization."},
            {"role": "user",
            "content": f"I have a data set. {description} What type of visualizations would you recommend from this data? Describe and if possible recommend options"}
        ],
        max_tokens=500  # Increase tokens if more context is desired
    )
    return response.choices[0].message.content

# Function to generate visualizations
def recommend_and_plot(df):
    # st.subheader("Recommended Visualizations")
    # Configurable parameters in the sidebar
    st.sidebar.header("Configuration")
    try:
        # Selectbox to select columns for X, Y, and Z axes
        x_axis = st.sidebar.selectbox("Select the column for the X axis", df.columns)
        y_axis = st.sidebar.selectbox("Select the column for the Y axis", df.columns)
        
        z_axis = None
        if chart_type in ["3D Scatter Chart", "Stacked Bar Chart", "Grouped Bar Chart"]:
            z_axis = st.sidebar.selectbox("Select the column for the Z axis", df.columns)

        # Additional parameters
        st.sidebar.subheader("Configurable Parameters")
        hist_bins = st.sidebar.slider("Number of bins for histograms", min_value=10, max_value=100, value=20, key="hist_bins_slider")
        scatter_size = st.sidebar.slider("Size of points on scatter plot", min_value=5, max_value=50, value=10, key="scatter_size_slider")

        # Check if selected columns exist in the DataFrame
        if x_axis not in df.columns or y_axis not in df.columns or (z_axis and z_axis not in df.columns):
            raise ValueError(f"One of the selected columns is invalid. Expected columns: {list(df.columns)}.")

        # Generate scatter plot (simple example)
        fig_scatter = px.bar(df, x=x_axis, y=y_axis, title=f'Bar chart from {x_axis} vs {y_axis}')
        
        # fig_scatter = px.scatter(df, x=x_axis, y=y_axis, size_max=scatter_size, title=f'Scatter Plot from {x_axis} vs {y_axis}')
        
        st.plotly_chart(fig_scatter)
    except ValueError as e:
        # Show warning if an error related to columns occurs
        st.warning(f"Warning: {str(e)}")
    except Exception as e:
        # Any other unforeseen error
        st.error(f"An unexpected error occurred: {str(e)}")

# Function to generate charts depending on selected data types and chart type
def generate_plot(df, chart_type, x_axis=None, y_axis=None, z_axis=None, hist_bins=30, scatter_size=10):
    fig = None
    if chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter plot from {x_axis} vs {y_axis}', size_max=scatter_size)
    elif chart_type == "Bar Chart":
        fig = px.bar(df, x=x_axis, y=y_axis, title=f'Bar chart from {x_axis} vs {y_axis}')
    elif chart_type == "Stacked Bar Chart":
        fig = px.bar(df, x=x_axis, y=y_axis, color=z_axis, title=f'Stacked Bar Chart from {x_axis} vs {y_axis} for {z_axis}', barmode='stack')
    elif chart_type == "Grouped Bar Chart":
        fig = px.bar(df, x=x_axis, y=y_axis, color=z_axis, title=f'Grouped Bar Chart from {x_axis} vs {y_axis} for {z_axis}', barmode='group')
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_axis, nbins=hist_bins, title=f'Histogram from {x_axis}')
    elif chart_type == "Line Graph":
        fig = px.line(df, x=x_axis, y=y_axis, title=f'Line Graph from {x_axis} vs {y_axis}')
    elif chart_type == "Area Chart":
         fig = px.area(df, x=x_axis, y=y_axis, title=f'Area Chart from {x_axis} vs {y_axis}')
    elif chart_type == "Pie Chart":
         fig = px.pie(df, names=x_axis, title=f'Pie Chart from {x_axis}')
    elif chart_type == "3D Scatter Plot":
         fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, title=f'Scatter plot 3D from {x_axis}, {y_axis} and {z_axis}')
    elif chart_type == "Boxplot":
         fig = px.box(df, x=x_axis, y=y_axis, title=f'Boxplot from {x_axis} vs {y_axis}')
    elif chart_type == "Violin plot":
         fig = px.violin(df, x=x_axis, y=y_axis, title=f'Violin from {x_axis} vs {y_axis}')
    elif chart_type == "Heat map":
         fig = px.density_heatmap(df, x=x_axis, y=y_axis, title=f'Heatmap from {x_axis} vs {y_axis}')
    elif chart_type == "Geospatial scatter map":
         fig = px.scatter_geo(df, lat=y_axis, lon=x_axis, title=f'Geospatial scatter map from {x_axis} and {y_axis}')
    elif chart_type == "Choropleth map":
         fig = px.choropleth(df, locations=x_axis, color=y_axis, title=f'Choropleth map from {x_axis} and {y_axis}')
    elif chart_type == "Sun diagram":
         fig = px.sunburst(df, path=[x_axis], title=f'Sun diagram from {x_AXIS} and {y_AXIS}')
    elif chart_type == "Time Series Plot":
        fig = px.line(df, x=x_axis, y=y_axis, title=f'Time Series Plot from {x_axis} vs {y_axis}')
    elif chart_type == "Treemap":
        fig = px.treemap(df, path=[x_axis], values=y_axis, title=f'Treemap from {x_axis} with values {y_axis}')
    
    
    return fig

# Main function to run the Streamlit app
def main():

    # Encabezado con logo y título
    logo_url = "https://raw.githubusercontent.com/disenodc/VisBot/refs/heads/main/bot_1.png"  # Cambia por la URL de tu logo o archivo local
    st.markdown(
        f"""
        <div style="display: flex; align-items: start; gap: 5px;">
        <img src="{logo_url}" alt="Logo" style="width: 5em; height: 5em;">
        <h1 style="margin: 0;">VisBot</h1>
        </div>
        """,
    unsafe_allow_html=True
)

    # User interface with Streamlit
    st.title("Visualization Recommender powered with Ai")
    
    # Input for OpenAI API key
    api_key = openai.api_key
    
    # Input for URL
    #url_input = st.text_input("Enter the URL of the data file")
    
    # Input to upload a local file
    uploaded_file = st.file_uploader("Here, Load your CSV, JSON, TXT OR XLSX file", type=["csv", "json", "txt", "xlsx"])
    
    df = None

    # Leer el archivo o URL
    
    if uploaded_file is not None:
        try:
            df = read_data(uploaded_file)
        except ValueError as e:
            st.error(f"Error processing file: {e}")

    if df is not None:
        st.write(df.head())  # Mostrar las primeras filas del DataFrame

        # Generar y mostrar las recomendaciones de visualización de OpenAI
        st.subheader("Ai Recommended Visualizations")
        try:
            recommendation = get_openai_recommendation(df)
            st.markdown(recommendation)  # Muestra las recomendaciones de GPT-4
        except Exception as e:
            st.error(f"Error getting OpenAI recommendations: {str(e)}")

        # Selección de tipo de gráfico (chart_type) y variables para los ejes
        chart_type = st.sidebar.selectbox(
            "Select chart type",
            [
                "Scatter Plot", "Bar Chart", "Stacked Bar Chart", 
                "Histogram","Line Chart", "Area Chart", "Boxplot", "Pie chart",
                "3D Scatter Plot",  "Violin plot", "Heat map",
                "Geospatial scatter map", "Choropleth map", "Sun diagram", "Time Series Plot", "Treemap"
            ]
        )

        x_axis = st.sidebar.selectbox("Select the column for the X axis", df.columns)
        y_axis = st.sidebar.selectbox("Select the column for the Y axis", df.columns)

        z_axis = None
        if chart_type in ["3D Scatter Plot", "Stacked Bar Chart", "Grouped Bar Chart"]:
            z_axis = st.sidebar.selectbox("Select column for Z axis (optional)", df.columns)

        hist_bins = st.sidebar.slider("Number of bins for histograms", min_value=10, max_value=100, value=20, key="hist_bins_slider")
        scatter_size = st.sidebar.slider("Size of points on scatter plot", min_value=5, max_value=50, value=10, key="scatter_size_slider")

        # Generate the selected chart
        fig = generate_plot(df, chart_type, x_axis, y_axis, z_axis, hist_bins, scatter_size)

        # Show the chart in the interface
        if fig:
            st.plotly_chart(fig)
        
        else:
            st.warning("Please select a chart type and valid columns.")

if __name__ == "__main__":
    main()