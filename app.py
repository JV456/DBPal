import base64
import io
import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import anthropic
import google.generativeai as genai
import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlalchemy as sa
import streamlit as st
from groq import Groq
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, inspect, text

# Analytics Engine Integration
try:
    from analytics_integration import get_analytics_integrator
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    st.warning("⚠️ Analytics features unavailable. Install required packages: prophet, statsmodels")

# Page configuration
st.set_page_config(
    page_title="DBPal",
    page_icon="🗃️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #ddd;
    }
    .user-message {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        color: #212529;
    }
    .assistant-message {
        background-color: #ffffff;
        border-left: 4px solid #28a745;
        color: #212529;
    }
    .sql-query {
        background-color: transparent;
        border: 2px solid #28a745;
        border-radius: 6px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
        color: #28a745;
        font-size: 14px;
        font-weight: 600;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
        color: #212529;
    }
</style>""", unsafe_allow_html=True)

def render_database_form():
    """Render database connection form with dynamic fields based on database type"""
    
    # Use session state to track database type changes
    db_type = st.selectbox(
        "Database Type",
        ["PostgreSQL", "MySQL", "SQL Server"],
        help="Select your database type",
        key="db_type_selector"
    )
    
    # Initialize connection params
    connection_params = {"db_type": db_type}
    
    # Default ports for each database type
    default_ports = {
        "PostgreSQL": 5432,
        "MySQL": 3306, 
        "SQL Server": 1433
    }
    
    # All databases need server connection details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        host = st.text_input(
            "Host", 
            value="localhost",
            help="Database server hostname or IP address",
            key="db_host"
        )
    
    with col2:
        # Update port dynamically based on database type
        current_port = default_ports.get(db_type, 5432)
        port = st.number_input(
            "Port", 
            min_value=1, 
            max_value=65535, 
            value=current_port,
            help=f"Default port for {db_type} is {current_port}",
            key="db_port"
        )
    
    # Database name field
    db_name = st.text_input(
        "Database Name", 
        help="Name of the database/schema to connect to",
        key="db_name"
    )
    
    # Username and Password
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input(
            "Username", 
            help="Database username",
            key="db_username"
        )
    
    with col2:
        password = st.text_input(
            "Password", 
            type="password", 
            help="Database password",
            key="db_password"
        )
    
    # SQL Server specific field - this should now appear dynamically
    instance = None
    if db_type == "SQL Server":
        instance = st.text_input(
            "Instance (Optional)", 
            help="SQL Server instance name (leave empty for default instance)",
            key="sql_server_instance"
        )
        if instance:
            connection_params["instance"] = instance
    
    # Store all parameters
    connection_params.update({
        "host": host,
        "port": port,
        "database": db_name,
        "username": username,
        "password": password
    })
    

    
    return connection_params

def validate_connection_params(params):
    """Validate connection parameters based on database type"""
    
    db_type = params.get("db_type")
    
    # Server database validation
    required_fields = ["host", "database", "username"]
    missing_fields = []
    
    for field in required_fields:
        if not params.get(field, "").strip():
            missing_fields.append(field.title())
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate port
    port = params.get("port")
    if not isinstance(port, int) or port < 1 or port > 65535:
        return False, "Port must be between 1 and 65535"
    
    return True, ""

def create_connection_string(params):
    """Create appropriate connection string based on database type"""
    
    db_type = params["db_type"]
    
    # Server databases
    host = params["host"]
    port = params["port"]  
    database = params["database"]
    username = quote_plus(params["username"]) if params["username"] else ""
    password = quote_plus(params.get("password", ""))
    
    if db_type == "PostgreSQL":
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    elif db_type == "MySQL":
        return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    elif db_type == "SQL Server":
        instance = params.get("instance", "")
        if instance:
            return f"mssql+pyodbc://{username}:{password}@{host}\\{instance}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
        else:
            return f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    
    raise ValueError(f"Unsupported database type: {db_type}")

class DatabaseChatAssistant:
    def __init__(self):
        self.engine = None
        self.inspector = None
        self.schema_info = {}
        self.connection_status = False
        
    def connect_to_database(self, db_type: str, host: str, port: str, 
                           database: str, username: str, password: str) -> bool:
        """Connect to database and cache schema information"""
        try:
            # URL encode credentials to handle special characters
            encoded_username = quote_plus(username) if username else ""
            encoded_password = quote_plus(password) if password else ""
            
            # Create connection string based on database type
            if db_type == "PostgreSQL":
                conn_string = f"postgresql://{encoded_username}:{encoded_password}@{host}:{port}/{database}"
            elif db_type == "MySQL":
                conn_string = f"mysql+pymysql://{encoded_username}:{encoded_password}@{host}:{port}/{database}"
            elif db_type == "SQL Server":
                conn_string = f"mssql+pyodbc://{encoded_username}:{encoded_password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            # Debug: Show connection attempt (without password)
            debug_conn_string = f"{db_type} - {encoded_username}@{host}:{port}/{database}"
            st.info(f"Connecting to: {debug_conn_string}")
            
            # Create engine and test connection
            self.engine = create_engine(conn_string, pool_pre_ping=True)
            self.inspector = inspect(self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Cache schema information
            self._cache_schema_info()
            self.connection_status = True
            return True
            
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages for common issues
            if "Can't connect to MySQL server" in error_msg:
                st.error("❌ **MySQL Connection Failed**")
                st.error("**Possible causes:**")
                st.error("• MySQL server is not running")
                st.error("• Incorrect host/port (try 127.0.0.1:3306)")
                st.error("• Firewall blocking connection")
                st.error("• Invalid credentials")
                st.error(f"**Technical details:** {error_msg}")
            elif "Access denied" in error_msg:
                st.error("❌ **Authentication Failed**")
                st.error("• Check your username and password")
                st.error("• Ensure user has permissions to access the database")
            elif "Unknown database" in error_msg:
                st.error("❌ **Database Not Found**")
                st.error("• Check the database name spelling")
                st.error("• Ensure the database exists")
            else:
                st.error(f"❌ **Connection Error:** {error_msg}")
            
            self.connection_status = False
            return False
    
    def connect_with_string(self, connection_string: str, db_type: str) -> bool:
        """Connect to database using a connection string"""
        try:
            # Create engine and test connection
            self.engine = create_engine(connection_string, pool_pre_ping=True)
            self.inspector = inspect(self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Cache schema information
            self._cache_schema_info()
            self.connection_status = True
            return True
            
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages for common issues
            # Server database error handling
            if "Can't connect to" in error_msg or "could not connect to server" in error_msg:
                st.error(f"❌ **{db_type} Connection Failed**")
                st.error("**Possible causes:**")
                st.error(f"• {db_type} server is not running")
                st.error("• Incorrect host/port")
                st.error("• Firewall blocking connection")
                st.error("• Network connectivity issues")
            elif "Access denied" in error_msg or "authentication failed" in error_msg:
                st.error("❌ **Authentication Failed**")
                st.error("• Check your username and password")
                st.error("• Ensure user has permissions to access the database")
            elif "Unknown database" in error_msg or "database does not exist" in error_msg:
                st.error("❌ **Database Not Found**")
                st.error("• Check the database name spelling")
                st.error("• Ensure the database exists")
            else:
                st.error(f"❌ **{db_type} Connection Error:** {error_msg}")
            
            self.connection_status = False
            return False
    
    def _cache_schema_info(self):
        """Cache database schema information for faster query generation"""
        try:
            tables = self.inspector.get_table_names()
            self.schema_info = {}
            
            for table in tables:
                columns = self.inspector.get_columns(table)
                self.schema_info[table] = {
                    'columns': [col['name'] for col in columns],
                    'types': {col['name']: str(col['type']) for col in columns}
                }
                
        except Exception as e:
            st.error(f"Error caching schema: {str(e)}")
    
    def get_schema_context(self) -> str:
        """Generate schema context for LLM"""
        context = "Database Schema:\n"
        for table, info in self.schema_info.items():
            context += f"\nTable: {table}\n"
            for col in info['columns']:
                col_type = info['types'].get(col, 'unknown')
                context += f"  - {col} ({col_type})\n"
        return context
    
    def validate_sql_query(self, query: str) -> bool:
        """Validate SQL query for safety"""
        # Convert to uppercase for checking
        query_upper = query.upper().strip()
        
        # Block dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'CREATE']
        
        # Allow in advanced mode
        if not st.session_state.get('advanced_mode', False):
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    return False
        
        return True
    
    def execute_query(self, query: str) -> tuple[pd.DataFrame, str]:
        """Execute SQL query and return results"""
        try:
            if not self.validate_sql_query(query):
                return None, "Query contains potentially dangerous operations. Enable Advanced Mode to proceed."
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df, "Success"
                
        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            return None, error_msg
    
    def generate_sql_from_natural_language(self, question: str, api_key: str, provider: str) -> str:
        """Convert natural language to SQL using selected AI provider"""
        try:
            schema_context = self.get_schema_context()
            
            prompt = f"""
            You are an expert SQL analyst. Convert the following natural language question into a SQL query.
            
            {schema_context}
            
            Rules:
            1. Generate ONLY the SQL query, no explanations
            2. Use proper SQL syntax
            3. For date/time queries, use appropriate date functions
            4. Use JOINs when necessary to connect related tables
            5. Add appropriate WHERE clauses for filtering
            6. Use GROUP BY and aggregation functions when asking for summaries
            7. Order results meaningfully (e.g., by date, amount, etc.)
            8. Limit results to reasonable numbers (e.g., TOP 10, LIMIT 100)
            
            Question: {question}
            
            SQL Query:
            """
            
            if provider == "Google Gemini":
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                sql_query = response.text.strip()
                
            elif provider == "OpenAI GPT":
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                sql_query = response.choices[0].message.content.strip()
                
            elif provider == "Anthropic Claude":
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                sql_query = response.content[0].text.strip()
                
            elif provider == "Groq":
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                sql_query = response.choices[0].message.content.strip()
                
            else:
                return f"Unsupported AI provider: {provider}"
            
            # Clean up the response
            sql_query = re.sub(r'^```sql\s*', '', sql_query)
            sql_query = re.sub(r'```\s*$', '', sql_query)
            
            return sql_query
            
        except Exception as e:
            return f"Error generating SQL with {provider}: {str(e)}"
    
    def generate_insights(self, df: pd.DataFrame, question: str, api_key: str, provider: str) -> str:
        """Generate insights from query results using selected AI provider"""
        try:
            # Sample data for context (first 5 rows)
            sample_data = df.head().to_string()
            summary_stats = df.describe().to_string() if len(df.select_dtypes(include=['number']).columns) > 0 else "No numeric data"
            
            prompt = f"""
            Analyze the following query results and provide clear, actionable insights.
            
            Original Question: {question}
            
            Data Sample:
            {sample_data}
            
            Summary Statistics:
            {summary_stats}
            
            Total Rows: {len(df)}
            
            Provide:
            1. Key findings from the data
            2. Notable trends or patterns
            3. Business insights or recommendations
            4. Any anomalies or interesting observations
            
            Keep the response concise and business-focused.
            """
            
            if provider == "Google Gemini":
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                return response.text
                
            elif provider == "OpenAI GPT":
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.3
                )
                return response.choices[0].message.content
                
            elif provider == "Anthropic Claude":
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            elif provider == "Groq":
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.3
                )
                return response.choices[0].message.content
                
            else:
                return f"Unsupported AI provider for insights: {provider}"
            
        except Exception as e:
            return f"Unable to generate insights with {provider}: {str(e)}"
    
    def create_visualizations(self, df: pd.DataFrame, question: str) -> List[go.Figure]:
        """Create appropriate visualizations based on data and question"""
        visualizations = []
        
        if df.empty:
            return visualizations
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Convert potential date columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    pass
        
        # Time series plot
        if date_cols and numeric_cols:
            for date_col in date_cols[:1]:  # Use first date column
                for num_col in numeric_cols[:2]:  # Max 2 numeric columns
                    fig = px.line(df, x=date_col, y=num_col, 
                                title=f"{num_col} over Time")
                    fig.update_layout(height=400)
                    visualizations.append(fig)
        
        # Bar chart for categorical data
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Aggregate data if needed
            if len(df) > 20:
                agg_df = df.groupby(cat_col)[num_col].sum().reset_index()
                agg_df = agg_df.nlargest(10, num_col)
            else:
                agg_df = df
            
            fig = px.bar(agg_df, x=cat_col, y=num_col,
                        title=f"{num_col} by {cat_col}")
            fig.update_layout(height=400)
            visualizations.append(fig)
        
        # Pie chart for categorical distributions
        if categorical_cols:
            cat_col = categorical_cols[0]
            value_counts = df[cat_col].value_counts().head(10)
            
            fig = px.pie(values=value_counts.values, names=value_counts.index,
                        title=f"Distribution of {cat_col}")
            fig.update_layout(height=400)
            visualizations.append(fig)
        
        # Correlation heatmap for multiple numeric columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                           title="Correlation Matrix",
                           color_continuous_scale="RdBu_r")
            fig.update_layout(height=400)
            visualizations.append(fig)
        
        return visualizations

def init_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'db_assistant' not in st.session_state:
        st.session_state.db_assistant = DatabaseChatAssistant()
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'show_sql' not in st.session_state:
        st.session_state.show_sql = True
    if 'advanced_mode' not in st.session_state:
        st.session_state.advanced_mode = False
    
    # Analytics settings
    if 'enable_auto_insights' not in st.session_state:
        st.session_state.enable_auto_insights = True
    if 'enable_anomaly_detection' not in st.session_state:
        st.session_state.enable_anomaly_detection = True
    if 'enable_forecasting' not in st.session_state:
        st.session_state.enable_forecasting = True
    if 'anomaly_sensitivity' not in st.session_state:
        st.session_state.anomaly_sensitivity = 2.0
    if 'max_charts' not in st.session_state:
        st.session_state.max_charts = 3
    if 'forecast_days' not in st.session_state:
        st.session_state.forecast_days = 30

def create_download_link(df: pd.DataFrame, filename: str, file_format: str) -> str:
    """Create download link for dataframe"""
    if file_format == "CSV":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📥 Download CSV</a>'
    elif file_format == "Excel":
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">📥 Download Excel</a>'

def main():
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header"><h1>🗃️ Database Chat & Analytics Assistant</h1><p>Natural language queries, AI-powered insights, and live visualizations — all in one chat</p></div>', unsafe_allow_html=True)
    
    # Sidebar for database connection
    with st.sidebar:
        st.header("🔌 Database Connection")
        
        # AI Provider Selection
        ai_provider = st.selectbox("AI Provider", 
                                 ["Google Gemini", "OpenAI GPT", "Anthropic Claude", "Groq"],
                                 help="Choose your preferred AI provider for natural language processing")
        
        # API Key input based on selected provider
        if ai_provider == "Google Gemini":
            api_key = st.text_input("Google Gemini API Key", type="password", 
                                   help="Get your free API key from https://aistudio.google.com/app/apikey")
        elif ai_provider == "OpenAI GPT":
            api_key = st.text_input("OpenAI API Key", type="password", 
                                   help="Get your API key from https://platform.openai.com/api-keys")
        elif ai_provider == "Anthropic Claude":
            api_key = st.text_input("Anthropic API Key", type="password", 
                                   help="Get your API key from https://console.anthropic.com/")
        elif ai_provider == "Groq":
            api_key = st.text_input("Groq API Key", type="password", 
                                   help="Get your free API key from https://console.groq.com/")
        
        if api_key:
            os.environ[f"{ai_provider.upper().replace(' ', '_')}_API_KEY"] = api_key
        
        # Database connection form - outside of st.form for dynamic updates
        connection_params = render_database_form()
        
        # Connect button - separate from the form to allow dynamic field updates
        connect_btn = st.button("Connect to Database", type="primary", key="connect_db_btn")
        
        if connect_btn and api_key:
            # Validate connection parameters
            is_valid, error_msg = validate_connection_params(connection_params)
            
            if not is_valid:
                st.error(error_msg)
                return
            
            with st.spinner("Connecting to database..."):
                # Show connection details (without sensitive info)
                db_type = connection_params["db_type"]
                host = connection_params["host"]
                port = connection_params["port"]
                database = connection_params["database"]
                username = connection_params["username"]
                st.info(f"Connecting to {db_type}: {username}@{host}:{port}/{database}")
                
                # Create connection string
                try:
                    connection_string = create_connection_string(connection_params)
                    
                    # Use simplified connection method
                    if st.session_state.db_assistant.connect_with_string(connection_string, db_type):
                        st.session_state.connected = True
                        st.success("✅ Connected successfully!")
                        st.info(f"Tables found: {len(st.session_state.db_assistant.schema_info)}")
                        
                        # Store connection params for reconnection if needed
                        st.session_state.connection_params = connection_params
                        
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
                    
        elif connect_btn and not api_key:
            st.error(f"Please enter your {ai_provider} API key first")
        
        # Settings
        if st.session_state.connected:
            st.header("⚙️ Settings")
            st.info(f"🤖 Using: {ai_provider}")
            st.session_state.show_sql = st.checkbox("Show SQL Queries", value=st.session_state.show_sql)
            st.session_state.advanced_mode = st.checkbox("Advanced Mode", value=st.session_state.advanced_mode,
                                                        help="Allow UPDATE, DELETE, and other potentially dangerous operations")
            
            # Analytics Controls
            if ANALYTICS_AVAILABLE:
                st.header("📊 Analytics Engine")
                st.session_state.enable_auto_insights = st.checkbox("Auto Insights", value=True, 
                                                                   help="Generate automatic data insights")
                st.session_state.enable_anomaly_detection = st.checkbox("Anomaly Detection", value=True,
                                                                       help="Detect unusual patterns in data")
                st.session_state.enable_forecasting = st.checkbox("Smart Forecasting", value=True,
                                                                 help="Enable predictive analytics")
                
                with st.expander("🔧 Advanced Analytics"):
                    st.session_state.anomaly_sensitivity = st.slider("Anomaly Sensitivity", 1.5, 3.0, 2.0, 0.1,
                                                                    help="Lower = more sensitive")
                    st.session_state.max_charts = st.number_input("Max Charts", 1, 5, 3,
                                                                 help="Maximum visualizations to generate")
                    st.session_state.forecast_days = st.number_input("Default Forecast Days", 7, 365, 30,
                                                                    help="Default forecasting period")
                    
                    # Analytics status
                    st.info("✅ Advanced Analytics Active")
                    
            else:
                st.header("📊 Analytics Status")
                st.warning("⚠️ Advanced analytics disabled")
                st.caption("Install prophet & statsmodels for full features")
            
            # Clear chat history
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Main chat interface
    if not st.session_state.connected:
        st.info("👈 Please connect to your database first using the sidebar.")
        
        # Sample questions for demo
        st.subheader("Sample Questions You Can Ask:")
        sample_questions = [
            "What are my total sales this year?",
            "Show me the top 10 customers by revenue",
            "How many orders were placed last month?",
            "What's the average order value by region?",
            "Show me sales trends over the last 6 months",
            "Which products are performing best?"
        ]
        
        for question in sample_questions:
            st.write(f"• {question}")
        
        return
    
    # Sample questions button
    if st.button("💡 Show Sample Questions"):
        sample_questions = [
            "What are my total sales?",
            "Show top 5 customers by revenue",
            "How many orders were placed last month?",
            "What's the average order value?",
            "Show me monthly sales trends",
            "Which products have the highest profit margin?"
        ]
        st.write("**Sample Questions:**")
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                st.session_state.user_input = q
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
                
                if 'sql' in message and st.session_state.show_sql:
                    st.markdown(f'<div class="sql-query"><strong>SQL Query:</strong><br><code>{message["sql"]}</code></div>', 
                               unsafe_allow_html=True)
                
                if 'dataframe' in message:
                    st.dataframe(message['dataframe'], use_container_width=True)
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(create_download_link(message['dataframe'], "query_result", "CSV"), 
                                   unsafe_allow_html=True)
                    with col2:
                        st.markdown(create_download_link(message['dataframe'], "query_result", "Excel"), 
                                   unsafe_allow_html=True)
                
                # Enhanced Analytics Display
                if ANALYTICS_AVAILABLE and 'analytics_results' in message:
                    analytics = message['analytics_results']
                    
                    # Display visualizations
                    if analytics.get('visualizations'):
                        st.subheader("📊 Auto-Generated Visualizations")
                        
                        num_charts = len(analytics['visualizations'])
                        if num_charts == 1:
                            st.plotly_chart(analytics['visualizations'][0], use_container_width=True)
                        elif num_charts == 2:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(analytics['visualizations'][0], use_container_width=True)
                            with col2:
                                st.plotly_chart(analytics['visualizations'][1], use_container_width=True)
                        else:
                            # Show first chart full width, others in grid
                            st.plotly_chart(analytics['visualizations'][0], use_container_width=True)
                            if num_charts > 1:
                                cols = st.columns(min(num_charts - 1, 2))
                                for i, chart in enumerate(analytics['visualizations'][1:]):
                                    with cols[i % len(cols)]:
                                        st.plotly_chart(chart, use_container_width=True)
                    
                    # Display anomalies
                    if analytics.get('anomalies', {}).get('anomalies'):
                        st.subheader("⚠️ Anomaly Detection")
                        st.warning(analytics['anomalies']['summary'])
                        
                        with st.expander("🔍 Anomaly Details"):
                            anomaly_data = []
                            for anomaly in analytics['anomalies']['anomalies'][:10]:
                                anomaly_data.append({
                                    'Date': anomaly['date'],
                                    'Value': f"{anomaly['value']:,.1f}",
                                    'Expected': f"{anomaly.get('expected', 0):,.1f}",
                                    'Severity': anomaly['severity'].upper(),
                                    'Z-Score': f"{anomaly['z_score']:.2f}σ"
                                })
                            if anomaly_data:
                                st.dataframe(pd.DataFrame(anomaly_data), use_container_width=True)
                    
                    # Display forecast results
                    if 'forecast_result' in message and message['forecast_result'].get('forecast'):
                        st.subheader("📈 Forecast Analysis")
                        st.success(message['forecast_result']['summary'])
                        
                        # Create forecast chart
                        forecast_df = pd.DataFrame(message['forecast_result']['forecast'])
                        if not forecast_df.empty:
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_df['ds'],
                                y=forecast_df['yhat'],
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(dash='dash', width=3, color='blue')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_df['ds'],
                                y=forecast_df['yhat_upper'],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_df['ds'],
                                y=forecast_df['yhat_lower'],
                                mode='lines',
                                line=dict(width=0),
                                fillcolor='rgba(0,100,255,0.2)',
                                fill='tonexty',
                                name='Confidence Interval',
                                hoverinfo='skip'
                            ))
                            
                            fig.update_layout(
                                title="📈 Forecast Projection",
                                xaxis_title="Date",
                                yaxis_title="Forecasted Value",
                                height=400,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Display drill-down analysis
                    if 'why_result' in message and message['why_result'].get('breakdowns'):
                        st.subheader("🕵️ Root Cause Analysis")
                        st.info(message['why_result']['summary'])
                        
                        for breakdown in message['why_result']['breakdowns']:
                            with st.expander(f"🔍 {breakdown['dimension'].title()} Analysis"):
                                for insight in breakdown['insights']:
                                    st.write(f"• **{insight['segment']}**: {insight['performance']} vs average ({insight['share']} of total)")
                
                # Fallback to standard visualizations if analytics not available
                elif 'visualizations' in message and message['visualizations']:
                    st.subheader("📊 Visualizations")
                    for viz in message['visualizations']:
                        st.plotly_chart(viz, use_container_width=True)
    
    # Chat input
    user_question = st.chat_input("Ask a question about your data...")
    
    if user_question and api_key:
        # Add user message to chat
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_question
        })
        
        with st.spinner("Analyzing your question and generating SQL..."):
            # Generate SQL from natural language
            sql_query = st.session_state.db_assistant.generate_sql_from_natural_language(
                user_question, api_key, ai_provider
            )
            
            if sql_query.startswith("Error"):
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': sql_query
                })
            else:
                # Execute query
                df, status = st.session_state.db_assistant.execute_query(sql_query)
                
                if df is not None:
                    # Enhanced Analytics Integration
                    if ANALYTICS_AVAILABLE:
                        # Get analytics integrator and process results
                        integrator = get_analytics_integrator()
                        
                        # Check for special question types
                        forecast_result = integrator.handle_forecasting_request(df, user_question)
                        why_result = integrator.handle_why_question(df, user_question, sql_query)
                        
                        # Get enhanced analytics results
                        analytics_results = integrator.enhance_query_results(
                            df, user_question, enable_insights=True
                        )
                        
                        # Generate AI-powered insights (legacy method as backup)
                        ai_insights = st.session_state.db_assistant.generate_insights(
                            df, user_question, api_key, ai_provider
                        )
                        
                        # Combine insights
                        enhanced_content = []
                        enhanced_content.append(f"📊 **Analysis Results** ({len(df):,} rows)")
                        
                        if analytics_results.get('insights'):
                            enhanced_content.append(f"🔍 **Quick Insights:** {analytics_results['insights']}")
                        
                        if ai_insights:
                            enhanced_content.append(f"🤖 **AI Analysis:** {ai_insights}")
                            
                        if forecast_result.get('summary'):
                            enhanced_content.append(f"📈 **Forecast:** {forecast_result['summary']}")
                            
                        if why_result.get('summary'):
                            enhanced_content.append(f"🕵️ **Root Cause:** {why_result['summary']}")
                        
                        response_content = "\n\n".join(enhanced_content)
                        
                        # Create assistant message with enhanced analytics
                        assistant_message = {
                            'type': 'assistant',
                            'content': response_content,
                            'sql': sql_query,
                            'dataframe': df,
                            'analytics_results': analytics_results,
                            'forecast_result': forecast_result,
                            'why_result': why_result,
                            'visualizations': analytics_results.get('visualizations', [])
                        }
                        
                    else:
                        # Fallback to original method if analytics unavailable
                        insights = st.session_state.db_assistant.generate_insights(
                            df, user_question, api_key, ai_provider
                        )
                        
                        visualizations = st.session_state.db_assistant.create_visualizations(
                            df, user_question
                        )
                        
                        response_content = f"{insights}\n\nFound {len(df)} rows of data."
                        
                        assistant_message = {
                            'type': 'assistant',
                            'content': response_content,
                            'sql': sql_query,
                            'dataframe': df,
                            'visualizations': visualizations
                        }
                    
                    st.session_state.chat_history.append(assistant_message)
                else:
                    st.session_state.chat_history.append({
                        'type': 'assistant',
                        'content': f"I encountered an error: {status}",
                        'sql': sql_query
                    })
        
        st.rerun()

if __name__ == "__main__":
    main()
