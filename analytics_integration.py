"""
Analytics Integration Module
Integrates the Advanced Analytics Engine with the main Streamlit app
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from advanced_analytics import get_analytics_engine


class AnalyticsIntegrator:
    """Handles integration of analytics features with the main app"""
    
    def __init__(self):
        self.engine = get_analytics_engine()
        
    def enhance_query_results(self, df: pd.DataFrame, user_question: str = "", enable_insights: bool = True) -> Dict[str, Any]:
        """
        Main integration point - enhances query results with analytics
        Returns dict with visualizations, insights, and anomalies
        """
        if df.empty:
            return {
                "visualizations": [],
                "insights": "No data available for analysis.",
                "anomalies": {},
                "forecast": {}
            }
        
        results = {
            "visualizations": [],
            "insights": "",
            "anomalies": {},
            "forecast": {}
        }
        
        # Generate visualizations
        with st.spinner("🎨 Creating visualizations..."):
            results["visualizations"] = self.engine.recommend_visualization(
                df, 
                user_hint=user_question,
                question=user_question
            )
        
        # Generate insights if enabled
        if enable_insights:
            with st.spinner("🧠 Generating insights..."):
                results["insights"] = self.engine.explain_in_plain_english(df, user_question)
        
        # Detect anomalies for time series data
        date_cols = self._detect_date_columns(df)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(date_cols) > 0 and len(numeric_cols) > 0 and len(df) > 10:
            with st.spinner("🔍 Detecting anomalies..."):
                results["anomalies"] = self.engine.detect_anomalies(
                    df, 
                    date_cols[0], 
                    numeric_cols[0]
                )
        
        return results
    
    def handle_forecasting_request(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Handle forecasting requests from user questions"""
        forecast_keywords = ['forecast', 'predict', 'future', 'next', 'will be', 'projection']
        
        if not any(keyword in question.lower() for keyword in forecast_keywords):
            return {}
        
        date_cols = self._detect_date_columns(df)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(date_cols) == 0 or len(numeric_cols) == 0:
            return {"summary": "Forecasting requires time series data (date + numeric columns)"}
        
        # Extract forecast period from question
        periods = self._extract_forecast_period(question)
        
        with st.spinner("📈 Generating forecast..."):
            forecast_result = self.engine.generate_forecast(
                df, 
                date_cols[0], 
                numeric_cols[0], 
                periods
            )
        
        return forecast_result
    
    def handle_why_question(self, df: pd.DataFrame, question: str, original_query: str = "") -> Dict[str, Any]:
        """Handle 'why' questions with drill-down analysis"""
        why_keywords = ['why', 'reason', 'cause', 'driver', 'what caused', 'explain']
        
        if not any(keyword in question.lower() for keyword in why_keywords):
            return {}
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) == 0:
            return {"summary": "Why analysis requires numeric metrics to analyze"}
        
        # Use first numeric column as the metric to analyze
        metric_col = numeric_cols[0]
        
        with st.spinner("🕵️ Analyzing root causes..."):
            drilldown_result = self.engine.ask_why_drilldown(df, metric_col)
        
        return drilldown_result
    
    def render_analytics_results(self, results: Dict[str, Any], show_raw_data: bool = True):
        """Render all analytics results in Streamlit UI"""
        
        # Display visualizations
        if results.get("visualizations"):
            st.subheader("📊 Visualizations")
            
            # Create columns for multiple charts
            num_charts = len(results["visualizations"])
            if num_charts == 1:
                st.plotly_chart(results["visualizations"][0], use_container_width=True)
            elif num_charts == 2:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(results["visualizations"][0], use_container_width=True)
                with col2:
                    st.plotly_chart(results["visualizations"][1], use_container_width=True)
            else:
                # Show first chart full width, others in columns
                st.plotly_chart(results["visualizations"][0], use_container_width=True)
                if num_charts > 1:
                    cols = st.columns(min(num_charts - 1, 2))
                    for i, chart in enumerate(results["visualizations"][1:]):
                        with cols[i % len(cols)]:
                            st.plotly_chart(chart, use_container_width=True)
        
        # Display insights
        if results.get("insights"):
            st.subheader("💡 Key Insights")
            st.info(results["insights"])
        
        # Display anomalies
        if results.get("anomalies") and results["anomalies"].get("anomalies"):
            st.subheader("⚠️ Anomaly Detection")
            st.warning(results["anomalies"]["summary"])
            
            # Show anomaly details
            with st.expander("View Anomaly Details"):
                for anomaly in results["anomalies"]["anomalies"][:5]:  # Show top 5
                    severity_color = "🔴" if anomaly["severity"] == "high" else "🟡"
                    st.write(f"{severity_color} **{anomaly['date']}**: {anomaly['value']:,.1f} "
                            f"({anomaly['z_score']:.1f}σ from normal)")
        
        # Display forecast
        if results.get("forecast") and results["forecast"].get("forecast"):
            st.subheader("📈 Forecast")
            st.success(results["forecast"]["summary"])
            
            # Create forecast visualization
            forecast_chart = self._create_forecast_chart(results["forecast"])
            if forecast_chart:
                st.plotly_chart(forecast_chart, use_container_width=True)
            
            # Show forecast data table
            with st.expander("View Forecast Data"):
                forecast_df = pd.DataFrame(results["forecast"]["forecast"])
                st.dataframe(forecast_df.head(10))
        
        # Display drill-down analysis
        if results.get("breakdowns"):
            st.subheader("🔍 Root Cause Analysis")
            st.info(results["summary"])
            
            for breakdown in results["breakdowns"]:
                st.write(f"**{breakdown['dimension'].title()} Analysis:**")
                for insight in breakdown["insights"]:
                    st.write(f"• {insight['insight']}")
                st.write("---")
    
    def add_analytics_controls(self):
        """Add analytics control widgets to sidebar"""
        st.sidebar.subheader("🎛️ Analytics Controls")
        
        # Toggle switches for features
        enable_insights = st.sidebar.toggle("Auto Insights", value=True, help="Generate plain-English insights")
        enable_anomaly = st.sidebar.toggle("Anomaly Detection", value=True, help="Detect unusual patterns")
        enable_forecast = st.sidebar.toggle("Smart Forecasting", value=True, help="Enable forecasting for time series")
        
        # Advanced settings
        with st.sidebar.expander("Advanced Settings"):
            anomaly_sensitivity = st.slider("Anomaly Sensitivity", 1.5, 3.0, 2.0, 0.1, 
                                          help="Lower = more sensitive to anomalies")
            forecast_periods = st.number_input("Forecast Days", 7, 365, 30, 
                                             help="Number of days to forecast")
            max_charts = st.number_input("Max Charts", 1, 5, 3, 
                                       help="Maximum number of charts to generate")
        
        return {
            "enable_insights": enable_insights,
            "enable_anomaly": enable_anomaly,
            "enable_forecast": enable_forecast,
            "anomaly_sensitivity": anomaly_sensitivity,
            "forecast_periods": forecast_periods,
            "max_charts": max_charts
        }
    
    def _detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Helper to detect date columns"""
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                date_cols.append(col)
            elif df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(5))
                    date_cols.append(col)
                except:
                    pass
        return date_cols
    
    def _extract_forecast_period(self, question: str) -> int:
        """Extract forecast period from user question"""
        import re

        # Look for patterns like "next 30 days", "next month", etc.
        patterns = {
            r'(\d+)\s*days?': lambda x: int(x),
            r'(\d+)\s*weeks?': lambda x: int(x) * 7,
            r'(\d+)\s*months?': lambda x: int(x) * 30,
            r'next\s*month': lambda x: 30,
            r'next\s*quarter': lambda x: 90,
            r'next\s*year': lambda x: 365
        }
        
        question_lower = question.lower()
        
        for pattern, converter in patterns.items():
            match = re.search(pattern, question_lower)
            if match:
                if match.groups():
                    return converter(match.group(1))
                else:
                    return converter(None)
        
        return 30  # Default to 30 days
    
    def _create_forecast_chart(self, forecast_data: Dict[str, Any]) -> Optional[go.Figure]:
        """Create visualization for forecast results"""
        try:
            if not forecast_data.get("forecast"):
                return None
            
            forecast_df = pd.DataFrame(forecast_data["forecast"])
            
            fig = go.Figure()
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                mode='lines+markers',
                name='Forecast',
                line=dict(dash='dash', width=3, color='blue'),
                marker=dict(size=6)
            ))
            
            # Add confidence interval
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
                title=f"📈 Forecast ({forecast_data.get('model', 'Unknown')} Model)",
                xaxis_title="Date",
                yaxis_title="Forecasted Value",
                height=400,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating forecast chart: {str(e)}")
            return None

# Global integrator instance
analytics_integrator = AnalyticsIntegrator()

def get_analytics_integrator():
    """Get the global analytics integrator instance"""
    return analytics_integrator
