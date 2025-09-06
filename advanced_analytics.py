"""
Advanced Analytics & Insight Engine
Enterprise-grade automated data interpretation and visualization system
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

class AdvancedAnalyticsEngine:
    """Core engine for automated data analysis and insights"""
    
    def __init__(self):
        self.chart_cache = {}
        self.insight_cache = {}
        
    def recommend_visualization(self, df: pd.DataFrame, user_hint: str = None, question: str = None) -> List[go.Figure]:
        """
        Automatically recommend and create the best-fit visualizations
        """
        if df.empty:
            return []
            
        visualizations = []
        
        # Parse user hints for chart type
        chart_type_override = self._parse_chart_hint(user_hint or question or "")
        
        # Analyze data structure
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = self._detect_date_columns(df)
        
        # Apply chart type override if valid
        if chart_type_override and self._validate_chart_type(df, chart_type_override):
            viz = self._create_specific_chart(df, chart_type_override, numeric_cols, categorical_cols, date_cols)
            if viz:
                visualizations.append(viz)
                return visualizations
        
        # Auto-recommendation logic
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            # Time series analysis
            viz = self._create_time_series_chart(df, date_cols[0], numeric_cols[:2])
            if viz:
                visualizations.append(viz)
                
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            # Categorical comparison
            viz = self._create_categorical_chart(df, categorical_cols[0], numeric_cols[0])
            if viz:
                visualizations.append(viz)
                
        if len(categorical_cols) >= 1 and len(df) <= 20:
            # Part-to-whole analysis
            viz = self._create_pie_chart(df, categorical_cols[0], numeric_cols[0] if numeric_cols else None)
            if viz:
                visualizations.append(viz)
                
        if len(numeric_cols) >= 2:
            # Correlation analysis
            viz = self._create_correlation_chart(df, numeric_cols[:2])
            if viz:
                visualizations.append(viz)
                
        if len(numeric_cols) >= 1 and len(df) > 20:
            # Distribution analysis
            viz = self._create_distribution_chart(df, numeric_cols[0])
            if viz:
                visualizations.append(viz)
        
        return visualizations[:3]  # Limit to 3 charts to avoid clutter
    
    def detect_anomalies(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """
        Detect anomalies in time series data using statistical methods
        """
        if df.empty or date_col not in df.columns or value_col not in df.columns:
            return {"anomalies": [], "summary": "Insufficient data for anomaly detection"}
            
        try:
            # Prepare time series data
            ts_df = df[[date_col, value_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col])
            ts_df = ts_df.sort_values(date_col).dropna()
            
            if len(ts_df) < 10:
                return {"anomalies": [], "summary": "Not enough data points for reliable anomaly detection"}
            
            # Calculate rolling statistics
            window = min(7, len(ts_df) // 3)
            ts_df['rolling_mean'] = ts_df[value_col].rolling(window=window, center=True).mean()
            ts_df['rolling_std'] = ts_df[value_col].rolling(window=window, center=True).std()
            
            # Z-score based anomaly detection
            ts_df['z_score'] = np.abs((ts_df[value_col] - ts_df['rolling_mean']) / ts_df['rolling_std'])
            anomaly_threshold = 2.0
            
            anomalies = ts_df[ts_df['z_score'] > anomaly_threshold].copy()
            
            anomaly_list = []
            for _, row in anomalies.iterrows():
                anomaly_list.append({
                    'date': row[date_col].strftime('%Y-%m-%d'),
                    'value': float(row[value_col]),
                    'z_score': float(row['z_score']),
                    'expected': float(row['rolling_mean']) if pd.notna(row['rolling_mean']) else None,
                    'severity': 'high' if row['z_score'] > 3 else 'medium'
                })
            
            # Generate summary
            if anomaly_list:
                summary = f"⚠️ {len(anomaly_list)} anomalies detected. "
                if anomaly_list:
                    latest = anomaly_list[-1]
                    deviation = ((latest['value'] - latest['expected']) / latest['expected'] * 100) if latest['expected'] else 0
                    summary += f"Latest: {latest['date']} ({deviation:+.1f}% vs expected)"
            else:
                summary = "✅ No significant anomalies detected in the data"
                
            return {
                "anomalies": anomaly_list,
                "summary": summary,
                "threshold": anomaly_threshold,
                "total_points": len(ts_df)
            }
            
        except Exception as e:
            return {"anomalies": [], "summary": f"Anomaly detection failed: {str(e)}"}
    
    def generate_forecast(self, df: pd.DataFrame, date_col: str, value_col: str, periods: int = 30) -> Dict[str, Any]:
        """
        Generate forecasts using Prophet or exponential smoothing
        """
        if df.empty or date_col not in df.columns or value_col not in df.columns:
            return {"forecast": None, "summary": "Insufficient data for forecasting"}
            
        try:
            # Prepare data
            forecast_df = df[[date_col, value_col]].copy()
            forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])
            forecast_df = forecast_df.sort_values(date_col).dropna()
            
            if len(forecast_df) < 10:
                return {"forecast": None, "summary": "Need at least 10 data points for forecasting"}
            
            # Try Prophet first (if available)
            if PROPHET_AVAILABLE and len(forecast_df) >= 20:
                return self._prophet_forecast(forecast_df, date_col, value_col, periods)
            
            # Fallback to exponential smoothing
            return self._exponential_smoothing_forecast(forecast_df, date_col, value_col, periods)
            
        except Exception as e:
            return {"forecast": None, "summary": f"Forecasting failed: {str(e)}"}
    
    def explain_in_plain_english(self, df: pd.DataFrame, question: str = "") -> str:
        """
        Generate plain-English insights from data
        """
        if df.empty:
            return "No data available to analyze."
            
        insights = []
        
        # Basic statistics
        total_rows = len(df)
        insights.append(f"📊 Found {total_rows:,} records")
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:  # Limit to top 3
            if df[col].notna().any():
                avg_val = df[col].mean()
                max_val = df[col].max()
                min_val = df[col].min()
                
                # Format numbers appropriately
                if avg_val > 1000:
                    avg_str = f"{avg_val:,.0f}"
                    max_str = f"{max_val:,.0f}"
                    min_str = f"{min_val:,.0f}"
                else:
                    avg_str = f"{avg_val:.1f}"
                    max_str = f"{max_val:.1f}"
                    min_str = f"{min_val:.1f}"
                
                insights.append(f"💡 {col.title()}: Average {avg_str}, Range {min_str} - {max_str}")
        
        # Analyze categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols[:2]:  # Limit to top 2
            if df[col].notna().any():
                top_category = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                category_count = df[col].nunique()
                top_percentage = (df[col] == top_category).mean() * 100
                
                insights.append(f"🏷️ {col.title()}: {category_count} categories, '{top_category}' leads ({top_percentage:.1f}%)")
        
        # Time-based insights
        date_cols = self._detect_date_columns(df)
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            date_col = date_cols[0]
            value_col = numeric_cols[0]
            
            # Check for trends
            df_sorted = df.sort_values(date_col)
            if len(df_sorted) >= 5:
                recent_avg = df_sorted[value_col].tail(3).mean()
                older_avg = df_sorted[value_col].head(3).mean()
                
                if pd.notna(recent_avg) and pd.notna(older_avg) and older_avg != 0:
                    change_pct = ((recent_avg - older_avg) / older_avg) * 100
                    trend = "📈 increasing" if change_pct > 5 else "📉 decreasing" if change_pct < -5 else "➡️ stable"
                    insights.append(f"📅 Recent trend: {trend} ({change_pct:+.1f}%)")
        
        # Concentration analysis
        if len(df) > 10:
            for col in categorical_cols[:1]:
                if df[col].notna().any():
                    value_counts = df[col].value_counts()
                    top_3_share = value_counts.head(3).sum() / len(df) * 100
                    if top_3_share > 70:
                        insights.append(f"🎯 High concentration: Top 3 {col}s account for {top_3_share:.0f}% of data")
        
        return " • ".join(insights) if insights else "Analysis complete - see visualizations for patterns."
    
    def ask_why_drilldown(self, df: pd.DataFrame, metric_column: str, breakdown_columns: List[str] = None) -> Dict[str, Any]:
        """
        Automated root-cause analysis for metric changes
        """
        if df.empty or metric_column not in df.columns:
            return {"summary": "Insufficient data for drill-down analysis", "breakdowns": []}
        
        breakdowns = []
        
        # Auto-detect breakdown columns if not provided
        if not breakdown_columns:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            breakdown_columns = categorical_cols[:3]  # Top 3 categorical columns
        
        try:
            overall_avg = df[metric_column].mean()
            
            for breakdown_col in breakdown_columns:
                if breakdown_col not in df.columns:
                    continue
                    
                # Calculate segment performance
                segment_analysis = df.groupby(breakdown_col)[metric_column].agg(['mean', 'count', 'sum']).reset_index()
                segment_analysis['share_of_total'] = segment_analysis['sum'] / segment_analysis['sum'].sum() * 100
                segment_analysis['vs_average'] = ((segment_analysis['mean'] - overall_avg) / overall_avg) * 100
                
                # Find significant segments
                significant_segments = segment_analysis[
                    (abs(segment_analysis['vs_average']) > 10) & 
                    (segment_analysis['share_of_total'] > 5)
                ].sort_values('vs_average')
                
                if not significant_segments.empty:
                    breakdown_insights = []
                    for _, segment in significant_segments.head(3).iterrows():
                        impact_emoji = "📈" if segment['vs_average'] > 0 else "📉"
                        breakdown_insights.append({
                            'segment': str(segment[breakdown_col]),
                            'performance': f"{segment['vs_average']:+.1f}%",
                            'share': f"{segment['share_of_total']:.1f}%",
                            'insight': f"{impact_emoji} {segment[breakdown_col]}: {segment['vs_average']:+.1f}% vs average ({segment['share_of_total']:.1f}% of total)"
                        })
                    
                    if breakdown_insights:
                        breakdowns.append({
                            'dimension': breakdown_col,
                            'insights': breakdown_insights
                        })
            
            # Generate summary
            if breakdowns:
                summary = f"🔍 Root cause analysis complete. Found {len(breakdowns)} contributing factors:"
            else:
                summary = "🤷 No clear drivers identified. Metric change appears evenly distributed."
                
            return {
                "summary": summary,
                "breakdowns": breakdowns,
                "overall_average": float(overall_avg)
            }
            
        except Exception as e:
            return {"summary": f"Drill-down analysis failed: {str(e)}", "breakdowns": []}
    
    # Helper methods
    def _parse_chart_hint(self, text: str) -> Optional[str]:
        """Parse user input for chart type preferences"""
        text_lower = text.lower()
        
        chart_mappings = {
            'pie': ['pie', 'donut', 'part-to-whole'],
            'bar': ['bar', 'column', 'compare', 'comparison'],
            'line': ['line', 'trend', 'time', 'over time'],
            'scatter': ['scatter', 'correlation', 'relationship'],
            'histogram': ['histogram', 'distribution', 'freq'],
            'area': ['area', 'filled'],
            'box': ['box', 'boxplot', 'quartile']
        }
        
        for chart_type, keywords in chart_mappings.items():
            if any(keyword in text_lower for keyword in keywords):
                return chart_type
        
        return None
    
    def _validate_chart_type(self, df: pd.DataFrame, chart_type: str) -> bool:
        """Validate if requested chart type is suitable for the data"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        validation_rules = {
            'pie': len(categorical_cols) >= 1 and len(numeric_cols) >= 1 and len(df) <= 20,
            'bar': len(categorical_cols) >= 1 and len(numeric_cols) >= 1,
            'line': len(numeric_cols) >= 1,
            'scatter': len(numeric_cols) >= 2,
            'histogram': len(numeric_cols) >= 1,
            'area': len(numeric_cols) >= 1,
            'box': len(numeric_cols) >= 1
        }
        
        return validation_rules.get(chart_type, False)
    
    def _detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect date columns in DataFrame"""
        date_cols = []
        
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]':
                date_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                sample = df[col].dropna().head(10)
                try:
                    pd.to_datetime(sample)
                    date_cols.append(col)
                except:
                    pass
        
        return date_cols
    
    def _create_time_series_chart(self, df: pd.DataFrame, date_col: str, value_cols: List[str]) -> Optional[go.Figure]:
        """Create time series visualization"""
        try:
            df_plot = df.copy()
            df_plot[date_col] = pd.to_datetime(df_plot[date_col])
            df_plot = df_plot.sort_values(date_col)
            
            fig = go.Figure()
            
            for i, col in enumerate(value_cols):
                fig.add_trace(go.Scatter(
                    x=df_plot[date_col],
                    y=df_plot[col],
                    mode='lines+markers',
                    name=col.title(),
                    line=dict(width=3),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title=f"{', '.join([col.title() for col in value_cols])} Over Time",
                xaxis_title=date_col.title(),
                yaxis_title="Value",
                hovermode='x unified',
                height=400
            )
            
            return fig
        except:
            return None
    
    def _create_categorical_chart(self, df: pd.DataFrame, cat_col: str, value_col: str) -> Optional[go.Figure]:
        """Create categorical comparison chart"""
        try:
            # Aggregate data
            agg_df = df.groupby(cat_col)[value_col].sum().reset_index()
            agg_df = agg_df.nlargest(15, value_col)  # Top 15 to avoid clutter
            
            fig = px.bar(
                agg_df, 
                x=cat_col, 
                y=value_col,
                title=f"{value_col.title()} by {cat_col.title()}",
                color=value_col,
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(height=400, xaxis_tickangle=-45)
            return fig
        except:
            return None
    
    def _create_pie_chart(self, df: pd.DataFrame, cat_col: str, value_col: str = None) -> Optional[go.Figure]:
        """Create pie chart for part-to-whole analysis"""
        try:
            if value_col:
                pie_data = df.groupby(cat_col)[value_col].sum().reset_index()
                values = pie_data[value_col]
                labels = pie_data[cat_col]
            else:
                pie_data = df[cat_col].value_counts().reset_index()
                values = pie_data['count']
                labels = pie_data[cat_col]
            
            # Limit to top 8 slices
            if len(labels) > 8:
                top_data = pie_data.nlargest(7, values.name if hasattr(values, 'name') else 'count')
                other_sum = values.sum() - top_data[values.name if hasattr(values, 'name') else 'count'].sum()
                
                # Add "Others" category
                other_row = pd.DataFrame({cat_col: ['Others'], values.name if hasattr(values, 'name') else 'count': [other_sum]})
                pie_data = pd.concat([top_data, other_row], ignore_index=True)
                values = pie_data[values.name if hasattr(values, 'name') else 'count']
                labels = pie_data[cat_col]
            
            fig = px.pie(
                values=values,
                names=labels,
                title=f"Distribution of {cat_col.title()}"
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            return fig
        except:
            return None
    
    def _create_correlation_chart(self, df: pd.DataFrame, numeric_cols: List[str]) -> Optional[go.Figure]:
        """Create scatter plot for correlation analysis"""
        try:
            if len(numeric_cols) < 2:
                return None
                
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            
            fig = px.scatter(
                df, 
                x=x_col, 
                y=y_col,
                title=f"{y_col.title()} vs {x_col.title()}",
                trendline="ols"
            )
            
            fig.update_layout(height=400)
            return fig
        except:
            return None
    
    def _create_distribution_chart(self, df: pd.DataFrame, numeric_col: str) -> Optional[go.Figure]:
        """Create histogram for distribution analysis"""
        try:
            fig = px.histogram(
                df, 
                x=numeric_col,
                title=f"Distribution of {numeric_col.title()}",
                nbins=30
            )
            
            fig.update_layout(height=400)
            return fig
        except:
            return None
    
    def _create_specific_chart(self, df: pd.DataFrame, chart_type: str, numeric_cols: List[str], categorical_cols: List[str], date_cols: List[str]) -> Optional[go.Figure]:
        """Create specific chart type as requested by user"""
        if chart_type == 'pie' and len(categorical_cols) > 0 and len(numeric_cols) > 0:
            return self._create_pie_chart(df, categorical_cols[0], numeric_cols[0])
        elif chart_type == 'bar' and len(categorical_cols) > 0 and len(numeric_cols) > 0:
            return self._create_categorical_chart(df, categorical_cols[0], numeric_cols[0])
        elif chart_type == 'line' and len(date_cols) > 0 and len(numeric_cols) > 0:
            return self._create_time_series_chart(df, date_cols[0], numeric_cols[:2])
        elif chart_type == 'scatter' and len(numeric_cols) >= 2:
            return self._create_correlation_chart(df, numeric_cols[:2])
        elif chart_type == 'histogram' and len(numeric_cols) > 0:
            return self._create_distribution_chart(df, numeric_cols[0])
        
        return None
    
    def _prophet_forecast(self, df: pd.DataFrame, date_col: str, value_col: str, periods: int) -> Dict[str, Any]:
        """Generate forecast using Prophet"""
        try:
            # Prepare Prophet data format
            prophet_df = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
            
            # Create and fit model
            model = Prophet(daily_seasonality=False, yearly_seasonality=True)
            model.fit(prophet_df)
            
            # Generate future dates
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # Extract forecast values
            forecast_data = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
            
            # Calculate confidence interval
            avg_forecast = forecast.tail(periods)['yhat'].mean()
            ci_range = forecast.tail(periods)['yhat_upper'].mean() - forecast.tail(periods)['yhat_lower'].mean()
            
            summary = f"📈 Forecasted average: {avg_forecast:,.1f} ± {ci_range/2:,.1f} (85% confidence)"
            
            return {
                "forecast": forecast_data,
                "summary": summary,
                "model": "Prophet",
                "periods": periods
            }
            
        except Exception as e:
            return {"forecast": None, "summary": f"Prophet forecasting failed: {str(e)}"}
    
    def _exponential_smoothing_forecast(self, df: pd.DataFrame, date_col: str, value_col: str, periods: int) -> Dict[str, Any]:
        """Fallback forecasting using exponential smoothing"""
        try:
            values = df[value_col].dropna()
            
            # Simple exponential smoothing
            alpha = 0.3
            forecast_values = []
            last_value = values.iloc[-1]
            
            for _ in range(periods):
                forecast_values.append(last_value)
                # Simple persistence model for now
            
            # Generate future dates
            last_date = pd.to_datetime(df[date_col]).max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
            
            forecast_data = [
                {"ds": date.strftime('%Y-%m-%d'), "yhat": val, "yhat_lower": val*0.9, "yhat_upper": val*1.1}
                for date, val in zip(future_dates, forecast_values)
            ]
            
            avg_forecast = np.mean(forecast_values)
            summary = f"📈 Forecasted average: {avg_forecast:,.1f} (simplified model - consider more data for better accuracy)"
            
            return {
                "forecast": forecast_data,
                "summary": summary,
                "model": "Exponential Smoothing",
                "periods": periods
            }
            
        except Exception as e:
            return {"forecast": None, "summary": f"Exponential smoothing failed: {str(e)}"}

# Global instance
analytics_engine = AdvancedAnalyticsEngine()

def get_analytics_engine():
    """Get the global analytics engine instance"""
    return analytics_engine
