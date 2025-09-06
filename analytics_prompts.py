"""
AI Prompt Templates for Advanced Analytics
Optimized prompts for different AI providers to generate high-quality insights
"""

def get_insight_generation_prompt(df_description: str, question: str, data_sample: str) -> str:
    """
    Generate prompt for AI-powered insight generation
    """
    return f"""
You are a senior data analyst providing clear, actionable insights to business stakeholders. 

ANALYSIS CONTEXT:
- Dataset: {df_description}
- User Question: {question}
- Sample Data: {data_sample}

YOUR TASK: Provide 2-3 key business insights in plain English, following these guidelines:

GUIDELINES:
1. Focus on actionable patterns, not just statistics
2. Use specific numbers and percentages from the actual data
3. Identify trends, outliers, or business opportunities
4. Explain WHY patterns matter for business decisions
5. Keep insights concise but meaningful
6. Use emojis sparingly for visual emphasis

FORMAT:
• [Insight 1]: [Specific finding with numbers] - [Business implication]
• [Insight 2]: [Pattern or trend] - [Actionable recommendation]  
• [Insight 3]: [Notable observation] - [Strategic consideration]

EXAMPLE:
• Top 3 products generate 67% of revenue - consider premium pricing strategy
• Weekend sales spike 40% above weekdays - expand weekend marketing campaigns
• Customer satisfaction drops after $200+ purchases - investigate premium product quality

Remember: Base ALL insights on actual data patterns. Never make assumptions or add information not present in the data.

Generate insights now:
"""

def get_sql_enhancement_prompt(basic_sql: str, user_question: str, schema_context: str) -> str:
    """
    Generate prompt for enhancing SQL queries for better analytics
    """
    return f"""
You are an expert SQL analyst. Enhance the following SQL query to provide richer analytical insights.

ORIGINAL SQL: {basic_sql}
USER QUESTION: {user_question}
SCHEMA: {schema_context}

ENHANCEMENT OBJECTIVES:
1. Add relevant calculated columns (growth rates, percentages, rankings)
2. Include time-based analysis when applicable (YoY, MoM comparisons)
3. Add statistical measures (averages, medians, standard deviations)
4. Include segmentation and grouping for deeper insights
5. Ensure results are properly ordered and limited for analysis

RULES:
- Maintain the core intent of the original query
- Add analytical value without over-complicating
- Use window functions for rankings and comparisons
- Include date functions for temporal analysis
- Limit results to manageable sizes (TOP 50 or reasonable LIMIT)
- Add meaningful column aliases

OUTPUT: Enhanced SQL query only (no explanations)
"""

def get_anomaly_explanation_prompt(anomaly_data: dict, context: str) -> str:
    """
    Generate prompt for explaining detected anomalies
    """
    return f"""
You are a data scientist explaining anomalies to business users.

ANOMALY DETECTED:
- Date: {anomaly_data.get('date', 'Unknown')}
- Value: {anomaly_data.get('value', 'Unknown')}
- Expected: {anomaly_data.get('expected', 'Unknown')}
- Z-Score: {anomaly_data.get('z_score', 'Unknown')}
- Context: {context}

TASK: Provide a clear, non-technical explanation of this anomaly in 1-2 sentences.

GUIDELINES:
1. Explain what happened in business terms
2. Quantify the deviation (e.g., "40% above normal")
3. Suggest possible causes or actions
4. Keep language simple and actionable

EXAMPLES:
- "Sales spiked 150% on Black Friday - likely due to promotional campaign success"
- "Website traffic dropped 60% on March 15 - investigate potential technical issues"
- "Customer signups increased 80% last week - marketing campaign performing exceptionally"

Your explanation:
"""

def get_forecast_explanation_prompt(forecast_data: dict, model_type: str, historical_context: str) -> str:
    """
    Generate prompt for explaining forecast results
    """
    return f"""
You are a business analyst explaining predictive insights to stakeholders.

FORECAST RESULTS:
- Model: {model_type}
- Forecast Period: {forecast_data.get('periods', 'Unknown')} days
- Average Prediction: {forecast_data.get('avg_forecast', 'Unknown')}
- Confidence Range: {forecast_data.get('confidence_range', 'Unknown')}
- Historical Context: {historical_context}

TASK: Provide a clear business explanation of the forecast in 1-2 sentences.

GUIDELINES:
1. Translate predictions into business terms
2. Mention confidence level and uncertainty
3. Relate to historical performance
4. Suggest implications for planning
5. Keep language accessible and actionable

EXAMPLES:
- "Revenue projected to grow 15% next month ($120K ± $15K) based on seasonal trends"
- "Customer acquisition expected to remain stable at 500±50 new customers weekly"
- "Sales forecast shows potential 25% decline - consider accelerating marketing efforts"

Your explanation:
"""

def get_root_cause_prompt(metric_change: str, breakdown_data: list, business_context: str) -> str:
    """
    Generate prompt for root cause analysis explanations
    """
    breakdown_summary = "\n".join([f"- {item['dimension']}: {item['impact']}" for item in breakdown_data])
    
    return f"""
You are a business analyst conducting root cause analysis.

SITUATION:
- Metric Change: {metric_change}
- Business Context: {business_context}

CONTRIBUTING FACTORS:
{breakdown_summary}

TASK: Provide a clear explanation of the main drivers behind this change.

GUIDELINES:
1. Rank factors by impact/importance
2. Explain the business implications
3. Suggest specific actions or investigations
4. Use bullet points for clarity
5. Focus on actionable insights

FORMAT:
🔍 **Primary Drivers:**
• [Factor 1]: [Impact] - [Explanation and recommendation]
• [Factor 2]: [Impact] - [Business implication]

📋 **Recommended Actions:**
• [Specific action 1]
• [Investigation to conduct]

Your analysis:
"""

# Chart-specific prompts for different visualization types
CHART_INSIGHTS_PROMPTS = {
    'time_series': """
    Analyze this time series chart and provide insights on:
    1. Overall trends (growth, decline, seasonality)
    2. Notable peaks, dips, or inflection points
    3. Cyclical patterns or seasonal effects
    4. Business implications and recommendations
    """,
    
    'bar_chart': """
    Analyze this categorical comparison and provide insights on:
    1. Top and bottom performers
    2. Significant gaps between categories
    3. Concentration patterns (80/20 rule)
    4. Opportunities for rebalancing or growth
    """,
    
    'pie_chart': """
    Analyze this distribution chart and provide insights on:
    1. Dominant segments and their share
    2. Market concentration risks
    3. Small segments with growth potential
    4. Portfolio balance recommendations
    """,
    
    'scatter_plot': """
    Analyze this correlation chart and provide insights on:
    1. Strength and direction of relationships
    2. Outliers and their significance
    3. Clustering patterns
    4. Predictive or strategic implications
    """
}

def get_chart_insight_prompt(chart_type: str, data_context: str) -> str:
    """
    Get specific prompt for chart-based insights
    """
    base_prompt = CHART_INSIGHTS_PROMPTS.get(chart_type, CHART_INSIGHTS_PROMPTS['bar_chart'])
    
    return f"""
{base_prompt}

DATA CONTEXT: {data_context}

Provide 2-3 specific, actionable insights based on the visualization patterns you can infer from the data.
"""
