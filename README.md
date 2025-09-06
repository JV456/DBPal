# 🧠 DBPal — Chat Naturally with Your Database

> **Ask questions in plain English. Get instant SQL, visualizations, insights, and forecasts — no coding needed.**

DBPal is an **AI-powered Database Chat & Analytics Assistant** built with Streamlit. Connect to any SQL database (PostgreSQL, MySQL, SQL Server), ask natural language questions like *“Show me sales by region last month”*, and get:

✅ Auto-generated SQL  
✅ Interactive charts (Plotly/Altair)  
✅ Anomaly & trend detection  
✅ Forecasting (Prophet/ARIMA)  
✅ Plain-English insights & “Ask Why?” root-cause analysis  
✅ Export results to CSV/Excel  
✅ Multi-LLM support (OpenAI, Anthropic, etc.) — pick your AI engine

Perfect for analysts, product managers, and non-technical users who want to **unlock insights without writing a single line of SQL**.

---

## ✨ Features

- **Natural Language to SQL** — powered by LangChain + LLMs
- **Auto-Visualization** — smart chart recommendations based on data shape
- **Anomaly Detection** — Z-score, moving average, trend shifts
- **Forecasting Engine** — Prophet-based predictions with confidence intervals
- **“Ask Why?” Drill-Down** — automated root-cause analysis for metric changes
- **Plain-English Summaries** — “Top 3 products drive 68% of revenue”
- **Multi-LLM Support** — switch between OpenAI, Anthropic, local models
- **SQL Transparency Toggle** — see and learn the queries behind answers
- **Secure Connection** — credentials never logged or exposed
- **Export Data** — download results as CSV or Excel
- **Sample Questions** — get started instantly
- **Advanced Mode Toggle** — unlock JOINs, window functions, complex aggregations

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Database Connectivity**: SQLAlchemy, psycopg2, mysql-connector-python
- **NL-to-SQL & AI**: LangChain, OpenAI, Anthropic, LlamaIndex
- **Analytics & Forecasting**: Pandas, Prophet, Statsmodels
- **Visualization**: Plotly Express, Altair
- **Caching & Performance**: Streamlit cache, async analysis
- **Security**: Environment variables, input sanitization, query validation

---

## Project Structure

```
DBPal/
├── app.py                      # Main Streamlit application
├── advanced_analytics.py       # Core analytics engine
├── analytics_integration.py    # Database integration layer
├── analytics_prompts.py        # Pre-defined analytical prompts
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

---

*DBPal - Your intelligent database companion for advanced analytics*