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

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.8+** (recommended: Python 3.10)
- **Git** for cloning the repository
- **Database access** (PostgreSQL, MySQL, or SQL Server)
- **API keys** for LLM providers (OpenAI, Anthropic, etc.)

### Quick Install

1. **Clone the repository**
   ```bash
   git clone https://github.com/JV456/DBPal.git
   cd DBPal
   ```

2. **Create virtual environment**
   ```bash
   # Using venv
   python -m venv dbpal-env
   
   # Activate on Windows
   dbpal-env\Scripts\activate
   
   # Activate on macOS/Linux
   source dbpal-env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

---

## 📝 Example Usage and Output
---

<img width="1919" height="869" alt="Screenshot 2025-09-06 154944" src="https://github.com/user-attachments/assets/e9be4888-31d3-47b3-b498-62b6bacf5542" />

---

<img width="1632" height="653" alt="Screenshot 2025-09-06 154741" src="https://github.com/user-attachments/assets/21e60e96-20ae-418e-8b6d-07f6aa5206bb" />

---

<img width="1632" height="504" alt="Screenshot 2025-09-06 154758" src="https://github.com/user-attachments/assets/62858140-39f1-40c5-83b8-abe0d291b5df" />

---

<img width="1634" height="272" alt="Screenshot 2025-09-06 154816" src="https://github.com/user-attachments/assets/ce3acbc9-7770-46de-a2f9-6793b3750fe3" />

---

<img width="1675" height="746" alt="Screenshot 2025-09-06 154911" src="https://github.com/user-attachments/assets/e1ac1d93-f50a-4156-ad74-0dbe7e3b81a8" />

---

*DBPal - Your intelligent database companion for advanced analytics*
