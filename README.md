# AI Financial Analyst Agent

A Streamlit-based financial research assistant using a Groq-hosted language model and Yahoo Finance data. Features include stock research chat, ticker discovery and validation, fundamentals comparisons, historical price charts, downloadable reports, portfolio tracking, and threshold alerts.

> Financial data can be delayed or unavailable. This application is for research and educational use, not personalized investment advice.

## Features

- Conversational finance research assistant
- Ticker discovery and exchange-aware suggestions
- Fundamental comparisons and CSV export
- Historical price and normalized performance charts
- Markdown research report export
- Portfolio value and unrealized gain/loss calculations
- Price-threshold alerts checked when the app runs/refreshes
- Data source notes and verification status

## Run locally

1. Install Python 3.11 or another compatible supported version.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your credentials:

   ```dotenv
   GROQ_API_KEY=put_your_groq_api_key_here
   OPENFIGI_API_KEY=put_your_openfigi_key_here
   ```

   `OPENFIGI_API_KEY` is optional. Never commit `.env` to GitHub.

5. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

## Deploy with Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open <https://share.streamlit.io/> and create an app from the repository.
3. Set the entrypoint file to `app.py` and choose Python 3.11 in Advanced settings if available.
4. In the app's Secrets field, add your credentials in TOML format (do not add them to the repository):

   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   OPENFIGI_API_KEY = "your_openfigi_api_key"
   ```

   The OpenFIGI key is optional. Deploy the app, then review the build logs if it fails.

## Security and limitations

- `.env`, `venv/`, and Streamlit secrets files are excluded by `.gitignore`.
- Chat history, portfolio holdings, and alert rules use Streamlit session state and are not permanent database storage.
- Price alerts are not background notifications; they are checked when the app executes or refreshes.
- Ticker discovery and financial data depend on external services and are not guaranteed to cover every market or listing.
