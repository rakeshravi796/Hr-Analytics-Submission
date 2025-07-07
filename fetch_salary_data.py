# fetch_salary_data.py

from sqlalchemy import create_engine
import pandas as pd

def fetch_salary_data(database_uri):
    """
    Connects to the PostgreSQL database and fetches the salary_data table.
    """
    try:
        engine = create_engine(database_uri)
        query = 'SELECT * FROM public.salary_data'
        df = pd.read_sql(query, con=engine)
        print(f"✅ Fetched {len(df)} rows.")
        return df

    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()
