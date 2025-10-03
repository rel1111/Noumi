import json
import pyodbc
import pandas as pd
from pathlib import Path

class SQLServerConnection:
    def __init__(self, config_file='config.json'):
        """Initialize connection with config file"""
        self.config_file = config_file
        self.connection = None
        self.load_config()
    
    def load_config(self):
        """Load database configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.db_config = config['database']
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{self.config_file}' not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file '{self.config_file}'")
    
    def get_connection_string(self):
        """Build connection string from config"""
        return (
            f"DRIVER={{{self.db_config['driver']}}};"
            f"SERVER={self.db_config['server']};"
            f"DATABASE={self.db_config['database_name']};"
            f"UID={self.db_config['username']};"
            f"PWD={self.db_config['password']};"
            f"PORT={self.db_config['port']};"
            f"TrustServerCertificate=yes;"
        )
    
    def connect(self):
        """Establish connection to SQL Server"""
        try:
            connection_string = self.get_connection_string()
            self.connection = pyodbc.connect(connection_string)
            print("✅ Successfully connected to SQL Server!")
            return self.connection
        except pyodbc.Error as e:
            print(f"❌ Error connecting to database: {e}")
            raise
    
    def execute_query(self, query):
        """Execute a SELECT query and return results as DataFrame"""
        if not self.connection:
            self.connect()
        
        try:
            df = pd.read_sql(query, self.connection)
            return df
        except Exception as e:
            print(f"❌ Error executing query: {e}")
            raise
    
    def execute_non_query(self, query):
        """Execute INSERT, UPDATE, DELETE queries"""
        if not self.connection:
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            self.connection.commit()
            print("✅ Query executed successfully!")
            return cursor.rowcount
        except Exception as e:
            print(f"❌ Error executing query: {e}")
            self.connection.rollback()
            raise
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("🔒 Database connection closed")

# Example usage
if __name__ == "__main__":
    # Initialize database connection
    db = SQLServerConnection('config.json')
    
    try:
        # Connect to database
        db.connect()
        
        # Example: Get list of tables
        tables_query = """
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE'
        """
        
        tables_df = db.execute_query(tables_query)
        print("Available tables:")
        print(tables_df)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always close the connection
        db.close()

