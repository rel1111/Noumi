import json
import pandas as pd
from pathlib import Path

class SQLServerConnection:
    def __init__(self, config_file='config.json'):
        """Initialize with your new config format"""
        script_dir = Path(__file__).parent if '__file__' in globals() else Path('.')
        self.config_file = script_dir / config_file
        self.connection = None
        self.engine = None
        self.connection_type = None
        self.load_config()
    
    def load_config(self):
        """Load configuration with new format"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.db_config = config['database']
            print("✅ Config loaded successfully!")
            print(f"Server: {self.db_config['server']}:{self.db_config['port']}")
            print(f"Database: {self.db_config['database_name']}")
        except Exception as e:
            raise Exception(f"Error loading config: {e}")
    
    def try_pymssql_connection(self):
        """Try pymssql (no ODBC required)"""
        try:
            import pymssql
            
            print("Trying pymssql with SQL Authentication...")
            self.connection = pymssql.connect(
                server=self.db_config['server'],
                user=self.db_config['sql_auth']['username'],
                password=self.db_config['sql_auth']['password'],
                database=self.db_config['database_name'],
                port=self.db_config['port'],
                timeout=10,
                login_timeout=10
            )
            self.connection_type = "pymssql"
            print("✅ Successfully connected with pymssql!")
            return True
            
        except ImportError:
            print("❌ pymssql not installed. Install with: pip install pymssql")
            return False
        except Exception as e:
            print(f"❌ pymssql failed: {type(e).__name__}: {e}")
            return False
    
    def try_pyodbc_sql_auth(self):
        """Try pyodbc with SQL Authentication"""
        try:
            import pyodbc
            
            # Find available SQL Server drivers
            drivers = [d for d in pyodbc.drivers() if 'SQL Server' in d]
            if not drivers:
                print("❌ No SQL Server ODBC drivers found")
                return False
            
            driver = drivers[0]  # Use first available driver
            print(f"Trying pyodbc with SQL Auth using driver: {driver}")
            
            connection_string = (
                f"DRIVER={{{driver}}};"
                f"SERVER={self.db_config['server']},{self.db_config['port']};"
                f"DATABASE={self.db_config['database_name']};"
                f"UID={self.db_config['sql_auth']['username']};"
                f"PWD={self.db_config['sql_auth']['password']};"
                f"Connection Timeout=10;"
                f"Login Timeout=10;"
                f"TrustServerCertificate=yes;"
                f"Encrypt=no;"
            )
            
            self.connection = pyodbc.connect(connection_string)
            self.connection_type = f"pyodbc_sql_auth_{driver}"
            print(f"✅ Successfully connected with pyodbc SQL Auth!")
            return True
            
        except ImportError:
            print("❌ pyodbc not installed. Install with: pip install pyodbc")
            return False
        except Exception as e:
            print(f"❌ pyodbc SQL Auth failed: {type(e).__name__}: {e}")
            return False
    
    def try_pyodbc_windows_auth(self):
        """Try pyodbc with Windows Authentication"""
        try:
            import pyodbc
            
            drivers = [d for d in pyodbc.drivers() if 'SQL Server' in d]
            if not drivers:
                print("❌ No SQL Server ODBC drivers found")
                return False
            
            driver = drivers[0]
            print(f"Trying pyodbc with Windows Auth using driver: {driver}")
            
            connection_string = (
                f"DRIVER={{{driver}}};"
                f"SERVER={self.db_config['server']},{self.db_config['port']};"
                f"DATABASE={self.db_config['database_name']};"
                f"Trusted_Connection=yes;"
                f"Connection Timeout=10;"
                f"Login Timeout=10;"
                f"TrustServerCertificate=yes;"
                f"Encrypt=no;"
            )
            
            self.connection = pyodbc.connect(connection_string)
            self.connection_type = f"pyodbc_windows_auth_{driver}"
            print("✅ Successfully connected with pyodbc Windows Auth!")
            return True
            
        except Exception as e:
            print(f"❌ pyodbc Windows Auth failed: {type(e).__name__}: {e}")
            return False
    
    def try_sqlalchemy_connection(self):
        """Try SQLAlchemy with pymssql"""
        try:
            from sqlalchemy import create_engine
            
            print("Trying SQLAlchemy with pymssql...")
            connection_string = (
                f"mssql+pymssql://{self.db_config['sql_auth']['username']}:"
                f"{self.db_config['sql_auth']['password']}"
                f"@{self.db_config['server']}:{self.db_config['port']}"
                f"/{self.db_config['database_name']}"
            )
            
            self.engine = create_engine(connection_string, echo=False)
            
            # Test the connection
            with self.engine.connect() as conn:
                result = conn.execute("SELECT 1")
                result.fetchone()
            
            self.connection_type = "sqlalchemy_pymssql"
            print("✅ Successfully connected with SQLAlchemy!")
            return True
            
        except ImportError:
            print("❌ SQLAlchemy not installed. Install with: pip install sqlalchemy pymssql")
            return False
        except Exception as e:
            print(f"❌ SQLAlchemy failed: {type(e).__name__}: {e}")
            return False
    
    def connect(self):
        """Try all connection methods in order of preference"""
        print("🔄 Attempting to connect to SQL Server...")
        print("-" * 50)
        
        methods = [
            ("pymssql (Recommended)", self.try_pymssql_connection),
            ("SQLAlchemy + pymssql", self.try_sqlalchemy_connection),
            ("pyodbc SQL Auth", self.try_pyodbc_sql_auth),
            ("pyodbc Windows Auth", self.try_pyodbc_windows_auth)
        ]
        
        for method_name, method in methods:
            print(f"\n=== Trying {method_name} ===")
            if method():
                print(f"🎉 Connected successfully using {method_name}!")
                return True
            print(f"Failed with {method_name}, trying next method...")
        
        print("❌ All connection methods failed!")
        return False
    
    def execute_query(self, query):
        """Execute query and return DataFrame"""
        if not (self.connection or self.engine):
            if not self.connect():
                raise Exception("Could not establish database connection")
        
        try:
            if self.connection_type.startswith("sqlalchemy"):
                df = pd.read_sql(query, self.engine)
            else:
                df = pd.read_sql(query, self.connection)
            
            print(f"✅ Query executed successfully! Retrieved {len(df)} rows.")
            return df
            
        except Exception as e:
            print(f"❌ Error executing query: {e}")
            raise
    
    def test_connection(self):
        """Test connection with sample queries"""
        if not (self.connection or self.engine):
            if not self.connect():
                return False
        
        try:
            print("\n=== Connection Test ===")
            
            # Basic info query
            info_query = """
            SELECT 
                @@VERSION as sql_version,
                COALESCE(SYSTEM_USER, USER_NAME()) as current_user,
                DB_NAME() as current_database,
                GETDATE() as current_time
            """
            
            result = self.execute_query(info_query)
            
            print(f"Connected as: {result['current_user'].iloc[0]}")
            print(f"Database: {result['current_database'].iloc[0]}")
            print(f"Time: {result['current_time'].iloc[0]}")
            print(f"SQL Server: {result['sql_version'].iloc[0][:60]}...")
            
            # Count tables
            tables_query = "SELECT COUNT(*) as table_count FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
            table_count = self.execute_query(tables_query)
            print(f"Database has {table_count['table_count'].iloc[0]} tables")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def list_tables(self):
        """Get list of all tables"""
        query = """
        SELECT 
            TABLE_SCHEMA,
            TABLE_NAME,
            TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
        return self.execute_query(query)
    
    def close(self):
        """Close connections"""
        if self.connection:
            self.connection.close()
            print("🔒 Database connection closed")
        if self.engine:
            self.engine.dispose()
            print("🔒 Database engine disposed")

# Example usage
if __name__ == "__main__":
    print("=== SQL Server Connection Test (Updated Config Format) ===")
    
    # Test the connection
    db = SQLServerConnection('config.json')
    
    try:
        if db.test_connection():
            print("\n=== Listing Tables ===")
            tables = db.list_tables()
            print(f"Found {len(tables)} tables:")
            print(tables.head(10).to_string(index=False))
            
            # Example query
            print("\n=== Sample Query ===")
            sample_query = "SELECT TOP 3 * FROM INFORMATION_SCHEMA.COLUMNS ORDER BY TABLE_NAME"
            result = db.execute_query(sample_query)
            print(result.to_string(index=False))
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        db.close()
    
    print("\n=== Connection test complete ===")