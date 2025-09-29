import json
import time
import signal
import sys
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def timeout(duration):
    """Context manager for timing out operations"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    # Set the signal handler
    if hasattr(signal, 'SIGALRM'):  # Unix/Linux/Mac
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(duration)
        try:
            yield
        finally:
            signal.alarm(0)
    else:  # Windows - use a different approach
        import threading
        
        def timeout_func():
            time.sleep(duration)
            print(f"\n⏰ TIMEOUT: Operation took longer than {duration} seconds")
            # On Windows, we can't interrupt the connection attempt easily
            # Just print a message
        
        timer = threading.Timer(duration, timeout_func)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

def load_config():
    """Load database configuration"""
    script_dir = Path(__file__).parent if '__file__' in globals() else Path('.')
    config_file = script_dir / 'config.json'
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config['database']

def test_pymssql_with_timeout():
    """Test pymssql connection with proper timeout handling"""
    try:
        import pymssql
        print("✅ pymssql is available")
    except ImportError:
        print("❌ pymssql not installed")
        return False
    
    config = load_config()
    
    # Test different scenarios
    scenarios = [
        {
            'name': 'Basic connection (5 sec timeout)',
            'params': {
                'server': config['server'],
                'user': config['username'],
                'password': config['password'],
                'database': config['database_name'],
                'port': config['port'],
                'timeout': 5,
                'login_timeout': 5
            }
        },
        {
            'name': 'Connection without database specified',
            'params': {
                'server': config['server'],
                'user': config['username'],
                'password': config['password'],
                'port': config['port'],
                'timeout': 5,
                'login_timeout': 5
            }
        },
        {
            'name': 'Connection with Windows Authentication attempt',
            'params': {
                'server': config['server'],
                'port': config['port'],
                'timeout': 5,
                'login_timeout': 5
                # No user/password - let it try Windows auth
            }
        },
        {
            'name': 'Connection to default instance',
            'params': {
                'server': config['server'],
                'user': config['username'],
                'password': config['password'],
                'database': 'master',  # Try connecting to master database
                'timeout': 5,
                'login_timeout': 5
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n=== Scenario {i}: {scenario['name']} ===")
        
        start_time = time.time()
        
        try:
            with timeout(10):  # 10 second overall timeout
                print("Attempting connection...")
                conn = pymssql.connect(**scenario['params'])
                
                elapsed = time.time() - start_time
                print(f"✅ SUCCESS! Connected in {elapsed:.2f} seconds")
                
                # Quick test query
                cursor = conn.cursor()
                cursor.execute("SELECT @@VERSION as version, DB_NAME() as current_db")
                row = cursor.fetchone()
                print(f"SQL Server Version: {row[0][:50]}...")
                print(f"Current Database: {row[1]}")
                
                cursor.close()
                conn.close()
                return True
                
        except TimeoutError as e:
            elapsed = time.time() - start_time
            print(f"⏰ TIMEOUT: {e} (waited {elapsed:.2f} seconds)")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ FAILED after {elapsed:.2f} seconds: {type(e).__name__}: {e}")
    
    return False

def test_alternative_drivers():
    """Test alternative connection methods"""
    config = load_config()
    
    print("\n=== Testing Alternative Drivers ===")
    
    # Test 1: Try pyodbc with SQL Server Native Client
    print("\n--- Testing pyodbc ---")
    try:
        import pyodbc
        
        # Check available drivers
        drivers = [d for d in pyodbc.drivers() if 'SQL Server' in d]
        print(f"Available ODBC drivers: {drivers}")
        
        if drivers:
            driver = drivers[0]  # Use first available driver
            connection_string = (
                f"DRIVER={{{driver}}};"
                f"SERVER={config['server']};"
                f"DATABASE={config['database_name']};"
                f"UID={config['username']};"
                f"PWD={config['password']};"
                f"Encrypt=no;"
                f"TrustServerCertificate=yes;"
                f"Connection Timeout=5;"
            )
            
            print(f"Trying with driver: {driver}")
            with timeout(10):
                conn = pyodbc.connect(connection_string)
                print("✅ pyodbc connection successful!")
                conn.close()
                return True
        else:
            print("❌ No SQL Server ODBC drivers found")
            
    except ImportError:
        print("❌ pyodbc not installed")
    except Exception as e:
        print(f"❌ pyodbc failed: {e}")
    
    # Test 2: Try SQLAlchemy
    print("\n--- Testing SQLAlchemy ---")
    try:
        from sqlalchemy import create_engine
        
        connection_string = (
            f"mssql+pymssql://{config['username']}:{config['password']}"
            f"@{config['server']}:{config['port']}/{config['database_name']}"
            f"?timeout=5"
        )
        
        print("Creating SQLAlchemy engine...")
        engine = create_engine(connection_string, echo=False)
        
        with timeout(10):
            with engine.connect() as conn:
                result = conn.execute("SELECT 1")
                result.fetchone()
            print("✅ SQLAlchemy connection successful!")
            return True
            
    except ImportError:
        print("❌ SQLAlchemy not installed")
    except Exception as e:
        print(f"❌ SQLAlchemy failed: {e}")
    
    return False

def main():
    """Run all connection tests"""
    print("=== SQL Server Connection Test with Timeouts ===")
    
    # Test pymssql with timeouts
    success = test_pymssql_with_timeout()
    
    if not success:
        print("\n=== pymssql failed, trying alternatives ===")
        success = test_alternative_drivers()
    
    if success:
        print("\n🎉 SUCCESS! Connection established!")
    else:
        print("\n❌ All connection methods failed.")
        print("\n🔍 DIAGNOSIS:")
        print("Since network connectivity works but authentication hangs:")
        print("1. SQL Server may be configured for Windows Authentication only")
        print("2. The username/password may be incorrect")
        print("3. The user may not have permission to access this database")
        print("4. SQL Server may be configured with different security settings")
        print("\n💡 NEXT STEPS:")
        print("1. Ask your DBA to verify the username/password")
        print("2. Check if the SQL Server allows SQL Authentication")
        print("3. Try connecting from SQL Server Management Studio with same credentials")
        print("4. Check SQL Server error logs for authentication failures")

if __name__ == "__main__":
    main()