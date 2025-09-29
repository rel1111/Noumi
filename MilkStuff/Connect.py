import json
import threading
import time
import signal
import sys
from pathlib import Path

class TimeoutSQLAuth:
    def __init__(self, config_file='config.json'):
        script_dir = Path(__file__).parent if '__file__' in globals() else Path('.')
        self.config_file = script_dir / config_file
        self.connection = None
        self.connection_result = None
        self.connection_error = None
        self.load_config()
    
    def load_config(self):
        with open(self.config_file, 'r') as f:
            config = json.load(f)
            self.db_config = config['database']
    
    def timeout_handler(self, signum, frame):
        """Handle timeout signal"""
        print(f"\n⏰ TIMEOUT: Connection attempt exceeded time limit")
        print("This suggests the connection is hanging during authentication")
        sys.exit(1)
    
    def connect_pymssql_thread(self):
        """Run pymssql connection in separate thread"""
        try:
            import pymssql
            
            print("Starting pymssql connection attempt...")
            
            conn = pymssql.connect(
                server=self.db_config['server'],
                user=self.db_config['sql_auth']['username'],
                password=self.db_config['sql_auth']['password'],
                database=self.db_config['database_name'],
                port=self.db_config['port'],
                timeout=10,
                login_timeout=10,
                charset='utf8'
            )
            
            self.connection_result = conn
            print("✅ pymssql connection successful in thread!")
            
        except Exception as e:
            self.connection_error = e
            print(f"❌ pymssql connection failed in thread: {e}")
    
    def try_pymssql_with_timeout(self, timeout_seconds=15):
        """Try pymssql with timeout"""
        print(f"=== Trying pymssql with {timeout_seconds}s timeout ===")
        print(f"Server: {self.db_config['server']}:{self.db_config['port']}")
        print(f"Database: {self.db_config['database_name']}")
        print(f"Username: {self.db_config['sql_auth']['username']}")
        
        try:
            import pymssql
            
            # Start connection in separate thread
            connection_thread = threading.Thread(target=self.connect_pymssql_thread)
            connection_thread.daemon = True
            connection_thread.start()
            
            # Wait for thread to complete or timeout
            connection_thread.join(timeout_seconds)
            
            if connection_thread.is_alive():
                print(f"⏰ TIMEOUT: pymssql connection took longer than {timeout_seconds} seconds")
                print("The connection is hanging - likely an authentication or server config issue")
                return False
            
            # Check results
            if self.connection_result:
                self.connection = self.connection_result
                print("✅ pymssql connection completed successfully!")
                return True
            elif self.connection_error:
                print(f"❌ pymssql connection failed: {self.connection_error}")
                return False
            else:
                print("❌ pymssql connection completed but no result")
                return False
                
        except ImportError:
            print("❌ pymssql not installed. Run: pip install pymssql")
            return False
    
    def try_pyodbc_minimal(self):
        """Try minimal pyodbc connection"""
        print("\n=== Trying pyodbc (minimal) ===")
        
        try:
            import pyodbc
            
            drivers = [d for d in pyodbc.drivers() if 'SQL Server' in d]
            if not drivers:
                print("❌ No ODBC drivers found")
                return False
            
            driver = drivers[0]
            print(f"Using driver: {driver}")
            
            connection_string = (
                f"DRIVER={{{driver}}};"
                f"SERVER={self.db_config['server']},{self.db_config['port']};"
                f"DATABASE={self.db_config['database_name']};"
                f"UID={self.db_config['sql_auth']['username']};"
                f"PWD={self.db_config['sql_auth']['password']};"
                f"Connection Timeout=10;"
                f"Login Timeout=10;"
                f"Encrypt=no;"
            )
            
            print("Attempting pyodbc connection...")
            start_time = time.time()
            
            conn = pyodbc.connect(connection_string)
            elapsed = time.time() - start_time
            
            print(f"✅ pyodbc connection successful in {elapsed:.2f}s!")
            self.connection = conn
            return True
            
        except ImportError:
            print("❌ pyodbc not installed")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ pyodbc failed after {elapsed:.2f}s: {e}")
            return False
    
    def test_basic_connectivity(self):
        """Test basic network connectivity first"""
        print("=== Testing Basic Network Connectivity ===")
        
        import socket
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            
            print(f"Testing connection to {self.db_config['server']}:{self.db_config['port']}...")
            result = sock.connect_ex((self.db_config['server'], int(self.db_config['port'])))
            sock.close()
            
            if result == 0:
                print("✅ Basic network connectivity successful")
                return True
            else:
                print(f"❌ Network connectivity failed (error {result})")
                return False
                
        except Exception as e:
            print(f"❌ Network test failed: {e}")
            return False
    
    def diagnose_hanging_connection(self):
        """Diagnose why connections are hanging"""
        print("\n=== Connection Hanging Diagnosis ===")
        print("Since the connection is hanging indefinitely, this suggests:")
        print()
        print("1. **Authentication Loop**: SQL Server is repeatedly trying to authenticate")
        print("   - Username/password might be incorrect")
        print("   - Account might be locked or disabled")
        print("   - SQL Server might not allow SQL authentication")
        print()
        print("2. **Network Issue**: Connection partially establishes but hangs")
        print("   - Firewall might be dropping packets")
        print("   - SQL Server might be overloaded")
        print("   - Network routing issues")
        print()
        print("3. **SQL Server Configuration**: Server settings causing issues")
        print("   - Mixed authentication mode not enabled")
        print("   - Connection limits reached")
        print("   - Database not accessible to user")
        print()
        print("💡 Next steps:")
        print("1. Ask DBA to check SQL Server error logs during connection attempt")
        print("2. Verify SQL Server is in 'Mixed Mode' authentication")
        print("3. Test connection from SQL Server Management Studio")
        print("4. Check if username/password work from another tool")
    
    def test_connection_safely(self):
        """Test connection with proper timeouts"""
        print("=== Safe SQL Server Connection Test ===")
        print("This will timeout connections that hang indefinitely")
        print()
        
        # Test 1: Basic network connectivity
        if not self.test_basic_connectivity():
            print("\n❌ Basic connectivity failed - check server/port")
            return False
        
        # Test 2: Try pymssql with timeout
        if self.try_pymssql_with_timeout(15):
            return self.test_working_connection()
        
        # Test 3: Try pyodbc
        if self.try_pyodbc_minimal():
            return self.test_working_connection()
        
        # If all failed, provide diagnosis
        self.diagnose_hanging_connection()
        return False
    
    def test_working_connection(self):
        """Test a working connection"""
        if not self.connection:
            return False
        
        try:
            print("\n=== Testing Working Connection ===")
            
            cursor = self.connection.cursor()
            cursor.execute("SELECT @@VERSION, USER_NAME(), DB_NAME(), GETDATE()")
            version, user, db, time = cursor.fetchone()
            
            print("🎉 Connection test successful!")
            print(f"User: {user}")
            print(f"Database: {db}")
            print(f"Time: {time}")
            print(f"SQL Server: {version[:50]}...")
            
            cursor.close()
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def close(self):
        """Close connection"""
        if self.connection:
            try:
                self.connection.close()
                print("🔒 Connection closed")
            except:
                pass

def main():
    """Main function with keyboard interrupt handling"""
    db = TimeoutSQLAuth('config.json')
    
    try:
        print("Press Ctrl+C at any time to cancel")
        print("=" * 50)
        
        success = db.test_connection_safely()
        
        if not success:
            print("\n" + "=" * 50)
            print("CONNECTION FAILED")
            print("The connection is likely hanging due to:")
            print("• Incorrect username/password")
            print("• SQL Server authentication not enabled") 
            print("• Network/firewall issues")
            print("• SQL Server configuration problems")
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Connection attempt cancelled by user")
        print("This confirms the connection was hanging")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()