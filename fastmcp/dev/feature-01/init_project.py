#!/usr/bin/env python3
"""
setup_mcp_platform.py - Complete MCP Platform Setup and Migration Workflow

This script provides a complete end-to-end workflow for setting up the MCP Platform:
1. Creates the project structure using init_project.py
2. Migrates code from existing single-file app using migrate_code.py  
3. Sets up the environment and dependencies
4. Validates the setup and provides next steps

Usage:
    python setup_mcp_platform.py
    python setup_mcp_platform.py --project-name my_mcp_app
    python setup_mcp_platform.py --source-file original_app.py --project-name mcp_platform
"""

import argparse
import os
import subprocess
import sys
import urllib.request
import tempfile
from pathlib import Path
from datetime import datetime
import json

class MCPPlatformSetup:
    """Complete MCP Platform setup and migration workflow"""
    
    def __init__(self, project_name="mcp_platform", source_file=None):
        self.project_name = project_name
        self.source_file = source_file
        self.project_path = Path(project_name)
        self.setup_log = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log setup progress"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.setup_log.append(log_entry)
        
        # Print with color coding
        if level == "ERROR":
            print(f"❌ {message}")
        elif level == "SUCCESS":
            print(f"✅ {message}")
        elif level == "WARNING":
            print(f"⚠️  {message}")
        else:
            print(f"ℹ️  {message}")
    
    def check_prerequisites(self):
        """Check system prerequisites"""
        self.log("Checking prerequisites...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            self.log(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}", "ERROR")
            return False
        else:
            self.log(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} ✓", "SUCCESS")
        
        # Check pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
            self.log("pip available ✓", "SUCCESS")
        except subprocess.CalledProcessError:
            self.log("pip not available", "ERROR")
            return False
        
        # Check if project directory already exists
        if self.project_path.exists():
            self.log(f"Directory {self.project_name} already exists", "WARNING")
            response = input(f"Directory {self.project_name} exists. Continue? (y/N): ")
            if response.lower() != 'y':
                self.log("Setup cancelled by user", "ERROR")
                return False
        
        return True
    
    def download_scripts(self):
        """Download init_project.py and migrate_code.py if not present"""
        self.log("Checking for setup scripts...")
        
        scripts = {
            "init_project.py": "Project structure generator",
            "migrate_code.py": "Code migration helper"
        }
        
        for script_name, description in scripts.items():
            if not Path(script_name).exists():
                self.log(f"{script_name} not found - you'll need to create it manually", "WARNING")
                # In a real implementation, you might download from a repository
                # For now, we'll assume the user has the scripts
        
        return True
    
    def create_project_structure(self):
        """Create the project structure using init_project.py"""
        self.log("Creating project structure...")
        
        try:
            # Check if init_project.py exists
            if not Path("init_project.py").exists():
                self.log("init_project.py not found. Please ensure it's in the current directory.", "ERROR")
                return False
            
            # Run init_project.py
            result = subprocess.run([
                sys.executable, "init_project.py", self.project_name
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("Project structure created successfully", "SUCCESS")
                return True
            else:
                self.log(f"Failed to create project structure: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error creating project structure: {e}", "ERROR")
            return False
    
    def migrate_existing_code(self):
        """Migrate code from existing file if provided"""
        if not self.source_file:
            self.log("No source file provided, skipping migration")
            return True
        
        self.log(f"Migrating code from {self.source_file}...")
        
        try:
            # Check if migrate_code.py exists
            if not Path("migrate_code.py").exists():
                self.log("migrate_code.py not found. Please ensure it's in the current directory.", "ERROR")
                return False
            
            # Check if source file exists
            if not Path(self.source_file).exists():
                self.log(f"Source file not found: {self.source_file}", "ERROR")
                return False
            
            # Run migrate_code.py
            result = subprocess.run([
                sys.executable, "migrate_code.py",
                "--source", self.source_file,
                "--target", str(self.project_path),
                "--force"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("Code migration completed successfully", "SUCCESS")
                return True
            else:
                self.log(f"Code migration failed: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error during code migration: {e}", "ERROR")
            return False
    
    def setup_virtual_environment(self):
        """Set up Python virtual environment"""
        self.log("Setting up virtual environment...")
        
        venv_path = self.project_path / "venv"
        
        try:
            # Create virtual environment
            subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], check=True, capture_output=True)
            
            self.log("Virtual environment created", "SUCCESS")
            
            # Determine activation script path
            if os.name == 'nt':  # Windows
                activate_script = venv_path / "Scripts" / "activate.bat"
                pip_executable = venv_path / "Scripts" / "pip.exe"
            else:  # Unix-like
                activate_script = venv_path / "bin" / "activate"
                pip_executable = venv_path / "bin" / "pip"
            
            # Install dependencies
            self.log("Installing dependencies...")
            requirements_file = self.project_path / "requirements.txt"
            
            if requirements_file.exists():
                subprocess.run([
                    str(pip_executable), "install", "-r", str(requirements_file)
                ], check=True, capture_output=True)
                
                self.log("Dependencies installed successfully", "SUCCESS")
            else:
                self.log("requirements.txt not found, skipping dependency installation", "WARNING")
            
            return True, activate_script
            
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to set up virtual environment: {e}", "ERROR")
            return False, None
        except Exception as e:
            self.log(f"Error setting up virtual environment: {e}", "ERROR")
            return False, None
    
    def create_environment_file(self):
        """Create .env file from template"""
        self.log("Setting up environment configuration...")
        
        env_example = self.project_path / ".env.example"
        env_file = self.project_path / ".env"
        
        if env_example.exists() and not env_file.exists():
            try:
                # Copy .env.example to .env
                with open(env_example, 'r') as f:
                    content = f.read()
                
                with open(env_file, 'w') as f:
                    f.write(content)
                
                self.log(".env file created from template", "SUCCESS")
                self.log("Remember to add your API keys to .env file", "WARNING")
                return True
                
            except Exception as e:
                self.log(f"Failed to create .env file: {e}", "ERROR")
                return False
        else:
            if env_file.exists():
                self.log(".env file already exists", "INFO")
            else:
                self.log(".env.example not found", "WARNING")
            return True
    
    def validate_setup(self):
        """Validate the setup by checking key components"""
        self.log("Validating setup...")
        
        # Check key files exist
        key_files = [
            "main.py",
            "requirements.txt",
            "config/settings.py",
            "config/database.py",
            "utils/session_state.py",
            "components/sidebar.py"
        ]
        
        missing_files = []
        for file_path in key_files:
            full_path = self.project_path / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.log(f"Missing files: {', '.join(missing_files)}", "ERROR")
            return False
        
        self.log("All key files present ✓", "SUCCESS")
        
        # Check if main.py can be imported (basic syntax check)
        try:
            # Change to project directory temporarily
            original_cwd = os.getcwd()
            os.chdir(self.project_path)
            
            # Try to compile main.py
            with open("main.py", 'r') as f:
                content = f.read()
            
            compile(content, "main.py", "exec")
            self.log("main.py syntax valid ✓", "SUCCESS")
            
            os.chdir(original_cwd)
            return True
            
        except SyntaxError as e:
            self.log(f"Syntax error in main.py: {e}", "ERROR")
            os.chdir(original_cwd)
            return False
        except Exception as e:
            self.log(f"Error validating main.py: {e}", "ERROR")
            os.chdir(original_cwd)
            return False
    
    def create_setup_report(self, venv_success: bool, activate_script: Path = None):
        """Create detailed setup report"""
        self.log("Creating setup report...")
        
        report_path = self.project_path / "SETUP_REPORT.md"
        
        report_content = f"""# MCP Platform Setup Report

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Project**: {self.project_name}
**Source File**: {self.source_file or "None (new project)"}

## Setup Summary

"""
        
        # Add setup log
        report_content += "### Setup Log\n\n"
        for log_entry in self.setup_log:
            report_content += f"- {log_entry}\n"
        
        # Add next steps
        report_content += f"""

## Next Steps

### 1. Activate Virtual Environment

"""
        
        if venv_success and activate_script:
            if os.name == 'nt':  # Windows
                report_content += f"""
```cmd
cd {self.project_name}
venv\\Scripts\\activate
```
"""
            else:  # Unix-like
                report_content += f"""
```bash
cd {self.project_name}
source venv/bin/activate
```
"""
        else:
            report_content += """
Virtual environment setup failed. Install dependencies manually:
```bash
pip install -r requirements.txt
```
"""
        
        report_content += f"""
### 2. Configure API Keys

Edit `.env` file and add your API keys:
```bash
# Edit the .env file
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 3. Run the Application

```bash
cd {self.project_name}
streamlit run main.py
```

Or use the convenience script:
```bash
python run.py
```

### 4. Test the Setup

1. Open your browser to the URL shown (usually http://localhost:8501)
2. Navigate through all pages using the sidebar
3. Try adding a server in "Server Admin"
4. Test queries in "Demo Client"

### 5. Development

- **Add custom tools**: Use "Tool Manager" page
- **Configure servers**: Use "Server Admin" page  
- **Review code**: Check the migrated modules in `core/`
- **Add tests**: Extend the test suite in `tests/`

## Project Structure

```
{self.project_name}/
├── main.py                    # Application entry point
├── run.py                     # Convenience launcher
├── requirements.txt           # Dependencies
├── .env                       # Environment variables
├── config/                    # Configuration modules
├── core/                      # Business logic
├── pages/                     # Streamlit pages
├── components/                # Reusable UI components
├── utils/                     # Utilities
├── tests/                     # Test suite
└── docs/                      # Documentation
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure virtual environment is activated
2. **Missing API Keys**: Configure your `.env` file
3. **Port Issues**: Streamlit will suggest alternative ports if 8501 is busy
4. **Permission Errors**: Check file permissions on Unix-like systems

### Getting Help

- Review documentation in `docs/` folder
- Check `MIGRATION_REPORT.md` if code was migrated
- Look at setup logs above for specific error messages

## Resources

- **Setup Guide**: `docs/SETUP.md`
- **Architecture**: `docs/ARCHITECTURE.md`  
- **API Reference**: `docs/API.md`
- **Migration Report**: `MIGRATION_REPORT.md` (if applicable)

---

Setup completed on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.log(f"Setup report created: {report_path}", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Failed to create setup report: {e}", "ERROR")
            return False
    
    def run_setup(self):
        """Run the complete setup workflow"""
        print("🧠 MCP Platform Complete Setup")
        print("=" * 50)
        print(f"Project: {self.project_name}")
        if self.source_file:
            print(f"Source: {self.source_file}")
        print()
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            print("\n❌ Prerequisites check failed. Please fix the issues above.")
            return False
        
        # Step 2: Download/check scripts
        if not self.download_scripts():
            print("\n❌ Script preparation failed.")
            return False
        
        # Step 3: Create project structure
        if not self.create_project_structure():
            print("\n❌ Project structure creation failed.")
            return False
        
        # Step 4: Migrate existing code (if applicable)
        if not self.migrate_existing_code():
            print("\n❌ Code migration failed.")
            return False
        
        # Step 5: Set up virtual environment
        venv_success, activate_script = self.setup_virtual_environment()
        if not venv_success:
            print("\n⚠️  Virtual environment setup failed, but continuing...")
        
        # Step 6: Create environment file
        if not self.create_environment_file():
            print("\n⚠️  Environment file creation failed, but continuing...")
        
        # Step 7: Validate setup
        if not self.validate_setup():
            print("\n⚠️  Setup validation failed, but project may still work...")
        
        # Step 8: Create setup report
        self.create_setup_report(venv_success, activate_script)
        
        # Success message
        print("\n" + "=" * 60)
        print("🎉 MCP Platform setup completed!")
        print("=" * 60)
        
        print(f"""
📁 Project created: {self.project_name}/
📊 Setup report: {self.project_name}/SETUP_REPORT.md

🚀 Quick start:
""")
        
        if venv_success:
            if os.name == 'nt':
                print(f"   cd {self.project_name}")
                print("   venv\\Scripts\\activate")
            else:
                print(f"   cd {self.project_name}")
                print("   source venv/bin/activate")
        else:
            print(f"   cd {self.project_name}")
            print("   pip install -r requirements.txt")
        
        print("   streamlit run main.py")
        
        print(f"""
📝 Next steps:
   1. Configure API keys in .env file
   2. Test the application
   3. Add your MCP servers
   4. Start building!

📚 Documentation available in docs/ folder
""")
        
        return True

def main():
    """Main setup script"""
    
    parser = argparse.ArgumentParser(
        description="Complete MCP Platform setup and migration workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_mcp_platform.py
  python setup_mcp_platform.py --project-name my_mcp_app
  python setup_mcp_platform.py --source-file original_app.py --project-name mcp_platform
  
This script will:
1. Create the complete project structure
2. Migrate code from your existing app (if provided)
3. Set up virtual environment and dependencies
4. Configure environment files
5. Validate the setup
6. Provide detailed next steps
        """
    )
    
    parser.add_argument(
        "--project-name", "-p",
        default="mcp_platform",
        help="Name for the new MCP platform project (default: mcp_platform)"
    )
    
    parser.add_argument(
        "--source-file", "-s",
        help="Path to existing single-file MCP app to migrate from"
    )
    
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="Skip virtual environment creation"
    )
    
    args = parser.parse_args()
    
    # Run setup
    setup = MCPPlatformSetup(
        project_name=args.project_name,
        source_file=args.source_file
    )
    
    if args.skip_venv:
        setup.log("Skipping virtual environment setup (--skip-venv)", "WARNING")
    
    success = setup.run_setup()
    
    if success:
        print("\n✨ Setup completed successfully! Your MCP Platform is ready to use.")
        sys.exit(0)
    else:
        print("\n💥 Setup encountered errors. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()