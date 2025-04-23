"""
Launcher script for the CodePilot Streamlit UI.
This ensures that Streamlit is properly initialized with the 'streamlit run' command.
"""
import os
import sys
import subprocess
import pkg_resources

def main():
    """Launch the Streamlit UI."""
    main_path = pkg_resources.resource_filename('codepilot.ui', 'main.py')
    
    if not os.path.exists(main_path):
        print(f"Error: Could not find Streamlit UI script at {main_path}")
        sys.exit(1)
    
    # Build the streamlit run command
    cmd = [sys.executable, "-m", "streamlit", "run", main_path, "--browser.serverAddress", "localhost"]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStreamlit UI stopped.")
    except Exception as e:
        print(f"Error launching Streamlit UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()