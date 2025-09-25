How to setup:
=============

conda init

conda create --name langgraph python=3.11

conda activate langgraph

git clone https://github.com/saiful1105020/stock_agent.git

cd [git folder]

pip install -r requirements.txt

export OPENAI_API_KEY=”YOUR-KEY”

export SERPER_API_KEY=”YOUR-KEY”

**FAQ1**
1. conda command does not work on Windows

   Locate your Anaconda installation:

    * Search for "Anaconda Navigator" in the Windows Start menu.
    * Right-click on "Anaconda Navigator" and select "Open file location."
    * In the folder that opens, right-click on "Anaconda Navigator" again and select "Open file location." This should lead you to the main Anaconda installation directory (e.g., C:\Users\YourUsername\Anaconda3).
    * Inside this directory, locate the Scripts folder (e.g., C:\Users\YourUsername\Anaconda3\Scripts).

  Add Anaconda to your system's PATH:
  
  * Search for "Environment Variables" in the Windows Start menu and select "Edit the system environment variables."
  * In the System Properties window, click on "Environment Variables..."
  * Under "System variables," find and select the "Path" variable, then click "Edit..."
  * Click "New" and add the path to your Anaconda Scripts directory (e.g., C:\Users\YourUsername\Anaconda3\Scripts).
  * Click "OK" on all open windows to save the changes.

  Restart your terminal or command prompt:
  
  * Close any open command prompt or terminal windows.
  * Open a new command prompt or terminal.

  Verify the installation:
  
  * Type conda --version and press Enter. If the conda command is now recognized, it should display the installed Conda version.
