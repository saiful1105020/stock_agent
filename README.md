conda create --name langgraph python=3.11
conda activate langgraph

git clone https://github.com/saiful1105020/stock_agent.git
cd [git folder]

pip install -r requirements.txt

pip install jupyterlab
export OPENAI_API_KEY=”YOUR-KEY”
export SERPER_API_KEY=”YOUR-KEY”
