sudo apt update
sudo apt install -y python3 python3-venv python3-pip
sudo apt install sqlite3

source venv/bin/activate

python init_helpdesk_db.py
python init_users_db.py

pip install openai faiss-cpu numpy

python build_faiss_index.py

pip install fastapi uvicorn openai faiss-cpu numpy

pip install intersight dotenv

#Run this to update DB
sqlite3 helpdesk.db
ALTER TABLE tickets
  ADD COLUMN requester_id INTEGER;

-- (Optional: ensure it’s NULL by default; no further action needed)

-- Verify the schema now has requester_id:
PRAGMA table_info(tickets);
.quit

#once DB is updated run this

 python backfill_requesters.py

#Get NVIDIA API Key
#Pull and Run the NIM
$ docker login nvcr.io
Username: $oauthtoken
Password: <PASTE_API_KEY_HERE>

#Pull and run the NVIDIA NIM with the command below. This will download the optimized model for your infrastructure.
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
docker run -it --rm \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    nvcr.io/nim/meta/llama-3.1-8b-instruct:latest

#still using OPENAI for embeddings - plan to use NVIDIA NIM
#set this in the enviroment:
OPENAI_API_KEY=

uvicorn intersight_agent:app --host 0.0.0.0 --port 8002 --reload
uvicorn main:app --host 0.0.0.0 --port 8001 --reload


##Intersight Agent (WIP)
#get Intersight API key
set these in the enviroment:
INTERSIGHT_API_KEY_ID = "your Intersight Key ID"
INTERSIGHT_SECRET_KEY_FILE_PATH = "path to secret key file - Example /home/ubuntu/app/SecretKey.txt"
