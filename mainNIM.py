#!/usr/bin/env python3
import os
import sqlite3
import pickle
import numpy as np
import faiss
from datetime import date, datetime # Added datetime for created_at
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI, OpenAIError

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
DB_FILENAME        = "helpdesk.db"         # SQLite file from Step 1
FAISS_INDEX_FILE   = "faiss_index.bin"     # FAISS index from Step 2
MAPPING_FILE       = "ticket_mapping.pkl"  # ticket_id ↔ text mapping

EMBEDDING_MODEL    = "text-embedding-3-small"   # OpenAI embedding model
CHAT_MODEL         = "meta/llama-3.1-8b-instruct" # NVIDIA NIM chat model
TOP_K              = 5                           # how many neighbors to retrieve

# -------------------------------------------------------------
# INITIALIZATION (load FAISS index + mapping, API clients, user lookup)
# -------------------------------------------------------------
# Ensure required files exist
if not os.path.isfile(DB_FILENAME):
    raise RuntimeError(f"Database file `{DB_FILENAME}` not found. Run Step 1 first.")
if not os.path.isfile(FAISS_INDEX_FILE):
    raise RuntimeError(f"FAISS index `{FAISS_INDEX_FILE}` not found. Run Step 2 first.")
if not os.path.isfile(MAPPING_FILE):
    raise RuntimeError(f"Mapping file `{MAPPING_FILE}` not found. Run Step 2 first.")

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_FILE)

# Load ticket mapping (ticket_id ↔ combined text)
with open(MAPPING_FILE, "rb") as f:
    ticket_mapping: List[dict] = pickle.load(f)

# Extract parallel lists for quick lookup (will be modified at runtime by uploads)
ticket_ids   = [entry["ticket_id"] for entry in ticket_mapping]
ticket_texts = [entry["text"] for entry in ticket_mapping]

# Build a user_lookup: user_id → full_name
# This will also be updated if users are created/updated via API
user_lookup = {}
def load_user_lookup():
    global user_lookup
    user_lookup = {}
    conn_tmp = sqlite3.connect(DB_FILENAME)
    cur_tmp = conn_tmp.cursor()
    cur_tmp.execute("SELECT user_id, full_name FROM users;")
    for uid, name in cur_tmp.fetchall():
        user_lookup[uid] = name
    conn_tmp.close()

load_user_lookup() # Initial load

# Initialize API clients
openai_client = OpenAI()
nim_client = OpenAI(
    base_url = "http://0.0.0.0:8100/v1/", # Ensure this is your NIM's correct address
    api_key = "nvapi-..." # Replace with your actual NIM API key if needed, or handle appropriately
)

# -------------------------------------------------------------
# FASTAPI APP + CORS
# -------------------------------------------------------------
app = FastAPI(
    title="IT Helpdesk RAG Chatbot",
    description="A FastAPI service that serves a helpdesk ticket DB, RAG-powered Chat, and file upload.",
    version="1.2.0" # Incremented version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Pydantic models for Tickets / Users / Chat / Uploads
# -------------------------------------------------------------
class TicketBase(BaseModel):
    issue:        str         = Field(..., example="User cannot connect to WiFi.")
    status:       str         = Field(..., example="Open")
    resolution:   Optional[str] = Field(None, example="Reset WiFi adapter and reboot machine.")
    date_opened:  date        = Field(..., example="2025-06-03")
    date_closed:  Optional[date] = Field(None, example="2025-06-05")
    requester_id: Optional[int]  = Field(None, example=3)

class TicketCreate(TicketBase):
    pass

class TicketUpdate(BaseModel):
    issue:        Optional[str] = None
    status:       Optional[str] = None
    resolution:   Optional[str] = None
    date_opened:  Optional[date] = None
    date_closed:  Optional[date] = None
    requester_id: Optional[int]  = None

class TicketInDB(TicketBase):
    ticket_id: int
    class Config: from_attributes = True

class UserBase(BaseModel):
    full_name:  str = Field(..., example="Alice Johnson")
    email:      str = Field(..., example="alice.johnson@example.com")
    department: str = Field(..., example="IT")

class UserCreate(UserBase):
    pass

class UserUpdate(BaseModel):
    full_name:  Optional[str] = None
    email:      Optional[str] = None
    department: Optional[str] = None

class UserInDB(UserBase):
    user_id:    int
    created_at: date # This should match the DB schema (TEXT as ISO8601 date string)
    class Config: from_attributes = True

class ChatRequest(BaseModel):
    question: str = Field(..., example="My printer is showing a paper jam.")

class TextUploadRequest(BaseModel):
    filename: str = Field(..., example="knowledge_base.txt")
    content: str  = Field(..., example="To restart the main server, use the 'restart-service' command.")

# -------------------------------------------------------------
# UTILITY: get a SQLite connection (per-request)
# -------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_FILENAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# -------------------------------------------------------------
# /users endpoints (CRUD) - ADDED/COMPLETED
# -------------------------------------------------------------
@app.post("/users/", response_model=UserInDB, status_code=201)
def create_user_endpoint(payload: UserCreate, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    # Use current date for created_at, matching DB schema which expects TEXT
    created_at_iso = date.today().isoformat()
    try:
        cursor.execute(
            """
            INSERT INTO users(full_name, email, department, created_at)
            VALUES (?, ?, ?, ?);
            """,
            (payload.full_name, payload.email, payload.department, created_at_iso)
        )
        db.commit()
    except sqlite3.IntegrityError as e:
        # This might happen if email is set to UNIQUE in DB and there's a duplicate
        raise HTTPException(status_code=409, detail=f"Error creating user: {e}")
    
    new_id = cursor.lastrowid
    # Update the in-memory user_lookup
    user_lookup[new_id] = payload.full_name
    
    # Fetch the newly created user to return it
    cursor.execute("SELECT user_id, full_name, email, department, created_at FROM users WHERE user_id = ?;", (new_id,))
    new_user_row = cursor.fetchone()
    if new_user_row is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve user after creation.")
    return UserInDB(**dict(new_user_row))

@app.get("/users/", response_model=List[UserInDB])
def read_all_users_endpoint(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT user_id, full_name, email, department, created_at FROM users ORDER BY user_id ASC;")
    rows = cursor.fetchall()
    return [UserInDB(**dict(row)) for row in rows]

@app.get("/users/{user_id}", response_model=UserInDB)
def read_user_endpoint(user_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT user_id, full_name, email, department, created_at FROM users WHERE user_id = ?;", (user_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserInDB(**dict(row))

@app.put("/users/{user_id}", response_model=UserInDB)
def update_user_endpoint(user_id: int, payload: UserUpdate, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    # Check if user exists
    cursor.execute("SELECT * FROM users WHERE user_id = ?;", (user_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = payload.dict(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields provided for update.")

    set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
    values = list(update_data.values())
    values.append(user_id)

    try:
        cursor.execute(f"UPDATE users SET {set_clause} WHERE user_id = ?;", tuple(values))
        db.commit()
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail=f"Error updating user: {e}")

    # If full_name was updated, refresh the user_lookup for that user
    if "full_name" in update_data:
        user_lookup[user_id] = update_data["full_name"]

    # Fetch the updated user
    cursor.execute("SELECT user_id, full_name, email, department, created_at FROM users WHERE user_id = ?;", (user_id,))
    updated_user_row = cursor.fetchone()
    if updated_user_row is None: # Should not happen if update was successful
        raise HTTPException(status_code=500, detail="Failed to retrieve user after update.")
    return UserInDB(**dict(updated_user_row))

# -------------------------------------------------------------
# /tickets endpoints (CRUD)
# -------------------------------------------------------------
@app.get("/tickets/", response_model=List[TicketInDB])
def read_all_tickets(db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT ticket_id, issue, status, resolution, date_opened, date_closed, requester_id FROM tickets ORDER BY ticket_id ASC;")
    rows = cursor.fetchall()
    return [TicketInDB(**dict(row)) for row in rows]

@app.get("/tickets/{ticket_id}", response_model=TicketInDB) # Added specific ticket endpoint
def read_ticket(ticket_id: int, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("SELECT ticket_id, issue, status, resolution, date_opened, date_closed, requester_id FROM tickets WHERE ticket_id = ?;", (ticket_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return TicketInDB(**dict(row))

@app.post("/tickets/", response_model=TicketInDB, status_code=201)
def create_ticket(payload: TicketCreate, db: sqlite3.Connection = Depends(get_db)):
    cursor = db.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO tickets(issue, status, resolution, date_opened, date_closed, requester_id)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                payload.issue,
                payload.status,
                payload.resolution,
                payload.date_opened.isoformat(),
                payload.date_closed.isoformat() if payload.date_closed else None,
                payload.requester_id
            )
        )
        db.commit()
    except sqlite3.IntegrityError as e: # e.g. foreign key constraint for requester_id if it doesn't exist
        raise HTTPException(status_code=400, detail=f"Could not create ticket: {e}")

    new_id = cursor.lastrowid
    cursor.execute("SELECT ticket_id, issue, status, resolution, date_opened, date_closed, requester_id FROM tickets WHERE ticket_id = ?;", (new_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve ticket after creation.")
    return TicketInDB(**dict(row))

# -------------------------------------------------------------
# /upload-text endpoint (embed and add text file to FAISS)
# -------------------------------------------------------------
@app.post("/upload-text/")
def upload_text(request: TextUploadRequest):
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")
    try:
        resp = openai_client.embeddings.create(
            input=[request.content],
            model=EMBEDDING_MODEL
        )
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Embedding error: {e}")
    vector = np.array([resp.data[0].embedding], dtype="float32")
    index.add(vector)
    new_id = f"file-{len(ticket_mapping)}"
    new_entry = {
        "ticket_id": new_id, # Using 'ticket_id' for consistency in RAG source type
        "text": request.content,
        "filename": request.filename
    }
    ticket_mapping.append(new_entry)
    # These global lists might need thread-safety if running with multiple workers,
    # but for uvicorn default single worker, this is okay.
    ticket_ids.append(new_id)
    ticket_texts.append(request.content)
    return {"message": f"File '{request.filename}' embedded successfully.", "new_vector_id": new_id}

# -------------------------------------------------------------
# /chat endpoint (RAG-powered helpdesk)
# -------------------------------------------------------------
@app.post("/chat")
def chat(request: ChatRequest, db: sqlite3.Connection = Depends(get_db)):
    user_q = request.question.strip()
    if not user_q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        resp = openai_client.embeddings.create(input=[user_q], model=EMBEDDING_MODEL)
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Embedding error: {e}")
    q_vec = np.array([resp.data[0].embedding], dtype="float32")
    D, I = index.search(q_vec, TOP_K)
    neighbor_indices = I[0].tolist()
    retrieved_tickets = []
    cursor = db.cursor()
    for i, idx in enumerate(neighbor_indices):
        if 0 <= idx < len(ticket_mapping): # ticket_mapping now includes files
            entry = ticket_mapping[idx]
            tid_or_filename = entry["ticket_id"] # This is "file-N" for uploaded files
            blob = entry["text"]
            if "filename" in entry: # It's an uploaded file
                retrieved_tickets.append({
                    "ticket_id": tid_or_filename,
                    "issue": blob,
                    "resolution": "",
                    "requester": f"Uploaded File ({entry['filename']})",
                    "distance": float(D[0][i])
                })
            else: # It's a ticket from the DB
                issue_part, resolution_part = blob.split("\nResolution: ", 1) if "\nResolution: " in blob else (blob, "")
                cursor.execute("SELECT requester_id FROM tickets WHERE ticket_id = ?;", (tid_or_filename,))
                row = cursor.fetchone()
                requester_name = user_lookup.get(row["requester_id"]) if row and row["requester_id"] else None
                retrieved_tickets.append({
                    "ticket_id": tid_or_filename,
                    "issue": issue_part,
                    "resolution": resolution_part,
                    "requester": requester_name,
                    "distance": float(D[0][i])
                })
    system_msg = "You are an expert IT helpdesk assistant..." # (shortened for brevity)
    prompt = ""
    for i, rt in enumerate(retrieved_tickets, start=1):
        context_header = rt['requester'] if rt['requester'] else f"Source {rt['ticket_id']}"
        prompt += f"Snippet {i}: [{context_header}]\nIssue: {rt['issue']}\n"
        if rt['resolution']: prompt += f"Resolution: {rt['resolution']}\n"
        prompt += "\n"
    prompt += f"User question: {user_q}\n\nProvide a clear, concise resolution..."
    try:
        chat_resp = nim_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.2
        )
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"NVIDIA NIM ChatCompletion error: {e}")
    answer = chat_resp.choices[0].message.content
    return {"answer": answer, "retrieved": retrieved_tickets}

# -------------------------------------------------------------
# Serve static files (if `static/` exists)
# -------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Optional: main block to run with uvicorn for development
if __name__ == "__main__":
    import uvicorn
    print(f"Starting IT Helpdesk Backend on http://localhost:8001 (or your specified port)")
    # Ensure this port matches what your HTML expects for apiBase, or adjust HTML
    uvicorn.run(app, host="0.0.0.0", port=8000)