import os

# Define the directory structure
folders = [
    "app/api/routers", 
    "app/core", 
    "app/models",
    "app/schemas", 
    "app/services", 
    "app/utils"
]

# Define the files to create
files = [
    "app/api/dependencies.py", "app/api/routers/__init__.py", "app/api/routers/chat.py",
    "app/core/__init__.py", "app/core/config.py", "app/core/database.py", "app/core/security.py",
    "app/models/__init__.py", "app/models/session.py", "app/models/message.py",
    "app/schemas/__init__.py", "app/schemas/chat.py", "app/schemas/travel.py",
    "app/services/__init__.py", "app/services/llm_service.py", "app/services/rag_service.py", "app/services/api_client.py",
    "app/utils/__init__.py", "app/utils/prompts.py",
    "app/main.py", ".env", ".gitignore", "requirements.txt", "README.md"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file in files:
    with open(file, 'w') as f:
        pass # Creates an empty file

print("✨ FastAPI project scaffolded successfully!")