import secrets
import base64
from cryptography.fernet import Fernet
import os

# Generate or load encryption key
def get_encryption_key():
    """Get or generate encryption key for asymmetric encryption simulation"""
    key_file = 'encryption_key.key'
    if os.path.exists(key_file):
        with open(key_file, 'rb') as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(key_file, 'wb') as f:
            f.write(key)
        return key

# Initialize cipher
ENCRYPTION_KEY = get_encryption_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

def generate_task_token(task_id):
    """Generate encrypted task token using task ID"""
    # Create a token with task ID and random signature
    token_data = f"{task_id}:{secrets.token_hex(16)}"
    encrypted_token = cipher_suite.encrypt(token_data.encode())
    return base64.urlsafe_b64encode(encrypted_token).decode()

def verify_task_token(token, task_id):
    """Verify task token matches the task ID"""
    try:
        encrypted_data = base64.urlsafe_b64decode(token.encode())
        decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
        stored_task_id, signature = decrypted_data.split(':')
        return stored_task_id == str(task_id)
    except Exception:
        return False

def save_token_to_file(token, task_id):
    """Save token to file for download"""
    filename = f"task_{task_id}.token"
    filepath = f"/tmp/{filename}"
    with open(filepath, 'w') as f:
        f.write(token)
    return filepath

def read_token_from_file(uploaded_file):
    """Read token from uploaded file"""
    try:
        content = uploaded_file.read()
        return content.decode('utf-8').strip()
    except AttributeError:
        # If it's not a file object, try opening as path
        with open(uploaded_file, 'r') as f:
            return f.read().strip()