import os
import json
import uuid
import base64
import traceback

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from crypto_utils import (
    generate_rsa_keypair,
    encrypt_document,
    decrypt_document,
    wrap_key_for_recipient,
    unwrap_key_for_recipient,
    sign_event,
    add_audit_event,
    sha256_bytes,
    load_audit,
    verify_audit_chain,
)

from face_utils import enroll_user, verify_user
from database import add_user, user_exists, get_user


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="CRYPTORA",
    description="Cryptographic Attribution & Immutable Decryption Provenance",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")


# ============================================================
# STATIC FILES
# ============================================================

if os.path.exists("static"):
    app.mount(
        "/static",
        StaticFiles(directory="static"),
        name="static"
    )


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/decrypted", exist_ok=True)
os.makedirs("data/keys", exist_ok=True)
os.makedirs("data/faces", exist_ok=True)
os.makedirs("data/audit", exist_ok=True)


DOCUMENT_META = "data/documents/metadata.json"


# ============================================================
# METADATA HELPERS
# ============================================================

def load_metadata():
    try:
        if not os.path.exists(DOCUMENT_META):
            return {}

        with open(DOCUMENT_META, "r") as f:
            return json.load(f)

    except Exception:
        traceback.print_exc()
        return {}


def save_metadata(data):
    with open(DOCUMENT_META, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# HOME
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    try:
        return templates.TemplateResponse(
            "index.html",
            {"request": request}
        )

    except Exception as e:

        traceback.print_exc()

        return HTMLResponse(
            content=f"""
            <html>
                <body>
                    <h1>CRYPTORA</h1>
                    <h3>Frontend could not be loaded.</h3>
                    <p>{str(e)}</p>
                </body>
            </html>
            """,
            status_code=500
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": "CRYPTORA"
    }


# ============================================================
# FACE ENROLLMENT / REGISTER USER
# ============================================================

@app.post("/enroll")
async def enroll(
    username: str = Form(...),
    image: UploadFile = File(...)
):

    try:

        # -----------------------------
        # Username validation
        # -----------------------------

        username = username.strip().lower()

        if not username:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Username required."
                }
            )

        # -----------------------------
        # Check duplicate user
        # -----------------------------

        if user_exists(username):

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "User already exists."
                }
            )

        # -----------------------------
        # Validate image
        # -----------------------------

        if image is None:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Face image is required."
                }
            )

        image_bytes = await image.read()

        if not image_bytes:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Empty image received."
                }
            )

        print(f"[ENROLL] User: {username}")
        print(f"[ENROLL] Image size: {len(image_bytes)} bytes")

        # -----------------------------
        # Face enrollment
        # -----------------------------

        success, message = enroll_user(
            username,
            image_bytes
        )

        if not success:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": message
                }
            )

        # -----------------------------
        # Add user to database
        # -----------------------------

        add_user(username)

        # -----------------------------
        # Generate RSA key pair
        # -----------------------------

        generate_rsa_keypair(username)

        # -----------------------------
        # Add audit event
        # -----------------------------

        add_audit_event(
            "RECIPIENT_REGISTERED",
            username,
            "identity",
            sha256_bytes(username.encode())
        )

        print(f"[ENROLL] SUCCESS: {username}")

        return {
            "success": True,
            "message": f"{username} registered successfully."
        }

    except Exception as e:

        print("\n========== ENROLL ERROR ==========")
        traceback.print_exc()
        print("==================================\n")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Registration error: {str(e)}"
            }
        )


# ============================================================
# ENCRYPT DOCUMENT
# ============================================================

@app.post("/encrypt")
async def encrypt(
    document: UploadFile = File(...),
    recipients: str = Form(...)
):

    try:

        # -----------------------------
        # Read document
        # -----------------------------

        data = await document.read()

        if not data:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Empty document."
                }
            )

        # -----------------------------
        # Parse recipients
        # -----------------------------

        recipient_list = [
            x.strip().lower()
            for x in recipients.split(",")
            if x.strip()
        ]

        if not recipient_list:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Add at least one recipient."
                }
            )

        # -----------------------------
        # Check recipients
        # -----------------------------

        for recipient in recipient_list:

            if not user_exists(recipient):

                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": f"{recipient} is not registered."
                    }
                )

        print(
            f"[ENCRYPT] Document: {document.filename}"
        )

        print(
            f"[ENCRYPT] Recipients: {recipient_list}"
        )

        # -----------------------------
        # AES encryption
        # -----------------------------

        aes_key, nonce, encrypted = encrypt_document(data)

        # -----------------------------
        # Generate document ID
        # -----------------------------

        document_id = str(uuid.uuid4())

        encrypted_path = os.path.join(
            "data",
            "documents",
            f"{document_id}.enc"
        )

        # -----------------------------
        # Save encrypted document
        # -----------------------------

        with open(encrypted_path, "wb") as f:
            f.write(encrypted)

        # -----------------------------
        # Document hash
        # -----------------------------

        document_hash = sha256_bytes(data)

        # -----------------------------
        # Wrap AES key for each recipient
        # -----------------------------

        wrapped_keys = {}

        for recipient in recipient_list:

            wrapped_keys[recipient] = wrap_key_for_recipient(
                aes_key,
                recipient
            )

        # -----------------------------
        # Save metadata
        # -----------------------------

        metadata = load_metadata()

        metadata[document_id] = {

            "document_id": document_id,

            "name": document.filename,

            "encrypted_file": encrypted_path,

            "nonce": base64.b64encode(
                nonce
            ).decode(),

            "document_hash": document_hash,

            "recipients": recipient_list,

            "wrapped_keys": wrapped_keys
        }

        save_metadata(metadata)

        # -----------------------------
        # Audit event
        # -----------------------------

        add_audit_event(
            "DOCUMENT_ENCRYPTED",
            "owner",
            document.filename,
            document_hash
        )

        print(
            f"[ENCRYPT] SUCCESS: {document_id}"
        )

        return {

            "success": True,

            "document_id": document_id,

            "document": document.filename,

            "recipients": recipient_list,

            "hash": document_hash
        }

    except Exception as e:

        print("\n========== ENCRYPT ERROR ==========")
        traceback.print_exc()
        print("===================================\n")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Encryption error: {str(e)}"
            }
        )


# ============================================================
# DECRYPT DOCUMENT
# ============================================================

@app.post("/decrypt")
async def decrypt(
    document_id: str = Form(...),
    username: str = Form(...),
    image: UploadFile = File(...)
):

    try:

        username = username.strip().lower()

        # -----------------------------
        # Check user
        # -----------------------------

        user = get_user(username)

        if not user:

            return {
                "success": False,
                "message": "User not found."
            }

        if not user["active"]:

            return {
                "success": False,
                "message": "User access revoked."
            }

        # -----------------------------
        # Load metadata
        # -----------------------------

        metadata = load_metadata()

        if document_id not in metadata:

            return {
                "success": False,
                "message": "Document not found."
            }

        document = metadata[document_id]

        # -----------------------------
        # Authorization check
        # -----------------------------

        if username not in document["recipients"]:

            add_audit_event(
                "UNAUTHORIZED_DECRYPTION_ATTEMPT",
                username,
                document["name"],
                document["document_hash"]
            )

            return {
                "success": False,
                "message": "You are not authorized for this document."
            }

        # -----------------------------
        # Read face image
        # -----------------------------

        image_bytes = await image.read()

        if not image_bytes:

            return {
                "success": False,
                "message": "Empty face image received."
            }

        print(
            f"[DECRYPT] Verifying face for: {username}"
        )

        # -----------------------------
        # Face verification
        # -----------------------------

        matched, confidence = verify_user(
            username,
            image_bytes
        )

        print(
            f"[DECRYPT] Match: {matched}"
        )

        print(
            f"[DECRYPT] Confidence: {confidence}"
        )

        if not matched:

            add_audit_event(
                "FACE_VERIFICATION_FAILED",
                username,
                document["name"],
                document["document_hash"]
            )

            return {
                "success": False,
                "message": "Face verification failed.",
                "confidence": round(confidence, 2)
            }

        # -----------------------------
        # Get wrapped AES key
        # -----------------------------

        wrapped_key = document["wrapped_keys"][username]

        aes_key = unwrap_key_for_recipient(
            wrapped_key,
            username
        )

        # -----------------------------
        # Read encrypted document
        # -----------------------------

        encrypted_file = document["encrypted_file"]

        if not os.path.exists(encrypted_file):

            return {
                "success": False,
                "message": "Encrypted file not found."
            }

        with open(encrypted_file, "rb") as f:

            encrypted_data = f.read()

        # -----------------------------
        # Decode nonce
        # -----------------------------

        nonce = base64.b64decode(
            document["nonce"]
        )

        # -----------------------------
        # AES decryption
        # -----------------------------

        decrypted = decrypt_document(
            aes_key,
            nonce,
            encrypted_data
        )

        # -----------------------------
        # Integrity verification
        # -----------------------------

        calculated_hash = sha256_bytes(
            decrypted
        )

        if calculated_hash != document["document_hash"]:

            return {
                "success": False,
                "message": "Document integrity verification failed."
            }

        # -----------------------------
        # Sign decryption event
        # -----------------------------

        event_text = (
            f"{document_id}|"
            f"{username}|"
            f"{document['document_hash']}"
        )

        signature = sign_event(
            event_text,
            username
        )

        # -----------------------------
        # Audit event
        # -----------------------------

        event = add_audit_event(
            "DOCUMENT_DECRYPTED",
            username,
            document["name"],
            document["document_hash"],
            signature
        )

        # -----------------------------
        # Save decrypted document
        # -----------------------------

        output_name = (
            f"{username}_{document['name']}"
        )

        output_path = os.path.join(
            "data",
            "decrypted",
            output_name
        )

        with open(output_path, "wb") as f:

            f.write(decrypted)

        print(
            f"[DECRYPT] SUCCESS: {output_name}"
        )

        return {

            "success": True,

            "message":
                "Face verified and document decrypted.",

            "confidence":
                round(confidence, 2),

            "event_id":
                event["event_id"],

            "signature":
                "VALID",

            "download":
                f"/download/{output_name}"
        }

    except Exception as e:

        print("\n========== DECRYPT ERROR ==========")
        traceback.print_exc()
        print("===================================\n")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Decryption error: {str(e)}"
            }
        )


# ============================================================
# DOWNLOAD DECRYPTED DOCUMENT
# ============================================================

@app.get("/download/{filename}")
async def download(filename: str):

    try:

        path = os.path.join(
            "data",
            "decrypted",
            filename
        )

        if not os.path.exists(path):

            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "File not found."
                }
            )

        return FileResponse(
            path,
            filename=filename
        )

    except Exception as e:

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Download error: {str(e)}"
            }
        )


# ============================================================
# AUDIT LEDGER
# ============================================================

@app.get("/audit")
async def audit():

    try:

        events = load_audit()

        return {

            "ledger_status":
                "VALID"
                if verify_audit_chain()
                else "COMPROMISED",

            "events": events
        }

    except Exception as e:

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Audit error: {str(e)}"
            }
        )


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    print("\n========== GLOBAL ERROR ==========")

    traceback.print_exc()

    print("==================================\n")

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": f"Internal server error: {str(exc)}"
        }
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )