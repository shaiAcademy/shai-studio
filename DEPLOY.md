# Deployment Update Guide

Since the repository has moved to `shaiAcademy/shai-studio`, you need to update the git remote on your server (Google Cloud) to pull the latest changes.

## 1. Connect to your Server
Open your terminal and SSH into your Google Cloud instance.

## 2. Update Git Remote
Navigate to your project folder and run the following commands to switch the origin URL:

```bash
cd /path/to/your/shai-studio-folder  #/home/username/shai-studio or similar

# Check current remote
git remote -v 

# Set new remote URL
git remote set-url origin https://github.com/shaiAcademy/shai-studio.git

# Verify it changed

```

## 3. Pull Latest Changes
Now you can pull the new code (including the Docker fixes and Firebase integration):

```bash
git pull origin main
```

## 4. CRITICAL: Update Firebase Key
The file `fastapi_api/firebase_key.json` is **ignored by git** (for security). The changes we made to it locally are **NOT** on the server yet.

You must manually update this file on the server.

### Option A: Edit manually on server
1.  Copy the content of your local `firebase_key.json`.
2.  On the server:
    ```bash
    nano fastapi_api/firebase_key.json
    ```
3.  Paste the new content.
4.  Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

### Option B: Upload via SCP (run from your Local Machine)
```bash
scp ./fastapi_api/firebase_key.json username@your-server-ip:/path/to/shai-studio/fastapi_api/
```

## 5. Rebuild Docker
We made changes to `docker-compose.yml` and `Dockerfile`. You need to rebuild:

```bash
# Rebuild and restart in detached mode
docker-compose up -d --build
```

## 6. Verification
Check logs to make sure Firebase initialized correctly:

```bash
docker-compose logs -f fastapi_api
```
You should see: `✅ Firebase Admin initialized successfully`.
