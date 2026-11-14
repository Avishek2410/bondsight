"""Create docker/.env with a new random database password (only if it does not exist yet).

What: PostgreSQL needs a password, and the inspector needs the same one. This writes both into
docker/.env, which git ignores, so the password never lands in the code or on GitHub.
Run it once before the first `docker compose up`:   python3 docker/make_env.py
The password is not printed. To see it later, open docker/.env on this machine.
The file is created already private (owner read/write only, on Linux and the Pi), so the
password is never readable by other users, not even for a moment.
"""
import os
import secrets
import sys
from pathlib import Path

env = Path(__file__).resolve().parent / ".env"
pw = secrets.token_urlsafe(24)
try:
    fd = os.open(env, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)   # fails if it already exists
except FileExistsError:
    print(f"{env} already exists; left unchanged")
    sys.exit(0)
with os.fdopen(fd, "w") as f:
    f.write(f"# made by make_env.py - keep private, never commit\nPOSTGRES_PASSWORD={pw}\nWBI_DB_PASSWORD={pw}\n")
if os.name == "posix" and (os.stat(env).st_mode & 0o077):
    env.unlink()
    sys.exit(f"could not make {env} private; removed it. Check the folder's permissions and run again.")
print(f"created {env}")
