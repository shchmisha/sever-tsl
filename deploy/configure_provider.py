"""Write the deployment credential through a hidden prompt, never argv or logs."""
import getpass
import os
from pathlib import Path

path = Path('/etc/sever-tsl/provider.env')
key = getpass.getpass('RouterAI API key (hidden; root-only service configuration): ').strip().rstrip('\\')
if not key or any(c in key for c in '\r\n\"\x00'):
    raise SystemExit('Invalid credential input')
path.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.fchmod(fd, 0o600)
with os.fdopen(fd, 'w') as stream:
    stream.write('HARNESS_MODEL_BASE_URL=https://routerai.ru/api/v1\n')
    stream.write('HARNESS_VISION_MODEL=openai/gpt-4.1-mini\n')
    stream.write('HARNESS_MODEL_API_KEY="' + key + '"\n')
key = ''
print('Provider configuration saved; mode 0600.')
