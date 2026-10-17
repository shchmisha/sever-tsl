"""Fetch public Yandex folder videos on the deployment host; verify every SHA."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = 'https://cloud-api.yandex.net/v1/disk/public/resources'

def metadata(url):
    with urlopen(Request(url, headers={'User-Agent': 'SeverTSL/0.1'}), timeout=60) as response:
        return json.load(response)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def run(public_key, destination, provenance):
    destination.mkdir(parents=True, exist_ok=True)
    pending = ['/']
    files = []
    while pending:
        parent = pending.pop()
        offset = 0
        while True:
            result = metadata(API + '?' + urlencode({'public_key': public_key, 'path': parent, 'limit': 100, 'offset': offset}))
            children = result.get('_embedded', {}).get('items', [])
            for item in children:
                if item['type'] == 'dir':
                    pending.append(item['path'])
                elif Path(item['name']).suffix.lower() in {'.mp4', '.mov', '.mkv', '.avi'}:
                    files.append(item)
            if len(children) < 100:
                break
            offset += len(children)
    names = [f['name'] for f in files]
    if len(names) != len(set(names)) or any(Path(n).name != n for n in names):
        raise ValueError('Video names must be unique safe basenames')

    def download(item):
        target = destination / item['name']
        expected = item['sha256']
        if target.is_file() and target.stat().st_size == item['size'] and digest(target) == expected:
            status = 'existing_copy_verified_against_public_source'
        else:
            link = metadata(API + '/download?' + urlencode({'public_key': public_key, 'path': item['path']}))
            temporary = target.with_suffix(target.suffix + '.download')
            with urlopen(link['href'], timeout=120) as response, temporary.open('wb') as stream:
                while block := response.read(4 * 1024 * 1024):
                    stream.write(block)
            if temporary.stat().st_size != item['size'] or digest(temporary) != expected:
                raise RuntimeError('Download integrity mismatch: ' + item['name'])
            temporary.replace(target)
            status = 'downloaded_on_server'
        row = {'name': item['name'], 'resource_path': item['path'], 'size': item['size'],
               'sha256': expected, 'status': status, 'destination': str(target)}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=3) as executor:
        verified = list(executor.map(download, files))
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps({'public_folder': public_key, 'files': verified}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('public_key')
    parser.add_argument('--destination', type=Path, default=Path('/srv/sever/input'))
    parser.add_argument('--provenance', type=Path, default=Path('/srv/sever/validation/download-provenance.json'))
    args = parser.parse_args()
    run(args.public_key, args.destination, args.provenance)
