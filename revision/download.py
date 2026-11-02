"""Fetch the versioned experiment artifacts and verify the archive SHA-256."""
from pathlib import Path
import hashlib,json,tarfile,urllib.request
ROOT=Path(__file__).resolve().parent

def main():
 meta=json.loads((ROOT/'artifact_download.json').read_text());dest=ROOT/'downloads';dest.mkdir(exist_ok=True)
 archive=dest/meta['filename']
 if not archive.exists():
  part=archive.with_suffix('.partial')
  print('Downloading',meta['url'],flush=True)
  urllib.request.urlretrieve(meta['url'],part);part.replace(archive)
 with archive.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 if actual!=meta['sha256']:raise ValueError('Archive checksum differs; remove the download and retry.')
 with tarfile.open(archive,'r:gz') as tf:
  for member in tf.getmembers():
   target=(ROOT/member.name).resolve()
   if not target.is_relative_to(ROOT) or member.issym() or member.islnk():raise ValueError('Unsafe archive member')
  tf.extractall(ROOT,filter='data')
 print('Artifacts extracted. Run: python revision/verify.py --replay')
if __name__=='__main__':main()
