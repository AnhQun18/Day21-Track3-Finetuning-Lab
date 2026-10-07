from pathlib import Path
import argparse, os, json, subprocess, sys, hashlib
from huggingface_hub import HfApi, get_token
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('repo_id'); args=parser.parse_args()
repo_id=args.repo_id
assert '/' in repo_id and len(repo_id.split('/'))==2, 'Use USERNAME/REPOSITORY'
token=get_token()
if not token:
 try:
  from google.colab import userdata
  token=userdata.get('HF_TOKEN')
 except Exception:
  from getpass import getpass
  token=getpass('Hugging Face token with Write permission (hidden): ')
api=HfApi(token=token)
user=api.whoami()['name']; print('Authenticated Hugging Face user:',user)
os.environ['LAB21_HF_REPO']=repo_id
subprocess.run([sys.executable,str(ROOT/'scripts/build_submission.py')],check=True,env=os.environ.copy())
api.create_repo(repo_id=repo_id,repo_type='model',private=False,exist_ok=True)
assert not api.model_info(repo_id).private, 'Repository must be public for Option B bonus; adjust visibility yourself if using an existing private repository.'
api.upload_folder(repo_id=repo_id,repo_type='model',folder_path=str(ROOT/'adapters/correct'),allow_patterns=['adapter_config.json','adapter_model.safetensors','README.md','tokenizer*','special_tokens_map.json','added_tokens.json','chat_template.jinja'],commit_message='Upload Lab21 correct LoRA adapter and honest full evaluation card')
for folder in ['results','submission']:
 api.upload_folder(repo_id=repo_id,repo_type='model',folder_path=str(ROOT/folder),path_in_repo=folder,allow_patterns=['*.json','*.csv','*.md','*.log'],commit_message='Add Lab21 full evaluation evidence and report')
api.upload_file(repo_id=repo_id,repo_type='model',path_or_fileobj=str(ROOT/'LINKS.md'),path_in_repo='LINKS.md',commit_message='Add submission links')
info=api.model_info(repo_id,files_metadata=True)
names={x.rfilename for x in info.siblings}
assert {'adapter_config.json','adapter_model.safetensors','README.md','submission/REPORT.md','results/verdict.json'} <= names
weight=next(x for x in info.siblings if x.rfilename=='adapter_model.safetensors')
lfs=getattr(weight,'lfs',None)
remote_sha=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
local_sha=hashlib.sha256((ROOT/'adapters/correct/adapter_model.safetensors').read_bytes()).hexdigest()
assert remote_sha==local_sha, 'Remote adapter checksum did not match the trained artifact'
result={'adapter_sha256':local_sha,'repo_id':repo_id,'url':'https://huggingface.co/'+repo_id,'revision':info.sha,'private':info.private,'uploaded_files':sorted(names)}
(ROOT/'results/hub_upload.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('HUB_UPLOAD_VERIFIED',result['url'],'revision',info.sha)
