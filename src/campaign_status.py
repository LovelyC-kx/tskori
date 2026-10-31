"""Inspect exact requested run IDs using only the Python standard library."""
import argparse,json
from pathlib import Path

def inspect(runs,manifest=None):
    root=Path(runs);request=root/'requested_cases.json'
    planned=len(json.loads(Path(manifest).read_text())['jobs']) if manifest else None
    if not request.exists():
        return dict(runs=str(root),planned=planned,requested=0,complete=0,pending=planned,failed=0,all_requested_complete=False)
    payload=json.loads(request.read_text());ids=payload['case_ids']
    if len(ids)!=len(set(ids)):raise RuntimeError('Duplicate requested case IDs.')
    done=[];failed=[];active=[]
    for ident in ids:
        folder=root/ident
        if (folder/'complete.json').is_file():done.append(ident)
        elif (folder/'failed.json').is_file():failed.append(ident)
        elif folder.exists():active.append(ident)
    complete=len(done)==len(ids) and len(ids)>0 and (planned is None or len(ids)==planned)
    return dict(runs=str(root),planned=planned,requested=len(ids),complete=len(done),pending=len(ids)-len(done),
                failed=len(failed),started_incomplete=active,failed_case_ids=failed,all_requested_complete=complete)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',required=True);ap.add_argument('--manifest');ap.add_argument('--require-complete',action='store_true')
    args=ap.parse_args();result=inspect(args.runs,args.manifest);print(json.dumps(result,indent=2))
    if args.require_complete and not result['all_requested_complete']:raise SystemExit(2)

if __name__=='__main__':main()
