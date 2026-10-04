from __future__ import annotations
import argparse, json, yaml
from pathlib import Path
from autogad_reproduction.data import load_mat
from autogad_reproduction.experiments import run_search, evaluate_selected, dump_result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--search-space",default="configs/search_spaces.yaml")
    p.add_argument("--model",default="anemone")
    p.add_argument("--k",type=int,required=True)
    p.add_argument("--max-configs",type=int,default=None)
    p.add_argument("--runs",type=int,default=5)
    p.add_argument("--output-dir",default="outputs/phase1")
    args=p.parse_args()
    spaces=yaml.safe_load(Path(args.search_space).read_text())
    defaults=yaml.safe_load(Path("configs/experiments.yaml").read_text())["anemone"]
    graph=load_mat(args.dataset)
    root=Path(args.output_dir); root.mkdir(parents=True,exist_ok=True)
    records=[]
    for run_id in range(args.runs):
        defaults_run=dict(defaults); defaults_run["seed"]=42+run_id
        result=run_search(graph,args.model,spaces[args.model if args.model in spaces else "anemone"],defaults_run,args.k,args.max_configs)
        path=root/f"run_{run_id+1}"
        dump_result(result,path)
        scores,auc=evaluate_selected(graph,args.model,result.best_params,defaults_run)
        records.append({"run":run_id+1,"best_params":result.best_params,"csm":result.best_csm,"auc":auc})
    (root/"summary.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    print(json.dumps(records,indent=2))

if __name__=="__main__": main()
