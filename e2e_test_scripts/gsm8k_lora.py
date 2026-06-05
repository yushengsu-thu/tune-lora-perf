import argparse, time, requests
from concurrent.futures import ThreadPoolExecutor
from sglang.test.few_shot_gsm8k import get_one_example, get_few_shot_examples, get_answer_value, INVALID
from sglang.utils import download_and_cache_file, read_jsonl
URL="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
STOP=["Question","Assistant:","<|separator|>"]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--num-questions",type=int,default=200); ap.add_argument("--num-shots",type=int,default=5)
    ap.add_argument("--parallel",type=int,default=32); ap.add_argument("--port",type=int,default=30000)
    ap.add_argument("--host",default="http://127.0.0.1"); ap.add_argument("--max-new-tokens",type=int,default=512)
    ap.add_argument("--lora",default=""); ap.add_argument("--chat",action="store_true"); ap.add_argument("--model",default="/root/Kimi-K2.5-NVFP4")
    a=ap.parse_args()
    lines=list(read_jsonl(download_and_cache_file(URL))); fs=get_few_shot_examples(lines,a.num_shots); base=f"{a.host}:{a.port}"
    def one(i):
        q=get_one_example(lines,i,False); gold=get_answer_value(lines[i]["answer"])
        try:
            if a.chat:
                model=f"{a.model}:{a.lora}" if a.lora else a.model
                r=requests.post(f"{base}/v1/chat/completions",json={"model":model,"messages":[{"role":"user","content":fs+q}],"temperature":0,"max_tokens":a.max_new_tokens,"stop":STOP},timeout=900).json()
                txt=(r["choices"][0]["message"].get("content") or ""); ct=r.get("usage",{}).get("completion_tokens",0)
            else:
                d={"text":fs+q,"sampling_params":{"max_new_tokens":a.max_new_tokens,"temperature":0,"stop":STOP}}
                if a.lora: d["lora_path"]=a.lora
                r=requests.post(f"{base}/generate",json=d,timeout=900).json(); txt=(r.get("text") or ""); ct=r.get("meta_info",{}).get("completion_tokens",0)
        except Exception as e: return False, 0, f"ERR:{e}"
        return get_answer_value(txt)==gold, ct, txt
    tic=time.perf_counter()
    with ThreadPoolExecutor(a.parallel) as ex: res=list(ex.map(one,range(a.num_questions)))
    lat=time.perf_counter()-tic
    correct=sum(1 for c,_,_ in res if c); trunc=sum(1 for _,ct,_ in res if ct>=a.max_new_tokens); empty=sum(1 for _,ct,_ in res if ct<=1)
    print(f"Accuracy: {correct/len(res):.3f}"); print(f"Truncated(>={a.max_new_tokens}): {trunc}/{len(res)}"); print(f"EOS-empty(ct<=1): {empty}/{len(res)}")
    print(f"Mode: {'chat' if a.chat else 'raw'}  lora={a.lora or 'BASE'}  osl={a.max_new_tokens}  Latency: {lat:.1f}s")
main()
