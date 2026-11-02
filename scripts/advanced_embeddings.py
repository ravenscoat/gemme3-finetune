"""Frozen GPT-2-large features for an experimental text PRDC adaptation."""
import importlib.metadata
import json
import os
from pathlib import Path
import time

from dataset_catalog import ROOT,read_json,sha
from json_io import write_json

OUT=ROOT/'reports/advanced_analysis/v1'


def main():
    lock=ROOT/'reports/assessment_gpu.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    state={'phase':'loading','completed':0,'total':16};started=time.time()
    try:
        protocol=read_json(OUT/'protocol.json');settings=protocol['embedding']
        if sha(OUT/'embedding_inputs.json')!=settings['input_sha256']:raise RuntimeError('Embedding inputs changed')
        for path,digest in protocol['source_hashes'].items():
            if sha(ROOT/path)!=digest:raise RuntimeError('Frozen source changed: '+path)
        inputs_all=read_json(OUT/'embedding_inputs.json')
        write_json(OUT/'embedding_progress.json',state,best_effort=True)
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel,AutoTokenizer
        model_path=Path(snapshot_download(settings['feature_model'],revision=settings['feature_model_revision'],local_files_only=True))
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
        model=AutoModel.from_pretrained(model_path,torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
        cache=OUT/'features';cache.mkdir(exist_ok=True)
        result={'protocol':settings,'model_weights_sha256':sha(model_path/'model.safetensors'),
                'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},'features':{}}
        completed=0
        for key,collections in inputs_all.items():
            result['features'][key]={}
            for label,fields in collections.items():
                result['features'][key][label]={}
                for field,texts in fields.items():
                    name=key+'_'+label+'_'+field;target=cache/(name+'.npy')
                    lengths=[len(tokenizer.encode(text)) for text in texts]
                    if not target.exists():
                        batches=[]
                        for i in range(0,len(texts),settings['batch_size']):
                            batch=tokenizer(texts[i:i+settings['batch_size']],return_tensors='pt',padding=True,truncation=True,max_length=settings['max_tokens']).to('cuda')
                            with torch.inference_mode():
                                hidden=model(**batch).last_hidden_state
                                terminal=batch['attention_mask'].sum(dim=1)-1
                                features=hidden[torch.arange(hidden.shape[0],device='cuda'),terminal].float()
                                features=features/torch.linalg.vector_norm(features,dim=1,keepdim=True).clamp_min(1e-12)
                                batches.append(features.cpu().numpy())
                            if i%32==0:print(name,f'{min(i+settings["batch_size"],len(texts))}/{len(texts)}',flush=True)
                        np.save(target,np.concatenate(batches))
                    array=np.load(target)
                    if array.shape[0]!=len(texts) or not np.isfinite(array).all():raise RuntimeError('Feature cache is incomplete or non-finite')
                    result['features'][key][label][field]={'path':str(target.relative_to(ROOT)),'sha256':sha(target),'n':len(texts),
                                                         'truncated':sum(v>settings['max_tokens'] for v in lengths),'max_original_tokens':max(lengths)}
                    completed+=1
                    state.update(phase='embedding',completed=completed,current=name,elapsed_seconds=round(time.time()-started,2))
                    write_json(OUT/'embedding_progress.json',state,best_effort=True)
        write_json(OUT/'embedding_manifest.json',result)
        state.update(phase='complete');print('All frozen features saved.',flush=True)
    except BaseException as exc:state.update(phase='failed',error=str(exc));raise
    finally:
        write_json(OUT/'embedding_progress.json',state,best_effort=True)
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()
