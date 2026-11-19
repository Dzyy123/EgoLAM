"""Release-candidate technical validation; does not claim human relabeling/audit."""
import argparse,csv,json
from pathlib import Path
from collections import Counter
from .inventory import write,sha
from .build import noun,VERBS
from abilities.train import validate_rows

def main(a):
 root=Path(a.work);data=root/'composition_rc3';m=json.loads((data/'manifest.json').read_text());lock=json.loads((data/'release_lock.json').read_text());assert lock['manifest_sha256']==sha(data/'manifest.json') and lock['extraction_csv_sha256']==sha(data/'extraction.csv')
 ontology=json.loads((data/'object_ontology.json').read_text());assert sha(data/'object_ontology.json')==m['object_ontology_sha256']
 rows=validate_rows(m,'composition');v={x['sample_id']:x for x in m['videos']};inv={x['sample_id']:x for x in map(json.loads,(root/'inventory/mapping.jsonl').read_text().splitlines())}
 group_sets={s:{v[r['sample_id']]['source_video_id'] for r in rows if r['split']==s} for s in ['train','dev','test']}
 assert not (group_sets['train']&group_sets['dev'] or group_sets['train']&group_sets['test'] or group_sets['dev']&group_sets['test'])
 for r in rows:
  vi=v[r['sample_id']];original=inv[r['sample_id']];assert vi['frame_indices']==original['frame_indices'] and vi['video_path']==original['video_path'];assert not original['reversed'];assert len(original['placeholders'])==1
  assert ontology['aliases'][noun(original['placeholders'][0])]==m['composition']['nouns'][r['noun']]
  assert original['upstream_action']==m['composition']['verbs'][r['verb']]==VERBS[original['template'].split()[0]]
  if r['split']=='test':assert original['upstream_split']=='val' and not original['pilot_exposed']
  else:assert original['upstream_split']=='train' and not original['source_crosses_upstream_splits']
 sourceproof=json.loads((root/'inventory/source_proof.json').read_text());assert all(sourceproof['files'][f'data/{n}']['matches'] for n in ['seed_ssv2.csv','human_1st_metadata_train.csv','human_1st_metadata_val.csv'])
 assert json.loads((root/'training_smoke.json').read_text())['passed']
 tests=[r for r in rows if r['split']=='test'];unseen=[r for r in tests if r['unseen_pair']]
 report={'protocol_version':m['protocol_version'],'manifest_sha256':sha(data/'manifest.json'),'technical_checks_passed':True,'technical_checks':['pinned official metadata and seed identity','all selected clips remain at original paths and frame indices','published single-object source annotation plus matching LARY action','no source-video overlap across splits','all exploratory pilot sources excluded from test','unseen combinations absent from probe training; all components train-seen','native full-batch head and scheduler numerical smoke passed'],'counts':dict(Counter(r['split'] for r in rows)),'unseen_test':len(unseen),'unseen_by_design':sum(r.get('held_by_design',False) for r in unseen),'naturally_unseen':sum(not r.get('held_by_design',False) for r in unseen),'independent_integrity_review':'pending','human_reannotation':'not performed; uses declared published-source transfer','scope':'SSv2 composition track only; other axes not released','upstream_endorsement':False}
 write(root/'dataset_validation.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',required=True);main(p.parse_args())
