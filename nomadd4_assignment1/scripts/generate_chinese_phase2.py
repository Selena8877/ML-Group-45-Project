import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase1', required=True, type=Path)
    parser.add_argument('--out-dir', default='results/phase2', type=Path)
    parser.add_argument('--sampled-ids', default='ablation/sampled_ids.json', type=Path)
    args = parser.parse_args()

    data = json.loads(args.phase1.read_text(encoding='utf-8'))
    posts = data['samples']
    assert len(posts) == 4860
    assert Counter(p['ground_truth'] for p in posts) == Counter({'depressed': 2430, 'not depressed': 2430})
    assert len({p['index'] for p in posts}) == len(posts)
    assert len({p['post'] for p in posts}) == len(posts)
    assert all(p['ground_truth'] == ('depressed' if p['label'] == 'depression' else 'not depressed') for p in posts)

    rng = random.Random(42)
    depressed = [p for p in posts if p['label'] == 'depression']
    normal = [p for p in posts if p['label'] == 'normal']
    selected = rng.sample(depressed, 125) + rng.sample(normal, 125)
    rng.shuffle(selected)
    assert len(selected) == len({p['index'] for p in selected}) == len({p['post'] for p in selected}) == 250

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = {'metadata': {'language': 'chinese', 'experiment': 'phase2', 'dataset': 'DepGLM (DW/train.csv)', 'source_phase1': args.phase1.name, 'seed': 42, 'total_sampled': 250, 'depressed_count': 125, 'normal_count': 125, 'sampling_method': 'balanced_random', 'sampling_note': 'No new model predictions; 45/45/10 prediction-based stratification unavailable'}, 'samples': selected}
    output_file = args.out_dir / 'chinese_depglm_phase2_seed42.json'
    output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    ids = [p['index'] for p in selected]
    args.sampled_ids.parent.mkdir(parents=True, exist_ok=True)
    if args.sampled_ids.exists():
        all_ids = json.loads(args.sampled_ids.read_text(encoding='utf-8'))
        if not isinstance(all_ids, dict) or 'chinese' not in all_ids or not isinstance(all_ids['chinese'], list):
            raise ValueError('Inspect sampled_ids.json manually: expected a dictionary with a chinese list')
        all_ids['chinese'] = ids
        args.sampled_ids.write_text(json.dumps(all_ids, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('Updated only Chinese indices:', args.sampled_ids)
    else:
        patch = args.sampled_ids.parent / 'chinese_sampled_ids_patch.json'
        patch.write_text(json.dumps({'chinese': ids}, indent=2) + '\n', encoding='utf-8')
        print('No original sampled_ids.json found; created Chinese-only patch:', patch)
    print('PASS: 250 unique posts, 125 depressed and 125 not depressed, seed 42')
    print('JSON:', output_file)


if __name__ == '__main__':
    main()