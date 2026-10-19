"""Training job planning and execution."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys
ROOT = Path(__file__).resolve().parent
MODELS = ['hrgp', 'lstm', 'lstm_nores', 'mclstm', 'dhbv', 'hbv']

def commands(args):
    cfg = json.loads((ROOT / 'configs/experiment.json').read_text(encoding='utf-8'))
    for model in args.models:
        for lead in [None] if model == 'hbv' else args.leads:
            for seed in args.seeds:
                if model == 'hbv':
                    cmd = [sys.executable, str(ROOT / 'src/hbv/run.py'), '--data', str(args.data), '--output', str(args.output / 'hbv'), '--seed', str(seed)]
                elif model == 'hrgp':
                    cmd = [sys.executable, str(ROOT / 'src/hrgp/train.py'), '--data', str(args.data), '--output', str(args.output / model / f'lead_{lead}' / f'seed_{seed}'), '--device', args.device, '--lead', str(lead), '--seed', str(seed), '--max-epochs', '160', '--patience', '20', '--min-delta', '0.00001', '--batch-size', '32', '--frontend', 'local_mlp', '--variant', 'full', '--travel-cap', '8', '--initial-time', '1', '--time-policy', 'flow', '--routing-lr-multiplier', '20', '--streams', '8', '--unroll', '4']
                else:
                    cmd = [sys.executable, str(ROOT / 'src/neural_baselines/train.py'), '--data', str(args.data), '--output', str(args.output / model / f'lead_{lead}' / f'seed_{seed}'), '--model', model, '--lead', str(lead), '--seed', str(seed), '--device', args.device]
                if args.action == 'preflight':
                    cmd += ['--preflight-only']
                elif args.action == 'evaluate':
                    cmd += ['--eval-only']
                if args.evaluate_test and args.action in ['train', 'evaluate']:
                    cmd += ['--evaluate-test']
                yield dict(model=model, lead=lead, seed=seed, command=cmd)

def verify():
    manifest = ROOT / 'manifest.json'
    if not manifest.exists():
        raise FileNotFoundError('Missing package manifest')
    for name, expected in json.loads(manifest.read_text(encoding='utf-8'))['files'].items():
        h = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if h != expected:
            raise ValueError('Package file changed: ' + name)


def check_data(folder, models):
    required = {}
    groups = set('hrgp' if m == 'hrgp' else 'hbv' if m == 'hbv' else 'neural_baselines' for m in models)
    for group in sorted(groups):
        profile = json.loads((ROOT / 'src' / group / 'expected_data.json').read_text(encoding='utf-8'))
        for name in profile['required_files']:
            expected = profile['reference_hashes'][name]
            if name in required and required[name] != expected:
                raise ValueError('Inconsistent input fingerprints: ' + name)
            required[name] = expected
    missing = [name for name in sorted(required) if not (folder / name).is_file()]
    if missing:
        print(json.dumps(dict(passed=False, missing_files=missing,
            message='Study inputs are not distributed with the source code. Obtain observational records through the data provider and consult DATA.md.'), indent=2))
        return False
    mismatched = []
    for name, expected in sorted(required.items()):
        h = hashlib.sha256()
        with (folder / name).open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != expected:
            mismatched.append(name)
    print(json.dumps(dict(passed=not mismatched, files_checked=len(required), mismatched_files=mismatched), indent=2))
    return not mismatched

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--action', choices=['plan', 'verify', 'check-data', 'preflight', 'train', 'evaluate'], default='plan')
    ap.add_argument('--data', type=Path, default=ROOT / 'data')
    ap.add_argument('--output', type=Path, default=ROOT / 'runs')
    ap.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    ap.add_argument('--leads', nargs='+', type=int, choices=[1, 3, 7], default=[1, 3, 7])
    ap.add_argument('--seeds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--evaluate-test', action='store_true', help='Explicitly score retrospective test dates after validation selection')
    args = ap.parse_args()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    verify()
    if args.action == 'verify':
        print('PACKAGE_HASHES_PASSED')
        return
    if args.action == 'check-data':
        if not check_data(args.data, args.models):
            raise SystemExit(2)
        return
    jobs = list(commands(args))
    if args.action == 'plan':
        print(json.dumps(dict(action='plan_only_no_execution', jobs=jobs, top_level_jobs=len(jobs), note='LSTM variants each fit 15- and 90-day candidates. HBV calibrates once per seed and outputs all three leads.'), indent=2))
        return
    if args.action == 'preflight' and args.output == ROOT / 'runs':
        raise ValueError('Use a separate --output for preflight; never use the formal training directory')
    for i, job in enumerate(jobs):
        print(json.dumps(dict(job=i + 1, total=len(jobs), model=job['model'], lead=job['lead'], seed=job['seed'])), flush=True)
        subprocess.run(job['command'], check=True, cwd=ROOT)
if __name__ == '__main__':
    main()
