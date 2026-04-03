import argparse
import json
import subprocess
import time
from pathlib import Path

from context_detector import ContextConditionedDetector


def gpu_mem_mb():
    try:
        out = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'
        ], text=True).strip().splitlines()
        vals = [int(x.strip()) for x in out if x.strip()]
        return max(vals) if vals else 0
    except Exception:
        return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-list', default='outputs/batch10/sample_list.txt')
    parser.add_argument('--output-dir', default='outputs/batch10')
    parser.add_argument('--context', default='data/toy-91/context.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--box-threshold', type=float, default=0.20)
    parser.add_argument('--text-threshold', type=float, default=0.15)
    parser.add_argument('--match-threshold', type=float, default=0.22)
    parser.add_argument('--nms-threshold', type=float, default=0.45)
    parser.add_argument('--skip-vis', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    sample_list = [
        line.strip()
        for line in Path(args.sample_list).read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if not sample_list:
        raise ValueError(f'No samples found in {args.sample_list}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = ContextConditionedDetector(device=args.device)

    # Warm up model kernels and caches without writing outputs.
    _ = detector.detect_from_files(
        context_json_path=args.context,
        query_image_path=sample_list[0],
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        match_threshold=args.match_threshold,
        nms_threshold=args.nms_threshold,
        vis_path=None,
    )

    times = []
    max_mem = gpu_mem_mb()
    summary = []
    total = len(sample_list)

    for idx, q in enumerate(sample_list, 1):
        stem = Path(q).stem
        vis_path = None if args.skip_vis else str(output_dir / f'{stem}.vis.jpg')

        t0 = time.perf_counter()
        pred = detector.detect_from_files(
            context_json_path=args.context,
            query_image_path=q,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            match_threshold=args.match_threshold,
            nms_threshold=args.nms_threshold,
            vis_path=vis_path,
        )
        dt = time.perf_counter() - t0
        det_count = len(pred.get('detections', []))

        times.append(dt)
        max_mem = max(max_mem, gpu_mem_mb())

        out_json = output_dir / f'{stem}.pred.json'
        out_json.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding='utf-8')
        summary.append({
            'query': q,
            'seconds': dt,
            'detections': det_count,
        })
        print(f'[{idx}/{total}] {q} sec={dt:.4f} det={det_count}')

    result = {
        'count': len(times),
        'avg_seconds': sum(times) / len(times) if times else 0.0,
        'max_seconds': max(times) if times else 0.0,
        'min_seconds': min(times) if times else 0.0,
        'max_gpu_mem_mb': max_mem,
        'items': summary,
    }
    (output_dir / 'benchmark_summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('SUMMARY')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
