#!/usr/bin/env python3
"""
Convert CadQuery .py files to STEP files
Filters by token count - only processes files with tokens > min_tokens
"""

import os
import sys
import argparse
import shutil
import json
import subprocess
import tokenize
import io
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def count_tokens(code_string):
    """Count Python tokens in code string"""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code_string).readline))
        # Exclude ENCODING, NEWLINE, NL, INDENT, DEDENT, ENDMARKER
        meaningful = [t for t in tokens if t.type not in (
            tokenize.ENCODING, tokenize.NEWLINE, tokenize.NL,
            tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER
        )]
        return len(meaningful)
    except:
        # Fallback to whitespace split
        return len(code_string.split())


def process_single_file(args):
    """Process a single CadQuery file - designed for parallel execution"""
    idx, py_file, output_dir = args

    sample_dir = os.path.join(output_dir, f"{idx:05d}")
    step_path = os.path.join(sample_dir, 'model.step')

    # Skip if already processed successfully
    if os.path.exists(step_path) and os.path.getsize(step_path) > 0:
        return idx, py_file.name, True, None, True  # Last True = skipped

    os.makedirs(sample_dir, exist_ok=True)

    # Copy code
    shutil.copy2(py_file, os.path.join(sample_dir, 'code.py'))

    # Execute conversion
    try:
        with open(py_file, 'r') as f:
            content = f.read()

        # Find result variable
        result_var = 'result'
        for var in ['result', 'solid', 'model', 'cad', 'part', 'shape', 'assembly', 'obj', 'body']:
            if f'{var} =' in content or f'{var}=' in content:
                result_var = var
                break

        # Add export
        script = content + f"\n\nimport cadquery as cq\ntry:\n    cq.exporters.export({result_var}, '{step_path}')\nexcept Exception as e:\n    print(f'Export error: {{e}}')\n"

        # Write temp script
        temp_py = step_path.replace('.step', '_exec.py')
        with open(temp_py, 'w') as f:
            f.write(script)

        # Execute
        result = subprocess.run(
            ['python', temp_py],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Cleanup
        if os.path.exists(temp_py):
            os.remove(temp_py)

        success = result.returncode == 0 and os.path.exists(step_path)
        error = result.stderr if not success else None
        return idx, py_file.name, success, error, False

    except subprocess.TimeoutExpired:
        return idx, py_file.name, False, "Timeout (60s)", False
    except Exception as e:
        return idx, py_file.name, False, str(e), False


def main():
    parser = argparse.ArgumentParser(description='Convert CadQuery to STEP files (with token filtering)')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already processed files')
    parser.add_argument('--min-tokens', type=int, default=0,
                        help='Only process files with more than this many tokens (default: 0 = no filter)')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Only process files with fewer than this many tokens (default: None = no limit)')
    args = parser.parse_args()

    print("="*70)
    print("Converting CadQuery → STEP (with token filtering)")
    print("="*70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"Resume: {args.resume}")
    print(f"Min tokens: {args.min_tokens}")
    print(f"Max tokens: {args.max_tokens}")
    print("="*70)

    os.makedirs(args.output_dir, exist_ok=True)

    # Find .py files
    print("\nScanning for .py files...")
    py_files = sorted(Path(args.input_dir).rglob('*.py'))
    print(f"Found {len(py_files)} total files")

    # Filter by token count
    if args.min_tokens > 0 or args.max_tokens is not None:
        print(f"\nFiltering by token count...")
        filtered_files = []
        for py_file in tqdm(py_files, desc="Counting tokens"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                token_count = count_tokens(content)

                # Check min
                if args.min_tokens > 0 and token_count <= args.min_tokens:
                    continue
                # Check max
                if args.max_tokens is not None and token_count >= args.max_tokens:
                    continue

                filtered_files.append(py_file)
            except Exception as e:
                continue

        py_files = filtered_files
        print(f"After filtering: {len(py_files)} files")

    if args.max_samples:
        py_files = py_files[:args.max_samples]

    print(f"\nProcessing {len(py_files)} files\n")

    # Prepare work items
    work_items = [(idx, py_file, args.output_dir) for idx, py_file in enumerate(py_files)]

    successes = 0
    failures = 0
    skipped = 0
    errors = []

    if args.num_workers == 1:
        # Sequential processing
        for item in tqdm(work_items, desc="Converting"):
            idx, filename, success, error, was_skipped = process_single_file(item)
            if was_skipped:
                skipped += 1
                successes += 1
            elif success:
                successes += 1
            else:
                failures += 1
                errors.append({'id': idx, 'file': filename, 'error': error})
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_file, item): item for item in work_items}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
                try:
                    idx, filename, success, error, was_skipped = future.result()
                    if was_skipped:
                        skipped += 1
                        successes += 1
                    elif success:
                        successes += 1
                    else:
                        failures += 1
                        errors.append({'id': idx, 'file': filename, 'error': error})
                except Exception as e:
                    failures += 1
                    errors.append({'id': -1, 'file': 'unknown', 'error': str(e)})

    # Save metadata
    metadata = {
        'total': len(py_files),
        'successful': successes,
        'skipped': skipped,
        'failed': failures,
        'min_tokens_filter': args.min_tokens,
        'max_tokens_filter': args.max_tokens,
        'errors': errors[:50]
    }

    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Total: {len(py_files)} | Success: {successes} | Skipped: {skipped} | Failed: {failures}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
