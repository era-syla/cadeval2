#!/usr/bin/env python3
"""
Step 1: Convert CadQuery .py files to STEP files
Can be run with regular Python (no Blender needed)
"""

import os
import sys
import argparse
import shutil
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

def execute_cadquery_to_step(py_file, output_step):
    """Execute CadQuery script to generate STEP file"""
    try:
        with open(py_file, 'r') as f:
            content = f.read()

        # Find result variable
        result_var = 'result'
        for var in ['result', 'model', 'cad', 'part', 'shape', 'assembly']:
            if f'{var} =' in content or f'{var}=' in content:
                result_var = var
                break

        # Add export
        script = content + f"\n\nimport cadquery as cq\ntry:\n    cq.exporters.export({result_var}, '{output_step}')\nexcept Exception as e:\n    print(f'Export error: {{e}}')\n"

        # Write temp script
        temp_py = output_step.replace('.step', '_exec.py')
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

        success = result.returncode == 0 and os.path.exists(output_step)
        return success, result.stderr if not success else None

    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Convert CadQuery to STEP files')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()

    print("="*70)
    print("Converting CadQuery → STEP")
    print("="*70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("="*70)

    os.makedirs(args.output_dir, exist_ok=True)

    # Find .py files
    py_files = sorted(Path(args.input_dir).rglob('*.py'))
    if args.max_samples:
        py_files = py_files[:args.max_samples]

    print(f"\nFound {len(py_files)} files\n")

    successes = 0
    failures = 0
    errors = []

    for idx, py_file in enumerate(tqdm(py_files, desc="Converting")):
        # Create sample dir
        sample_dir = os.path.join(args.output_dir, f"{idx:05d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Copy code
        shutil.copy2(py_file, os.path.join(sample_dir, 'code.py'))

        # Generate STEP
        step_path = os.path.join(sample_dir, 'model.step')
        success, error = execute_cadquery_to_step(str(py_file), step_path)

        if success:
            successes += 1
        else:
            failures += 1
            errors.append({'id': idx, 'file': py_file.name, 'error': error})

    # Save metadata
    metadata = {
        'total': len(py_files),
        'successful': successes,
        'failed': failures,
        'errors': errors[:50]
    }

    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Total: {len(py_files)} | Success: {successes} | Failed: {failures}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
