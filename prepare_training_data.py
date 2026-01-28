#!/usr/bin/env python3
"""
Unified Training Data Preparation Script

This script prepares training data by:
1. Converting CadQuery .py files to STEP files
2. Rendering STEP files to 224x224 grayscale PNG images
3. Organizing output as (image, code) pairs

Output structure:
    output_dir/
        00000/
            image.png      # 224x224 grayscale PNG
            code.py        # Original CadQuery script
            model.step     # Generated STEP file (optional, can be deleted)
        00001/
            ...
        metadata.json      # Statistics and error tracking

Usage:
    # Full pipeline (requires running in appropriate conda env):
    python prepare_training_data.py --input-dir /path/to/cq_scripts --output-dir /path/to/training_data

    # Step 1 only (CadQuery env): Convert .py to .step
    python prepare_training_data.py --input-dir /path/to/cq_scripts --output-dir /path/to/training_data --step convert

    # Step 2 only (Render env): Render .step to images
    python prepare_training_data.py --output-dir /path/to/training_data --step render

    # With options:
    python prepare_training_data.py --input-dir /path/to/cq_scripts --output-dir /path/to/training_data \
        --max-samples 1000 --num-workers 8 --view-index 0 --keep-step
"""

import os
import sys
import argparse
import shutil
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def execute_cadquery_to_step(py_file, output_step):
    """Execute CadQuery script to generate STEP file"""
    try:
        with open(py_file, 'r') as f:
            content = f.read()

        # Find result variable
        result_var = 'result'
        for var in ['result', 'model', 'cad', 'part', 'shape', 'assembly', 'obj', 'body']:
            if f'{var} =' in content or f'{var}=' in content:
                result_var = var
                break

        # Add export code
        script = content + f"""

import cadquery as cq
try:
    cq.exporters.export({result_var}, '{output_step}')
except Exception as e:
    print(f'Export error: {{e}}')
"""

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

        # Cleanup temp file
        if os.path.exists(temp_py):
            os.remove(temp_py)

        success = result.returncode == 0 and os.path.exists(output_step)
        return success, result.stderr if not success else None

    except subprocess.TimeoutExpired:
        return False, "Timeout (60s)"
    except Exception as e:
        return False, str(e)


def convert_cadquery_to_step(input_dir, output_dir, max_samples=None):
    """
    Step 1: Convert CadQuery .py files to STEP files
    Run this in the cadquery_env conda environment
    """
    print("=" * 70)
    print("Step 1: Converting CadQuery → STEP")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Find .py files
    py_files = sorted(Path(input_dir).rglob('*.py'))
    if max_samples:
        py_files = py_files[:max_samples]

    print(f"\nFound {len(py_files)} CadQuery files\n")

    successes = 0
    failures = 0
    errors = []

    for idx, py_file in enumerate(tqdm(py_files, desc="Converting")):
        # Create sample directory
        sample_dir = os.path.join(output_dir, f"{idx:05d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Copy original code
        shutil.copy2(py_file, os.path.join(sample_dir, 'code.py'))

        # Generate STEP
        step_path = os.path.join(sample_dir, 'model.step')
        success, error = execute_cadquery_to_step(str(py_file), step_path)

        if success:
            successes += 1
        else:
            failures += 1
            errors.append({'id': idx, 'file': py_file.name, 'error': error})
            # Remove failed sample directory
            shutil.rmtree(sample_dir, ignore_errors=True)

    # Save metadata
    metadata = {
        'step': 'convert',
        'total': len(py_files),
        'successful': successes,
        'failed': failures,
        'errors': errors[:100]  # Keep first 100 errors
    }

    metadata_path = os.path.join(output_dir, 'metadata_convert.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Conversion Complete: {successes}/{len(py_files)} successful")
    print(f"{'=' * 70}")

    return successes


def render_single_sample(args):
    """Render a single STEP file to 224x224 grayscale PNG"""
    sample_dir, view_index = args
    step_path = os.path.join(sample_dir, 'model.step')
    image_path = os.path.join(sample_dir, 'image.png')

    if not os.path.exists(step_path):
        return sample_dir, False, "STEP file not found"

    try:
        from GVis.RenderTools.blender_utils import grayscale_224_cad_image
        _, _ = grayscale_224_cad_image(step_path, save_path=image_path, view_index=view_index)
        return sample_dir, True, None
    except Exception as e:
        return sample_dir, False, str(e)


def render_step_to_images(output_dir, num_workers=8, view_index=0):
    """
    Step 2: Render STEP files to 224x224 grayscale PNG images
    Run this in the cad_render_env conda environment
    """
    print("=" * 70)
    print("Step 2: Rendering STEP → 224x224 Grayscale PNG")
    print("=" * 70)
    print(f"Data directory: {output_dir}")
    print(f"Workers: {num_workers}")
    print(f"View index: {view_index}")
    print("=" * 70)

    # Find all sample directories with STEP files
    sample_dirs = []
    for entry in sorted(os.listdir(output_dir)):
        sample_path = os.path.join(output_dir, entry)
        if os.path.isdir(sample_path) and os.path.exists(os.path.join(sample_path, 'model.step')):
            sample_dirs.append(sample_path)

    print(f"\nFound {len(sample_dirs)} samples to render\n")

    if len(sample_dirs) == 0:
        print("No samples found. Run conversion step first.")
        return 0

    successes = 0
    failures = 0
    errors = []

    # Prepare arguments
    render_args = [(sd, view_index) for sd in sample_dirs]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(render_single_sample, arg): arg[0] for arg in render_args}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering"):
            sample_dir = futures[future]
            try:
                _, success, error = future.result()
                if success:
                    successes += 1
                else:
                    failures += 1
                    errors.append({'dir': os.path.basename(sample_dir), 'error': error})
            except Exception as e:
                failures += 1
                errors.append({'dir': os.path.basename(sample_dir), 'error': str(e)})

    # Save metadata
    metadata = {
        'step': 'render',
        'total': len(sample_dirs),
        'successful': successes,
        'failed': failures,
        'view_index': view_index,
        'resolution': '224x224',
        'format': 'grayscale PNG',
        'errors': errors[:100]
    }

    metadata_path = os.path.join(output_dir, 'metadata_render.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Rendering Complete: {successes}/{len(sample_dirs)} successful")
    print(f"{'=' * 70}")

    return successes


def cleanup_step_files(output_dir):
    """Remove STEP files to save space (optional)"""
    print("Cleaning up STEP files...")
    removed = 0
    for entry in os.listdir(output_dir):
        sample_path = os.path.join(output_dir, entry)
        if os.path.isdir(sample_path):
            step_path = os.path.join(sample_path, 'model.step')
            if os.path.exists(step_path):
                os.remove(step_path)
                removed += 1
    print(f"Removed {removed} STEP files")


def validate_output(output_dir):
    """Validate that all samples have both image.png and code.py"""
    print("\nValidating output...")
    valid = 0
    invalid = []

    for entry in sorted(os.listdir(output_dir)):
        sample_path = os.path.join(output_dir, entry)
        if os.path.isdir(sample_path):
            has_image = os.path.exists(os.path.join(sample_path, 'image.png'))
            has_code = os.path.exists(os.path.join(sample_path, 'code.py'))

            if has_image and has_code:
                valid += 1
            else:
                invalid.append({
                    'dir': entry,
                    'has_image': has_image,
                    'has_code': has_code
                })

    print(f"Valid samples: {valid}")
    print(f"Invalid samples: {len(invalid)}")

    # Save final metadata
    metadata = {
        'valid_samples': valid,
        'invalid_samples': len(invalid),
        'output_format': {
            'image': '224x224 grayscale PNG',
            'code': 'CadQuery Python script'
        }
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return valid


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data: CadQuery scripts → rendered images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (run Step 1 in cadquery_env, Step 2 in cad_render_env):

  # In cadquery_env:
  python prepare_training_data.py --input-dir /orcd/data/faez/001/erasyla/deepcad_filtered \\
      --output-dir ./training_data --step convert --max-samples 1000

  # In cad_render_env:
  python prepare_training_data.py --output-dir ./training_data --step render --num-workers 8

  # Or run both steps together (make sure you're in an env with both cadquery and rendering deps):
  python prepare_training_data.py --input-dir /path/to/scripts --output-dir ./training_data --step all
"""
    )

    parser.add_argument('--input-dir', type=str,
                        help='Directory containing CadQuery .py files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for training data')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of parallel workers for rendering')
    parser.add_argument('--view-index', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Isometric view angle (0-3)')
    parser.add_argument('--step', type=str, default='all',
                        choices=['convert', 'render', 'all'],
                        help='Which step to run: convert (CQ→STEP), render (STEP→PNG), or all')
    parser.add_argument('--keep-step', action='store_true',
                        help='Keep STEP files after rendering (default: delete to save space)')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing output, do not process')

    args = parser.parse_args()

    if args.validate_only:
        validate_output(args.output_dir)
        return

    if args.step in ['convert', 'all']:
        if not args.input_dir:
            print("ERROR: --input-dir required for conversion step")
            sys.exit(1)
        convert_cadquery_to_step(args.input_dir, args.output_dir, args.max_samples)

    if args.step in ['render', 'all']:
        render_step_to_images(args.output_dir, args.num_workers, args.view_index)

        if not args.keep_step:
            cleanup_step_files(args.output_dir)

    # Final validation
    validate_output(args.output_dir)


if __name__ == "__main__":
    main()
