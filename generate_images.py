from GVis.RenderTools.blender_utils import gray_cad_image, multicolor_cad_image, grayscale_224_cad_image
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pickle

def run_metric_parallel(metric_function, data_list, save_path=None, num_workers=8):
    """
    Run a metric function in parallel across a data list using ProcessPoolExecutor.

    Args:
        metric_function: The function to run on each data item. Should accept a single
                        data item and return a tuple of (item_name, result).
        data_list: List of data items to process
        num_workers: Number of parallel workers (default: 8)

    Returns:
        list: List of tuples containing (item_name, result) for each processed item
    """
    all_results = []
    num_failures = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        # Submit all tasks
        future_to_item = {executor.submit(metric_function, item): item for item in data_list}

        completed = 0
        for future in tqdm(as_completed(future_to_item), total=len(data_list),
                          desc=f"Processing {metric_function.__name__}"):
            try:
                item_name, result = future.result()
                all_results.append((str(item_name), result))
            except Exception:
                num_failures += 1
            completed += 1

    # Report number of successes vs failures
    print(f"Completed {completed} tasks with {num_failures} failures.")
    
    # Report average result
    try:
        print(f"Average for {metric_function.__name__}: {sum(r for _, r in all_results if r >= 0) / max(1, sum(1 for _, r in all_results if r >= 0))}")
    except:
        print(f"Could not compute average for {metric_function.__name__}")

    # Report failures
    if num_failures > 0:
        print(f"Number of failures: {num_failures}")

    if save_path is not None:
        # Check that results folder exists, if not, create it
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(all_results, f)
    return all_results

def get_step_files(data_path):
    """
    Get all STEP files from the given folder and its subdirectories.

    Args:
        data_path: Path to the folder containing data

    Returns:
        list: List of absolute file paths to STEP files

    Raises:
        ValueError: If path doesn't exist, is not a directory, or no STEP files found
    """
    if not os.path.exists(data_path):
        raise ValueError(f"Path does not exist: {data_path}")

    if not os.path.isdir(data_path):
        raise ValueError(f"Path is not a directory: {data_path}")

    # Collect all STEP files
    step_files = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.upper().endswith('.STEP'):
                step_files.append(os.path.abspath(os.path.join(root, file)))

    if len(step_files) == 0:
        raise ValueError("No STEP files found in the directory")

    return step_files

def main():
    parser = argparse.ArgumentParser(description='Evaluate dataset metrics')

    parser.add_argument(
        '--gray_images',
        action='store_true',
        default=False,
        help='Generate gray CAD images'
    )
    
    parser.add_argument(
        '--multicolor_images',
        action='store_true',
        default=False,
        help='Generate multicolor CAD images'
    )

    parser.add_argument(
        '--grayscale_224',
        action='store_true',
        default=False,
        help='Generate 224x224 grayscale PNG images (for training data)'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required = True,
        help='Path to the folder containing data'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of worker processes to use'
    )

    args = parser.parse_args()

    # Your evaluation logic here
    print(f"Will generate gray CAD images: {args.gray_images}")
    print(f"Data path: {args.data_path}")
    print(f"Number of workers: {args.num_workers}")
    
    # Get the data files
    data_list = get_step_files(args.data_path)
    
    if args.gray_images:
        run_metric_parallel(
            gray_cad_image,
            data_list,
            num_workers=args.num_workers
        )
        
    if args.multicolor_images:
        run_metric_parallel(
            multicolor_cad_image,
            data_list,
            num_workers=args.num_workers
        )

    if args.grayscale_224:
        run_metric_parallel(
            grayscale_224_cad_image,
            data_list,
            num_workers=args.num_workers
        )

if __name__ == "__main__":
    main()