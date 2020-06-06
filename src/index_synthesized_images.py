#!/usr/bin/env python3
"""Responsible for indexing synthesized waste images into trashnet for test.

Takes an argument, `in-dataroot` points to a directory of subdirs, each
cataloging a set of synthesized waste images. For example:
```
+ data/
|-+ synthesized-data/
    |-+ metal/
    |-+ glass/
    |-+ cardboard/
    ...
```

Where `synthesized-data` would be provided as `--in-dataroot`.
The `--out-dataroot` specifies a similar directory structure (should have
**exactly** the same subdir names) for where to dump the synthesized images for
classifier training.

Finally, the `--train-list` should point to a file leveraged by the classifier-
under-test to index the images used for training. This file will be augmented
with the data supplied in `--in-dataroot` (e.g. these images will be added to
the index).
"""

import argparse
import logging
logging.root.setLevel(logging.INFO)
import os
import shutil

# Constants #

WASTE_LABELS = {
    'glass': 0,
    'paper': 1,
    'cardboard': 2,
    'plastic': 3,
    'metal': 4,
    'trash': 5,
}

# Public Functions #

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in-dataroot',
        type=str,
        help='The root of the directory of synthesized images.',
        required=True,
    )
    parser.add_argument(
        '--out-dataroot',
        type=str,
        help='The output dataroot to dump the indexed synthesized images in.',
        required=True,
    )
    parser.add_argument(
        '--train-list',
        type=str,
        help='The file path to the list of images to be supplied for training.',
        required=True,
    )
    args = parser.parse_args()
    filepaths = [args.in_dataroot, args.out_dataroot, args.train_list]
    for filepath in filepaths:
        if not os.path.exists(filepath):
            raise ValueError(f'{filepath} does not exist!')
    return args

def main():
    """main."""
    args = parse_args()

    # Index all synthesized images and place into their respective output
    # subdirectories. Note that filenames are one-indexed.
    with open(args.train_list, 'a+') as train_f:
        for root, _, files in os.walk(args.in_dataroot):

            # Only interested in populated image subdirectories
            if not files:
                continue

            # Move synthesized images
            dir_ = os.path.basename(root)
            label = WASTE_LABELS.get(dir_.lower())
            if label is None:
                raise ValueError(
                        f'Unknown classification label for directory: {dir_}.')

            logging.info(f'Indexing synthesized {dir_} images...')
            for i, filename in enumerate(files):
                new_filename = f'{dir_}{i + 1}_synth.png'

                # Copy and index new file
                old_filepath = os.path.join(root, filename)
                new_filepath = os.path.join(
                    args.out_dataroot, dir_, new_filename)
                shutil.copy(old_filepath, new_filepath)

                # Write label and apply *one-indexed* format
                train_f.write(f'{new_filename} {label + 1}\n')

if __name__ == '__main__':
    main()
