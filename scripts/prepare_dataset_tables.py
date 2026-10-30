"""Create isolated v2 labels with inline tables; never replaces v1/raw objects."""
from annotation_targets import MANIFEST_VERSION, resolve_annotation
from prepare_dataset import main

if __name__ == '__main__':
    main(version=MANIFEST_VERSION,target_builder=resolve_annotation)
