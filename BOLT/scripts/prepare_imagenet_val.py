#!/usr/bin/env python3
"""Extract the official ILSVRC2012 validation archive and make a label manifest.

The validation tar contains images only.  Labels come from the official devkit
ground-truth file and are converted from the devkit's 1-based class ids to the
zero-based ids used by ViT-B/16 classifiers.  No labels or images are
synthetically generated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


EXPECTED_IMAGES = 50_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_path(root: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive member path: {member_name}")
    target = (root / Path(*relative.parts)).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"archive member escapes destination: {member_name}")
    return target


def extract_tar_safe(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as handle:
        members = handle.getmembers()
        file_total = sum(member.isfile() for member in members)
        file_count = 0
        print(
            f"IMAGENET_EXTRACT_START archive={archive} files={file_total} "
            f"destination={destination}",
            flush=True,
        )
        for member in members:
            target = safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported non-file archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            with target.open("wb") as output:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            file_count += 1
            if file_count % 5000 == 0 or file_count == file_total:
                print(
                    f"IMAGENET_EXTRACT_PROGRESS archive={archive.name} "
                    f"files={file_count}/{file_total}",
                    flush=True,
                )


def find_ground_truth(root: Path) -> Path:
    candidates = sorted(root.rglob("ILSVRC2012_validation_ground_truth.txt"))
    if len(candidates) != 1:
        raise ValueError(
            "official devkit must contain exactly one "
            f"ILSVRC2012_validation_ground_truth.txt, found {len(candidates)}")
    return candidates[0]


def find_meta_mat(root: Path) -> Path:
    candidates = sorted(root.rglob("meta.mat"))
    if len(candidates) != 1:
        raise ValueError(
            f"official devkit must contain exactly one meta.mat, found {len(candidates)}")
    return candidates[0]


def parse_official_labels(devkit_root: Path) -> tuple[list[int], list[dict[str, object]]]:
    """Map devkit synset ids to the standard zero-based ImageNet class order.

    Values in ILSVRC2012_validation_ground_truth.txt are ILSVRC2012 synset
    identifiers, not classifier indices.  The standard PyTorch/Hugging Face
    ImageNet-1k class order is the lexicographic order of the 1,000 leaf WNIDs.
    """
    try:
        import scipy.io
    except ImportError as error:
        raise RuntimeError(
            "parsing the official ImageNet devkit requires scipy; install "
            "requirements-vit-b16.txt in .venv-vit-b16") from error

    meta_path = find_meta_mat(devkit_root)
    synsets = scipy.io.loadmat(meta_path, squeeze_me=True)["synsets"]
    child_counts = list(zip(*synsets))[4]
    leaves = [synsets[index] for index, count in enumerate(child_counts)
              if int(count) == 0]
    if len(leaves) != 1000:
        raise ValueError(f"official devkit leaf synset count mismatch: {len(leaves)}")

    ilsvrc_ids, wnids, class_names = list(zip(*leaves))[:3]
    id_to_wnid = {int(class_id): str(wnid)
                  for class_id, wnid in zip(ilsvrc_ids, wnids)}
    wnid_to_names = {str(wnid): str(names)
                     for wnid, names in zip(wnids, class_names)}
    ordered_wnids = sorted(wnid_to_names)
    wnid_to_label = {wnid: label for label, wnid in enumerate(ordered_wnids)}

    ground_truth = find_ground_truth(devkit_root)
    synset_ids = [int(line.strip()) for line in
                  ground_truth.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    if len(synset_ids) != EXPECTED_IMAGES:
        raise ValueError(
            f"official validation label count mismatch: {len(synset_ids)} "
            f"!= {EXPECTED_IMAGES}")
    unknown = sorted(set(synset_ids) - set(id_to_wnid))
    if unknown:
        raise ValueError(f"ground truth references unknown leaf synset ids: {unknown[:8]}")
    labels = [wnid_to_label[id_to_wnid[class_id]] for class_id in synset_ids]
    class_index = [
        {
            "label": label,
            "wnid": wnid,
            "class_names": wnid_to_names[wnid],
        }
        for label, wnid in enumerate(ordered_wnids)
    ]
    return labels, class_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-tar", type=Path, required=True)
    parser.add_argument("--devkit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.val_tar.is_file() or not args.devkit.is_file():
        raise FileNotFoundError("official validation tar and devkit are required")
    image_root = args.output_root / "images"
    devkit_root = args.output_root / "devkit"
    manifest = args.output_root / "validation_manifest.csv"
    metadata_path = args.output_root / "metadata.json"
    class_index_path = args.output_root / "class_index.json"

    if manifest.exists() and not args.force:
        raise FileExistsError(
            f"{manifest} exists; use --force only to rebuild this dataset artifact")
    if image_root.exists() and any(image_root.iterdir()) and not args.force:
        raise FileExistsError(
            f"{image_root} is non-empty; use --force only to rebuild this dataset artifact")

    if args.force:
        image_root.mkdir(parents=True, exist_ok=True)
        devkit_root.mkdir(parents=True, exist_ok=True)

    if not any(image_root.glob("*.JPEG")):
        extract_tar_safe(args.val_tar, image_root)
    if (not devkit_root.exists() or
            not any(devkit_root.rglob("ILSVRC2012_validation_ground_truth.txt"))):
        extract_tar_safe(args.devkit, devkit_root)

    images = sorted(image_root.glob("ILSVRC2012_val_*.JPEG"))
    if len(images) != EXPECTED_IMAGES:
        raise ValueError(
            f"official validation image count mismatch: {len(images)} != {EXPECTED_IMAGES}")
    ground_truth = find_ground_truth(devkit_root)
    labels, class_index = parse_official_labels(devkit_root)
    if any(label < 0 or label >= 1000 for label in labels):
        raise ValueError("official validation labels are outside ImageNet-1k")

    args.output_root.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for image, label in zip(images, labels):
            writer.writerow((str(image.relative_to(args.output_root)), label))

    class_index_path.write_text(json.dumps(class_index, indent=2) + "\n",
                                encoding="utf-8")

    metadata = {
        "format": "ILSVRC2012_VALIDATION_MANIFEST_V1",
        "source": {
            "images_url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
            "devkit_url": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
            "images_tar_sha256": sha256(args.val_tar),
            "devkit_sha256": sha256(args.devkit),
        },
        "images": EXPECTED_IMAGES,
        "classes": 1000,
        "label_mapping": (
            "official devkit ILSVRC2012 synset id -> leaf WNID -> "
            "lexicographically sorted WNID index"
        ),
        "image_root": str(image_root),
        "manifest": str(manifest),
        "class_index": str(class_index_path),
        "ground_truth": str(ground_truth),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n",
                             encoding="utf-8")
    print(f"IMAGENET_VAL_PREPARED=PASS images={len(images)} labels={len(labels)}")
    print(f"IMAGENET_VAL_MANIFEST={manifest}")
    print(f"IMAGENET_VAL_CLASS_INDEX={class_index_path}")
    print(f"IMAGENET_VAL_METADATA={metadata_path}")


if __name__ == "__main__":
    main()
