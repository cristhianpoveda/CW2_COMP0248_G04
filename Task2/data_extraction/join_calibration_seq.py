from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def create_calibration_symlinks(folder1_path: Path, folder2_path: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    def link_images_from_folder(source_folder: Path, prefix: str):
        if not source_folder.exists():
            print(f"Warning: Source dir {source_folder} does not exist.")
            return
        
        images = sorted(list(source_folder.glob("*.png")) + list(source_folder.glob("*.jpg")))
        
        step_size = 15 
        sampled_images = images[::step_size]
        
        count = 0
        for img_path in sampled_images:
            prefix_plus_orig_name = f"{prefix}_{img_path.name}"
            abs_symlink_path = output_path / prefix_plus_orig_name

            if not abs_symlink_path.exists():
                abs_symlink_path.symlink_to(img_path.resolve())
                count += 1
                
        print(f"Linked {count} images from {source_folder.name}")

    link_images_from_folder(folder1_path, "seq1")
    link_images_from_folder(folder2_path, "seq2")
    print(f"Done! Combined calibration images are available in: {output_path.resolve()}")

def main() -> None:
    SRC_DIR_1 = PROJECT_ROOT / "data/GROUP_04_calibration/camera_color_image_raw"
    SRC_DIR_2 = PROJECT_ROOT / "data/GROUP_04_calibration_02/camera_color_image_raw"
    OUTPUT_DIR = PROJECT_ROOT / "data/combined_calibration_images"
    
    create_calibration_symlinks(SRC_DIR_1, SRC_DIR_2, OUTPUT_DIR)

if __name__ == "__main__":
    main()