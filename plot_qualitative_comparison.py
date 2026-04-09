import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def create_qualitative_figure(real_img_path, forward_img_path, inverse_img_path, output_path):
    
    img_real = cv2.cvtColor(cv2.imread(str(real_img_path)), cv2.COLOR_BGR2RGB)
    img_fwd = cv2.cvtColor(cv2.imread(str(forward_img_path)), cv2.COLOR_BGR2RGB)
    img_inv = cv2.cvtColor(cv2.imread(str(inverse_img_path)), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    # Real Image
    axes[0][0].imshow(img_real)
    axes[0][0].set_title("Ground Truth", fontsize=12)
    axes[0][0].axis('off')

    # Forward Warping
    axes[0][1].imshow(img_fwd)
    axes[0][1].set_title("Forward Warping", fontsize=12)
    axes[0][1].axis('off')

    # Inverse Warping
    axes[1][0].imshow(img_inv)
    axes[1][0].set_title("Inverse Warping", fontsize=12)
    axes[1][0].axis('off')

    fig.delaxes(axes[1][1])

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved successfully to {output_path}")

if __name__ == "__main__":
    
    real = "data/Sequence_A/camera_color_image_raw/12_1772626903083449320.png" 
    forward = "results/Sequence_A/vanilla_pnp_nosplat_20260408_200953/synthetics_imgs/synth_12_1772626903083449320.png"
    inverse = "results/Sequence_A/vanilla_pnp_splat_20260408_201123/synthetics_imgs/synth_12_1772626903083449320.png"
    
    out_file = "data/Report_figures/qualitative_comparison_v.png"
    
    create_qualitative_figure(real, forward, inverse, out_file)