import os
import json
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO

from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
matplotlib.use('Agg')  # Use non-GUI backend
import glob


app = Flask(__name__)
CORS(app)

# Get sorted file paths and file names
file_paths1 = glob.glob('./dataset/*')  # Fixed the glob pattern
file_paths1.sort()

file_names1 = [os.path.basename(path) for path in file_paths1]  # Extract file names from paths
file_names1.sort()

# Initialize lists for different MRI modalities and segmentation labels
t1c, t1n, t2f, t2w, label = [], [], [], [], []

# Use the total number of files instead of a fixed 330
num_files = len(file_paths1)

# Populate the lists with file paths
for i in range(num_files):
    t1c.append(os.path.join(file_paths1[i], file_names1[i] + '-t1c.nii.gz'))
    t1n.append(os.path.join(file_paths1[i], file_names1[i] + '-t1n.nii.gz'))
    t2f.append(os.path.join(file_paths1[i], file_names1[i] + '-t2f.nii.gz'))
    t2w.append(os.path.join(file_paths1[i], file_names1[i] + '-t2w.nii.gz'))
    label.append(os.path.join(file_paths1[i], file_names1[i] + '-seg.nii.gz'))

# Store in a dictionary with combined image modalities and separate label
file_list = []
for i in range(num_files):
    file_list.append({
        "image": [t1c[i], t1n[i], t2f[i], t2w[i]],  # Combine modalities into one "image" field
        "label": label[i]
    })

file_json = {
    "training": file_list
}

# Save to JSON file
file_path = 'python/dataset.json'
with open(file_path, 'w') as json_file:
    json.dump(file_json, json_file, indent=4)

# Define path to dataset and model
root_dir = 'python/'
dataset_path = "python/dataset.json"

# Class for label conversion
class ConvertLabels(MapTransform):
    """
    Convert labels to multi channels based on BRATS 2023 classes:
    label 1 is Necrotic Tumor Core (NCR)
    label 2 is Edema (ED)
    label 3 is Enhancing Tumor (ET)
    label 0 is everything else (non-tumor)
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # Tumor Core (TC) = NCR + Enhancing Tumor (ET)
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            # Whole Tumor (WT) = NCR + Edema + Enhancing Tumor
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 3))
            # Enhancing Tumor (ET) = Enhancing Tumor (label 3)
            result.append(d[key] == 3)
            d[key] = torch.stack(result, axis=0).float()
        return d

# Define transforms
val_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys="image"),
    EnsureTyped(keys=["image", "label"]),
    ConvertLabels(keys="label"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "nearest"),
    ),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
])

# Load dataset
with open(dataset_path) as f:
    datalist = json.load(f)["training"]

# Initialize model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=4,
    out_channels=3,
    feature_size=24,
    use_checkpoint=True,
)

# Load model weights
model.load_state_dict(torch.load(os.path.join(root_dir, "best_distilled_model.pth"), map_location=device))
model = model.to(device)
model.eval()

# Post-processing
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# Define color scheme
class_colors = [
    [0.7, 0.7, 0.7, 1],        # Background (neutral gray)
    [0.85, 0.37, 0.35, 0.7],   # Class 1 (rust red)
    [0.46, 0.78, 0.56, 0.7],   # Class 2 (sage green)
    [0.31, 0.51, 0.9, 0.7]     # Class 3 (medium blue)
]
custom_cmap = ListedColormap(class_colors)
class_names = ["Tumor Core", "Whole Tumor", "Enhancing"]
class_cmaps = ['RdPu', 'BuGn', 'PuBu']

def get_sample_by_id(idx):
    """Get a sample from the dataset by ID"""
    from monai.data import Dataset
    val_ds = Dataset(data=[datalist[idx]], transform=val_transform)
    return val_ds[0]

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

@app.route('/process_sample', methods=['POST'])
def process_sample():
    data = request.json
    idx = int(data.get('id', 0))
    
    # Validate index
    if idx < 0 or idx >= len(datalist):
        return jsonify({'error': f'Invalid ID. Must be between 0 and {len(datalist)-1}'}), 400
    
    try:
        # Get sample
        sample = get_sample_by_id(idx)
        
        with torch.no_grad():
            # Process input
            val_input = sample["image"].unsqueeze(0).to(device)
            val_label = sample["label"]
            
            # Run inference
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            val_output = sliding_window_inference(val_input, roi_size, sw_batch_size, model)
            val_output = post_trans(val_output[0])
            
            # Move tensors to CPU and convert to numpy
            val_input_np = val_input[0, 0].cpu().numpy()
            val_label_np = val_label.cpu().numpy()
            val_output_np = val_output.cpu().numpy()
            
            # Normalize image for visualization
            val_input_np = (val_input_np - val_input_np.min()) / (val_input_np.max() - val_input_np.min())
            val_input_np = (val_input_np * 255).astype(np.uint8)
            
            # Determine slice to use
            total_slices = val_input_np.shape[-1]
            middle_slice = total_slices // 2
            slice_idx = 77 if total_slices > 77 else middle_slice
            
            # Generate images
            images = {}
            
            # 1. Overview comparison
            fig1 = plt.figure(figsize=(18, 6), facecolor='white')
            
            # Create combined segmentation maps
            num_classes = val_label_np.shape[0]
            gt_combined = np.zeros((val_label_np.shape[1], val_label_np.shape[2], 4))
            pred_combined = np.zeros((val_output_np.shape[1], val_output_np.shape[2], 4))
            
            for c in range(num_classes):
                # Ground truth
                mask = val_label_np[c, :, :, slice_idx]
                for i in range(4):
                    gt_combined[:, :, i] = np.where(mask > 0, class_colors[c+1][i], gt_combined[:, :, i])
                
                # Prediction
                mask = val_output_np[c, :, :, slice_idx]
                for i in range(4):
                    pred_combined[:, :, i] = np.where(mask > 0, class_colors[c+1][i], pred_combined[:, :, i])
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.title("Original MRI", fontsize=14, fontweight='bold')
            plt.imshow(val_input_np[:, :, slice_idx], cmap="gray")
            plt.axis('off')
            
            # Ground truth
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth", fontsize=14, fontweight='bold')
            plt.imshow(val_input_np[:, :, slice_idx], cmap="gray")
            plt.imshow(gt_combined)
            plt.axis('off')
            
            # Prediction
            plt.subplot(1, 3, 3)
            plt.title("Predicted Segmentation", fontsize=14, fontweight='bold')
            plt.imshow(val_input_np[:, :, slice_idx], cmap="gray")
            plt.imshow(pred_combined)
            plt.axis('off')
            
            # Legend
            legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=class_colors[i+1][:3], alpha=0.7) for i in range(num_classes)]
            plt.figlegend(legend_patches, class_names, loc='lower center', ncol=num_classes, 
                        bbox_to_anchor=(0.5, -0.05), fontsize=12, frameon=True, edgecolor='black')
            
            plt.tight_layout(pad=1.5)
            plt.subplots_adjust(bottom=0.15)
            images['overview'] = fig_to_base64(fig1)
            
            # 2. Per-class comparisons
            fig2 = plt.figure(figsize=(15, 5 * num_classes), facecolor='white')
            
            for c in range(num_classes):
                # Ground Truth for this class
                plt.subplot(num_classes, 2, 2*c+1)
                plt.title(f"Ground Truth - {class_names[c]}", fontsize=12, fontweight='bold')
                plt.imshow(val_input_np[:, :, slice_idx], cmap="gray")
                plt.imshow(val_label_np[c, :, :, slice_idx], cmap=class_cmaps[c], alpha=0.7, vmin=0, vmax=1)
                plt.axis('off')
                
                # Prediction for this class
                plt.subplot(num_classes, 2, 2*c+2)
                plt.title(f"Prediction - {class_names[c]}", fontsize=12, fontweight='bold')
                plt.imshow(val_input_np[:, :, slice_idx], cmap="gray")
                plt.imshow(val_output_np[c, :, :, slice_idx], cmap=class_cmaps[c], alpha=0.7, vmin=0, vmax=1)
                plt.axis('off')
                
                # Colorbar
                plt.colorbar(shrink=0.8, ax=plt.gca())
            
            plt.tight_layout(pad=2.0)
            images['per_class'] = fig_to_base64(fig2)
            
            # 3. Individual modalities
            fig3 = plt.figure(figsize=(18, 6), facecolor='white')
            for i in range(4):
                plt.subplot(1, 4, i + 1)
                modality_names = ["T1C", "T1N", "T2F", "T2W"]
                plt.title(f"{modality_names[i]}", fontsize=12, fontweight='bold')
                plt.imshow(val_input[0, i, :, :, slice_idx].cpu().numpy(), cmap="gray")
                plt.axis('off')
            
            plt.tight_layout()
            images['modalities'] = fig_to_base64(fig3)
            
            return jsonify({
                'success': True,
                'images': images,
                'slice_info': {
                    'current': int(slice_idx),
                    'total': int(total_slices)
                },
                'sample_id': idx
            })
            
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/dataset_info', methods=['GET'])
def dataset_info():
    """Return information about the dataset"""
    return jsonify({
        'total_samples': len(datalist),
        'sample_filenames': [os.path.basename(item['image'][0]).split('-')[0] for item in datalist]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)