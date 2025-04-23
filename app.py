import os
import json
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
from io import BytesIO
from werkzeug.utils import secure_filename
import nibabel as nib
from PIL import Image, ImageOps
from skimage import measure
import trimesh
import glob

from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch, Dataset
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

# Use non-GUI backend
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

# Define absolute paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
GIF_FOLDER = os.path.join(STATIC_FOLDER, 'gifs')
MODEL_FOLDER = os.path.join(STATIC_FOLDER, 'models')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GIF_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GIF_FOLDER'] = GIF_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# ===== Dataset preparation =====
# Get sorted file paths and file names
file_paths1 = glob.glob('./dataset/*')
file_paths1.sort()

file_names1 = [os.path.basename(path) for path in file_paths1]
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
file_path = './dataset.json'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, 'w') as json_file:
    json.dump(file_json, json_file, indent=4)

# Define path to dataset and model
root_dir = './'
dataset_path = "./dataset.json"

# ===== MRI PROCESSING CLASSES =====
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

class ImageToGIF:
    def __init__(self):
        self.frames = []

    def add(self, image, mask):
        image = Image.fromarray(np.uint8(image * 255), 'L').convert("RGBA")
        mask = Image.fromarray(np.uint8(mask * 255), 'L')
        mask_colored = ImageOps.colorize(mask, black="black", white="red").convert("RGBA")
        mask_colored = mask_colored.resize(image.size)
        combined = Image.blend(image, mask_colored, alpha=0.5)
        self.frames.append(combined)

    def save(self, filename, fps=10):
        if not self.frames:
            raise ValueError("No frames to save")
        self.frames[0].save(filename, save_all=True, append_images=self.frames[1:], loop=0, duration=1000/fps)

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
model_path = os.path.join(root_dir, "best_distilled_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
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

# ===== HELPER FUNCTIONS =====
def get_sample_by_id(idx):
    """Get a sample from the dataset by ID"""
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

def create_gif_from_slices(nifti_file, mask_file, output_filename, orientation):
    img = nib.load(nifti_file).get_fdata()
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-10)

    if mask_file:
        mask = nib.load(mask_file).get_fdata()
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-10)
    else:
        mask = np.zeros_like(img)

    gif_maker = ImageToGIF()
    slices, get_image, get_mask = {
        'axial': (img.shape[2], lambda i: np.rot90(img[:, :, i]), lambda i: np.clip(np.rot90(mask[:, :, i]), 0, 1)),
        'sagittal': (img.shape[0], lambda i: np.rot90(img[i, :, :], 1), lambda i: np.clip(np.rot90(mask[i, :, :], 1), 0, 1)),
        'coronal': (img.shape[1], lambda i: np.rot90(img[:, i, :], 2), lambda i: np.clip(np.rot90(mask[:, i, :], 2), 0, 1))
    }[orientation]

    step = max(1, slices // 50) if slices > 50 else 1
    for i in range(0, slices, step):
        gif_maker.add(get_image(i), get_mask(i))
    gif_maker.save(output_filename)

def create_3d_models(nifti_file, mask_file, output_base):
    """Create a single 3D model: cutaway brain with tumor, fuller brain."""
    img_nii = nib.load(nifti_file)
    img = img_nii.get_fdata()

    if mask_file:
        mask = nib.load(mask_file).get_fdata()
    else:
        mask = np.zeros_like(img)

    # Store paths of created models
    model_paths = {}

    # APPROACH 1: Create a cutaway view of the brain (more conservative cut)
    midpoint = img.shape[0] // 1  # Adjust midpoint to cut less of the brain
    cutaway_mask = np.ones_like(img)
    cutaway_mask[midpoint:, :, :] = 0  # Cut only a portion of the brain
    cutaway_brain = img * cutaway_mask

    try:
        # Create the cutaway brain mesh
        brain_verts, brain_faces, brain_normals, _ = measure.marching_cubes(cutaway_brain, level=0.5)
        brain_mesh = trimesh.Trimesh(vertices=brain_verts, faces=brain_faces, vertex_normals=brain_normals)
        brain_mesh.visual.face_colors = [220, 220, 220, 255]  # Light gray
    except Exception as e:
        print(f"Failed to create brain mesh: {e}")
        return model_paths  # if no brain mesh, no other meshes.

    # APPROACH 2: Create a mesh for just the tumor with bright colors
    tumor_meshes = []
    if mask_file is not None:
        for label, color in [
            (1, [255, 0, 0, 255]),     # Necrotic core - Red
            (2, [0, 255, 0, 255]),     # Peritumoral edema - Green
            (4, [0, 0, 255, 255])      # Enhancing tumor - Blue
        ]:
            # Extract this specific tumor type
            tumor_region = np.zeros_like(img)
            tumor_region[mask == label] = img[mask == label]

            # Only proceed if this tumor type exists
            if np.sum(tumor_region) > 0:
                try:
                    # Create mesh for this tumor type
                    tumor_verts, tumor_faces, tumor_normals, _ = measure.marching_cubes(tumor_region, level=0.3)
                    tumor_mesh = trimesh.Trimesh(vertices=tumor_verts, faces=tumor_faces, vertex_normals=tumor_normals)

                    # Color the tumor mesh
                    tumor_mesh.visual.face_colors = color

                    # Add to our collection
                    tumor_meshes.append(tumor_mesh)
                except Exception as e:
                    print(f"Failed to create mesh for tumor type {label} - may not exist: {e}")

        # Export all the different visualizations
        # 1. Cutaway brain with tumors
        if tumor_meshes:
            combined_cutaway = trimesh.util.concatenate([brain_mesh] + tumor_meshes)
            combined_cutaway_filename = f"{output_base}_brain_cutaway_with_tumor.ply"
            combined_cutaway_path = os.path.join(MODEL_FOLDER, combined_cutaway_filename)
            combined_cutaway.export(combined_cutaway_path)
            model_paths['brain_cutaway_with_tumor'] = combined_cutaway_filename
            print("Created cutaway brain with tumors")

    return model_paths

# ===== API ROUTES =====

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

@app.route('/create_gif', methods=['POST'])
def create_gifs():
    try:
        if 'nifti_file' not in request.files:
            return jsonify({"error": "No nifti_file part"}), 400
        
        nifti_file = request.files['nifti_file']
        nifti_filename = secure_filename(nifti_file.filename)
        nifti_path = os.path.join(UPLOAD_FOLDER, nifti_filename)
        nifti_file.save(nifti_path)

        mask_path = None
        if 'mask_file' in request.files:
            mask_file = request.files['mask_file']
            if mask_file.filename:
                mask_filename = secure_filename(mask_file.filename)
                mask_path = os.path.join(UPLOAD_FOLDER, mask_filename)
                mask_file.save(mask_path)

        # Create GIFs for different orientations
        gif_urls = {}
        base_url = request.host_url.rstrip('/')
        for orientation in ['axial', 'sagittal', 'coronal']:
            gif_filename = f'output_{orientation}_{os.path.splitext(nifti_filename)[0]}.gif'
            gif_path = os.path.join(GIF_FOLDER, gif_filename)
            create_gif_from_slices(nifti_path, mask_path, gif_path, orientation)
            gif_urls[orientation] = f"{base_url}/gif/{gif_filename}"

        return jsonify(gif_urls)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/create_3d', methods=['POST'])
def create_3d():
    try:
        if 'nifti_file' not in request.files:
            return jsonify({"error": "No nifti_file part"}), 400
        
        nifti_file = request.files['nifti_file']
        nifti_filename = secure_filename(nifti_file.filename)
        nifti_path = os.path.join(UPLOAD_FOLDER, nifti_filename)
        nifti_file.save(nifti_path)

        mask_path = None
        if 'mask_file' in request.files:
            mask_file = request.files['mask_file']
            if mask_file.filename:
                mask_filename = secure_filename(mask_file.filename)
                mask_path = os.path.join(UPLOAD_FOLDER, mask_filename)
                mask_file.save(mask_path)
        
        # Create base filename for 3D models
        output_base = os.path.splitext(nifti_filename)[0]
        
        # Generate 3D models
        model_paths = create_3d_models(nifti_path, mask_path, output_base)
        
        # Create URLs for each model
        base_url = request.host_url.rstrip('/')
        model_urls = {model_type: f"{base_url}/model/{filename}" 
                     for model_type, filename in model_paths.items()}
        
        return jsonify({"models": model_urls})
        
    except Exception as e:
        import traceback
        print(f"Error creating 3D models: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/gif/<filename>')
def serve_gif(filename):
    return send_from_directory(GIF_FOLDER, filename)

@app.route('/model/<filename>')
def serve_model(filename):
    return send_from_directory(MODEL_FOLDER, filename)

@app.route('/debug')
def debug():
    return jsonify({
        'base_dir': BASE_DIR,
        'upload_folder': UPLOAD_FOLDER,
        'gif_folder': GIF_FOLDER,
        'model_folder': MODEL_FOLDER,
        'upload_exists': os.path.exists(UPLOAD_FOLDER),
        'gif_exists': os.path.exists(GIF_FOLDER),
        'model_exists': os.path.exists(MODEL_FOLDER),
        'gifs': os.listdir(GIF_FOLDER) if os.path.exists(GIF_FOLDER) else [],
        'models': os.listdir(MODEL_FOLDER) if os.path.exists(MODEL_FOLDER) else [],
        'model_file_exists': os.path.exists(model_path),
        'dataset_file_exists': os.path.exists(dataset_path),
        'dataset_folder_exists': os.path.exists('./dataset'),
        'num_dataset_files': len(file_paths1) if 'file_paths1' in locals() else 0
    })

@app.route('/')
def index():
    return jsonify({
        'status': 'Server is running',
        'endpoints': {
            'process_sample': '/process_sample (POST)',
            'dataset_info': '/dataset_info (GET)', 
            'create_gif': '/create_gif (POST)',
            'create_3d': '/create_3d (POST)',
            'debug': '/debug (GET)'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)