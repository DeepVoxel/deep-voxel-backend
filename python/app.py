from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import nibabel as nib
import numpy as np
from PIL import Image, ImageOps
from flask_cors import CORS
from skimage import measure
import trimesh
import matplotlib.cm as cm

app = Flask(__name__)   
CORS(app)  # Enable CORS for all routes

# Define absolute paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER = '/tmp/static'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
GIF_FOLDER = os.path.join(STATIC_FOLDER, 'gifs')
MODEL_FOLDER = os.path.join(STATIC_FOLDER, 'models')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GIF_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GIF_FOLDER'] = GIF_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER


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
    print(f"Starting 3D model creation with {nifti_file} and {mask_file}")
    
    try:
        img_nii = nib.load(nifti_file)
        img = img_nii.get_fdata()
        
        # Downsample the image to reduce memory usage
        downsample_factor = 2  # Reduce the image size to conserve memory
        img_downsampled = img[::downsample_factor, ::downsample_factor, ::downsample_factor]
        print(f"Original shape: {img.shape}, Downsampled shape: {img_downsampled.shape}")
        
        # Normalize image to 0-1 range
        img_min, img_max = np.min(img_downsampled), np.max(img_downsampled)
        img_normalized = (img_downsampled - img_min) / (img_max - img_min + 1e-10)
        
        if mask_file:
            mask = nib.load(mask_file).get_fdata()
            mask_downsampled = mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
        else:
            mask_downsampled = np.zeros_like(img_downsampled)

        # Store paths of created models
        model_paths = {}

        # APPROACH 1: Create a cutaway view of the brain (more conservative cut)
        midpoint = img_downsampled.shape[0] // 2  # Cut in half for better visualization
        cutaway_mask = np.ones_like(img_downsampled)
        cutaway_mask[midpoint:, :, :] = 0  # Cut only a portion of the brain
        cutaway_brain = img_normalized * cutaway_mask

        print("Generating brain mesh with marching cubes...")
        try:
            # Use a lower threshold for better mesh generation
            threshold = 0.3  # Lower threshold may capture more of the brain structure
            brain_verts, brain_faces, brain_normals, _ = measure.marching_cubes(cutaway_brain, level=threshold)
            
            print(f"Brain mesh generated: {len(brain_verts)} vertices, {len(brain_faces)} faces")
            
            # Create the brain mesh with simplified geometry (fewer triangles)
            brain_mesh = trimesh.Trimesh(vertices=brain_verts, faces=brain_faces, vertex_normals=brain_normals)
            
            # Simplify the mesh to reduce file size and memory usage (if it has enough faces)
            if len(brain_faces) > 5000:
                target_faces = min(len(brain_faces) // 2, 10000)  # Reduce to 50% or max 10K faces
                print(f"Simplifying brain mesh from {len(brain_faces)} to ~{target_faces} faces")
                brain_mesh = brain_mesh.simplify_quadratic_decimation(target_faces)
            
            brain_mesh.visual.face_colors = [220, 220, 220, 255]  # Light gray
            
            # Export just the brain mesh first as a test
            brain_only_filename = f"{output_base}_brain_only.ply"
            brain_only_path = os.path.join(MODEL_FOLDER, brain_only_filename)
            brain_mesh.export(brain_only_path)
            model_paths['brain_only'] = brain_only_filename
            print(f"Exported brain mesh to {brain_only_path}")
            
        except Exception as e:
            print(f"Failed to create brain mesh: {e}")
            import traceback
            print(traceback.format_exc())
            return model_paths  # Return early if we can't create the brain mesh
        
        # APPROACH 2: Create a mesh for just the tumor with bright colors
        tumor_meshes = []
        if mask_file is not None:
            for label, color in [
                (1, [255, 0, 0, 255]),     # Necrotic core - Red
                (2, [0, 255, 0, 255]),     # Peritumoral edema - Green
                (3, [255, 0, 0, 255])      # Enhancing tumor - Red
            ]:
                # Extract this specific tumor type
                tumor_region = np.zeros_like(img_normalized)
                tumor_region[mask_downsampled == label] = 1.0  # Binary mask is more stable
                
                # Only proceed if this tumor type exists and has enough voxels
                if np.sum(tumor_region) > 10:  # Ensure there are enough voxels
                    try:
                        print(f"Generating tumor mesh for label {label}...")
                        # Create mesh for this tumor type with a lower threshold
                        tumor_verts, tumor_faces, tumor_normals, _ = measure.marching_cubes(tumor_region, level=0.5)
                        
                        # Only proceed if we got enough vertices
                        if len(tumor_verts) > 0 and len(tumor_faces) > 0:
                            tumor_mesh = trimesh.Trimesh(vertices=tumor_verts, faces=tumor_faces, vertex_normals=tumor_normals)
                            
                            # Simplify tumor mesh if needed
                            if len(tumor_faces) > 5000:
                                tumor_mesh = tumor_mesh.simplify_quadratic_decimation(min(len(tumor_faces) // 2, 5000))
                            
                            # Color the tumor mesh
                            tumor_mesh.visual.face_colors = color
                            
                            # Add to our collection
                            tumor_meshes.append(tumor_mesh)
                            print(f"Created tumor mesh for label {label}: {len(tumor_verts)} vertices")
                            
                            # Export individual tumor meshes for debugging
                            tumor_filename = f"{output_base}_tumor_label_{label}.ply"
                            tumor_path = os.path.join(MODEL_FOLDER, tumor_filename)
                            tumor_mesh.export(tumor_path)
                            model_paths[f'tumor_label_{label}'] = tumor_filename
                            
                    except Exception as e:
                        print(f"Failed to create mesh for tumor type {label}: {e}")
                        import traceback
                        print(traceback.format_exc())
            
            # Export all the different visualizations
            # 1. Cutaway brain with tumors
            if tumor_meshes:
                try:
                    print("Combining brain and tumor meshes...")
                    combined_cutaway = trimesh.util.concatenate([brain_mesh] + tumor_meshes)
                    
                    combined_cutaway_filename = f"{output_base}_brain_cutaway_with_tumor.ply"
                    combined_cutaway_path = os.path.join(MODEL_FOLDER, combined_cutaway_filename)
                    
                    print(f"Exporting combined mesh with {len(combined_cutaway.vertices)} vertices")
                    combined_cutaway.export(combined_cutaway_path)
                    model_paths['brain_cutaway_with_tumor'] = combined_cutaway_filename
                    print(f"Created cutaway brain with tumors: {combined_cutaway_path}")
                except Exception as e:
                    print(f"Failed to create combined mesh: {e}")
                    import traceback
                    print(traceback.format_exc())

        return model_paths
        
    except Exception as e:
        print(f"Exception in create_3d_models: {e}")
        import traceback
        print(traceback.format_exc())
        return {}


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
    


@app.route('/test_3d')
def test_3d():
    try:
        # Generate a simple sphere
        print("Creating test sphere mesh...")
        sphere = trimesh.creation.icosphere(radius=10)
        
        # Save it to the model folder
        test_path = os.path.join(MODEL_FOLDER, 'test_sphere.ply')
        sphere.export(test_path)
        
        # Also try a simple cube
        box = trimesh.creation.box(extents=[20, 20, 20])
        box.visual.face_colors = [255, 0, 0, 255]  # Red
        box_path = os.path.join(MODEL_FOLDER, 'test_box.ply')
        box.export(box_path)
        
        # Return success with file paths
        return jsonify({
            "success": True, 
            "sphere_path": test_path,
            "box_path": box_path,
            "sphere_url": f"{request.host_url.rstrip('/')}/model/test_sphere.ply",
            "box_url": f"{request.host_url.rstrip('/')}/model/test_box.ply"
        })
    except Exception as e:
        import traceback
        print(f"Error in test_3d: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})
    

@app.route('/memory')
def memory_info():
    import psutil
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Get memory info
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    return jsonify({
        "process_memory_mb": memory_info.rss / (1024 * 1024),
        "system_total_mb": system_memory.total / (1024 * 1024),
        "system_available_mb": system_memory.available / (1024 * 1024),
        "system_used_percent": system_memory.percent
    })


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
        'models': os.listdir(MODEL_FOLDER) if os.path.exists(MODEL_FOLDER) else []
    })


# For Render deployment
port = int(os.environ.get("PORT", 5000))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)