import os
import numpy as np
import pickle

def save_data(camera, output_dir, frame):
    # Extract camera data
    rgb_image = camera.data.output["rgb"].cpu().numpy()[0]
    depth_image = camera.data.output["distance_to_image_plane"].cpu().numpy()[0]
    seg_image = camera.data.output["semantic_segmentation"].cpu().numpy()[0]
    metadata = {
        'intrinsic_matrix': camera.data.intrinsic_matrices[0].cpu().numpy(),
        'camera_position': camera.data.pos_w[0].cpu().numpy(),
        'camera_orientation': camera.data.quat_w_ros[0].cpu().numpy()
    }

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Save RGB image
    rgb_path = os.path.join(output_dir, f'rgb_{frame:04d}.png')
    depth_path = os.path.join(output_dir, f'depth_{frame:04d}.npy')
    seg_path = os.path.join(output_dir, f'seg_{frame:04d}.png')
    metadata_path = os.path.join(output_dir, f'meta_data_{frame:04d}.pkl')

    # Save depth and segmentation as numpy arrays
    np.save(depth_path, depth_image)
    np.save(seg_path, seg_image)

    # Save RGB as an image
    from PIL import Image
    Image.fromarray(rgb_image).save(rgb_path)

    # Save metadata as pickle file
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)