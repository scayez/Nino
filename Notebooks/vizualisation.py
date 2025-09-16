import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image_with_pixels_over_threshold(image, threshold=20):
    """
    Display a 2D image with pixels above a threshold marked in red.

    Parameters
    ----------
    image : np.ndarray
        2D image (height, width)
    threshold : float
        Threshold used to detect pixels
    """
    # Find the coordinates of pixels > threshold
    coords = np.where(image > threshold)  # (rows, cols)

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.scatter(coords[1], coords[0], color='red', s=5, marker='o', label=f'>{threshold}')
    plt.title(f'Pixels > {threshold}')
    plt.axis('off')
    plt.legend()
    plt.show()

    # Print the number of pixels above the threshold
    print(f"Number of pixels > {threshold}: {len(coords[0])}")




import matplotlib.pyplot as plt
import numpy as np

def plot_six_images(data, indices, cmap='gray'):
    """
    Plot six images from a 3D array (N, H, W) in a 2x3 grid.

    Parameters
    ----------
    data : np.ndarray
        3D array of images (N, H, W)
    indices : list of int
        List of 6 indices to plot
    cmap : str, optional
        Colormap for displaying the images (default: 'gray')
    """
    if len(indices) != 6:
        raise ValueError("Please provide exactly 6 indices.")

    fig, axes = plt.subplots(2, 3, figsize=(14, 16))  # 2 rows, 3 columns

    for i, idx in enumerate(indices):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(data[idx], cmap=cmap)
        axes[row, col].set_title(f"Image {idx}", fontsize=12)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()



def create_video_from_images(images, output_path='output_video.mp4', fps=5):
    """
    Create a video file from the image sequence
    
    Parameters:
    images: numpy array of shape (n_images, height, width)
    output_path: path to save the video file
    fps: frames per second (speed of the video)
    """
    # Get image dimensions
    n_images, height, width = images.shape
    
    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    print(f"Creating video with {n_images} frames at {fps} FPS...")
    
    for i in range(n_images):
        # Get current image and convert to float32
        current_image = images[i, :, :].astype(np.float32)
        
        # Normalize image to 0-255 range
        img_normalized = cv2.normalize(current_image, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_normalized.astype(np.uint8)
        
        # Write frame to video file
        out.write(img_uint8)
        
        # Show progress
        if (i + 1) % 5 == 0 or i == 0 or i == n_images - 1:
            print(f"Processed frame {i+1}/{n_images}")
    
    # Release the video writer
    out.release()
    print(f"Video successfully saved as: {output_path}")
    print(f"Video dimensions: {width}x{height}, Duration: {n_images/fps:.1f} seconds")