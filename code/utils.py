import rasterio
import matplotlib.pyplot as plt
import numpy as np

def sample_tiff(in_path, out_path=None, chunk_size = (1000, 1000), xy_coords = (0,0), plot_result=False):

    # Open the TIFF file
    with rasterio.open(in_path) as src:

        window = rasterio.windows.Window(xy_coords[0], xy_coords[1], chunk_size[0], chunk_size[1])
        
        # Read the chunk data
        # Assuming the TIFF is RGB, we read the first three bands
        chunk_data = src.read([1, 2, 3], window=window)

        # Write new file
        if out_path is not None:

            # Get the affine transform from the original file to use on the window.
            # In Euclidean geometry, an affine transformation or affinity (from the Latin, affinis, "connected with")
            # is a geometric transformation that preserves lines and parallelism, but not necessarily Euclidean
            # distances and angles.
            transform = src.window_transform(window)

            # Define the metadata for the new chunk file
            chunk_meta = src.meta.copy()
            chunk_meta.update({
                'driver': 'GTiff',
                'height': window.height,
                'width': window.width,
                'transform': transform
            })

            # Write the chunk data to a new TIFF file
            with rasterio.open(out_path, 'w', **chunk_meta) as dest:
                dest.write(chunk_data)

        # Else, return output
        else:
            return chunk_data
        
    # Visualize the chunk
    if plot_result:

        # Convert the data to a format suitable for matplotlib
        # We need to move the channel dimension to the end
        chunk_data = np.moveaxis(chunk_data, 0, -1)

        plt.figure(figsize=(10, 10))
        plt.imshow(chunk_data)
        plt.title('Chunk from Upper Left Corner')
        plt.axis('off')  # Hide the axis
        plt.show()

    