import matplotlib.pyplot as plt

def plot_masks_vs_tiles(tiles, masks, num_plots=10):
    """
    Plots image tiles alongside their corresponding masks.

    Args:
        tiles (list): List of image tiles as NumPy arrays.
        masks (list): List of mask images as NumPy arrays.
        num_plots (int, optional): Number of tiles/masks to plot. Defaults to 10.

    Returns:
        None
    """
    num_plots = min(num_plots, len(tiles))  # Ensure not to exceed the number of available tiles

    fig, axs = plt.subplots(num_plots, 2, figsize=(10, 5 * num_plots))

    for i in range(num_plots):
        # Plot tile
        axs[i, 0].imshow(tiles[i])
        axs[i, 0].set_title('Tile')
        axs[i, 0].axis('off')

        # Plot mask
        axs[i, 1].imshow(masks[i], cmap='binary')
        axs[i, 1].set_title('Mask')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
