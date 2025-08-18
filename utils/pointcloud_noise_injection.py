import numpy as np
import argparse


def add_gaussian_noise(point_clouds, noise_scale=0.01):
    """
    Add Gaussian noise to point cloud coordinates.

    Parameters:
    - point_clouds: Point cloud data (N, num_points, 6), where the last dimension is xyz+normal
    - noise_scale: Standard deviation of the Gaussian noise, default is 0.01
    """
    noise = np.random.normal(
        loc=0.0,
        scale=noise_scale,
        size=point_clouds.shape
    ).astype(np.float32)

    # Add noise only to the coordinate part (first 3 dimensions), keep normals unchanged
    noisy_point_clouds = point_clouds.copy()
    noisy_point_clouds[..., :3] += noise[..., :3]
    return noisy_point_clouds


def process_modelnet40_data(input_path, output_path, noise_scale=0.01):
    """
    Process ModelNet40 binary data file, add noise, and save the result.

    Parameters:
    - input_path: Path to the original data file (.dat)
    - output_path: Path to save the noisy data
    - noise_scale: Noise intensity (standard deviation), default is 0.01
    """
    # Read binary data (ModelNet40 format: 6 values per point (xyz+normal))
    with open(input_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

    # Reshape to (number of samples, points per sample, 6)
    points_per_sample = 8192  # Default for ModelNet40
    num_samples = data.shape[0] // (points_per_sample * 6)
    data = data.reshape(num_samples, points_per_sample, 6)

    # Add noise
    noisy_data = add_gaussian_noise(data, noise_scale=noise_scale)

    # Save as binary file
    with open(output_path, 'wb') as f:
        noisy_data.flatten().tofile(f)

    print(f"Processing complete! Saved to {output_path}")
    print(f"Data shape: {noisy_data.shape}, Noise scale: {noise_scale}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add noise to ModelNet40 point cloud data")
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path to the original ModelNet40 data file (.dat)")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save the noisy data")
    parser.add_argument("--noise-scale", type=float, default=0.01,
                        help="Noise intensity (standard deviation), default is 0.01")
    args = parser.parse_args()

    process_modelnet40_data(
        input_path=args.input_path,
        output_path=args.output_path,
        noise_scale=args.noise_scale
    )
