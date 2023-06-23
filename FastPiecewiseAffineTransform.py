import time
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp
import matplotlib.pyplot as plt
from skimage import data

class FastPiecewiseAffineTransform(PiecewiseAffineTransform):
    def __call__(self, coords):
        coords = np.asarray(coords)

        simplex = self._tesselation.find_simplex(coords)

        # stack of affine transforms to be applied to every coord
        affines = np.stack([affine.params for affine in self.affines])[simplex]

        # convert coords to homgeneous points
        points = np.c_[coords, np.ones((coords.shape[0], 1))]
        # apply affine transform to every point
        result = np.einsum("ikj,ij->ik", affines, points)

        # coordinates outside of mesh
        result[simplex == -1, :] = -1
        # convert back to 2d coords
        result = result[:, :2]
        return result


def main2():
    #image = data.astronaut()
    image = data.grass()
    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
    dst_cols = src[:, 0]
    dst_rows *= 1.5
    dst_rows -= 1.5 * 50
    dst = np.vstack([dst_cols, dst_rows]).T
    tform = FastPiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = image.shape[0] - 1.5 * 50
    out_cols = cols
    out = warp(image, tform, output_shape=(out_rows, out_cols))

    fig, ax = plt.subplots()
    ax.imshow(out)
    ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    ax.axis((0, out_cols, out_rows, 0))
    plt.show()    

def main():
    # Create dummy data for testing
    src = np.random.rand(500, 2)
    dst = src + 0.1 * np.random.rand(500, 2)

    size = 2000
    src *= size
    dst *= size
    image = np.random.rand(size, size, 3)

    # Run the standard PiecewiseAffine transform
    tf = PiecewiseAffineTransform()
    tf.estimate(src, dst)
    start = time.time()
    slow = warp(image, tf)
    print(f"    PiecewiseAffineTransform took: {time.time() - start} s")

    # Run the FastPiecewiseAffine transform
    tf = FastPiecewiseAffineTransform()
    tf.estimate(src, dst)
    start = time.time()
    fast = warp(image, tf)
    print(f"FastPiecewiseAffineTransform took: {time.time() - start} s")

    # Ensure images are identical
    assert np.allclose(fast, slow)
    # Test FastPiecewiseAffineTransform with real image
    main2()


if __name__ == "__main__":
    main()
