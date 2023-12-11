import cv2
import numpy as np

def compute_homography(points_src, points_dst):
    """
    Computes the homography between two sets of points using SVD.
    :param points_src: A list of points in the source image (e.g., world coordinates of the chessboard)
    :param points_dst: A list of corresponding points in the destination image
    :return: Homography matrix
    """

    if len(points_src) != len(points_dst) or len(points_src) < 4:
        raise ValueError("At least 4 pairs of points must be provided and the number of points must be equal.")

    # Matrix to store the linear equations
    A = []

    for i in range(len(points_src)):
        x, y = points_src[i][0], points_src[i][1]
        u, v = points_dst[i][0], points_dst[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

    A = np.array(A)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(A)

    # The solution is the last vector of Vt
    H = Vt[-1].reshape(3, 3)

    # Normalize the homography matrix
    H = H / H[-1, -1]

    return H

def decompose_homography(H):
    """
    Estimates the intrinsic camera matrix A and the extrinsic parameters R and t
    from a given homography matrix.
    :param H: Homography matrix
    :return: A (intrinsic matrix), R (rotation matrix), t (translation vector)
    """

    # Normalize the homography matrix
    H = H / H[2, 2]

    # Estimate the intrinsic camera matrix A
    # Assuming the optical center is at the image center
    width = 961  # Set the width of your image
    height = 931 # Set the height of your image
    cx = width / 2
    cy = height / 2
    A = np.array([
        [H[0, 0], H[0, 1], cx],
        [H[1, 0], H[1, 1], cy],
        [0, 0, 1]
    ])

    # Extract the rotation and translation matrix
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = np.cross(h1, h2)

    norm = np.linalg.norm(np.linalg.inv(A).dot(h1))
    r1 = h1 / norm
    r2 = h2 / norm
    r3 = h3 / norm
    t = H[:, 2] / norm

    # Ensure the determinant of R is positive
    R = np.column_stack((r1, r2, r3))
    if np.linalg.det(R) < 0:
        R = -R

    return A, R, t

def calculate_reprojection_error(points_src, points_dst, H):
    total_error = 0
    for i in range(len(points_src)):
        src_pt = np.array([points_src[i][0], points_src[i][1], 1]).reshape(-1, 1)
        est_dst = np.dot(H, src_pt)
        est_dst /= est_dst[2]
        actual_dst = np.array([points_dst[i][0], points_dst[i][1], 1])
        error = np.linalg.norm(actual_dst - est_dst.flatten())
        total_error += error
    return total_error / len(points_src)

# Load the chessboard image
image_path = "../calibration_images/02.jpg"  # This path might need to be adjusted
image = cv2.imread(image_path)

# Define the number of corners in width and height
width = 7
height = 9

# Find the corners on the chessboard
success, found_corners = cv2.findChessboardCorners(image, (width, height))

if success:
    # Refine the corner positions
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    found_corners_refined = cv2.cornerSubPix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                             found_corners, (11, 11), (-1, -1), criteria)

    # Define world coordinates for the corners
    world_corners = np.zeros((width * height, 2), np.float32)
    for i in range(height):
        for j in range(width):
            world_corners[i * width + j] = [j, i]

    # Compute the homography
    homography_matrix = compute_homography(world_corners, found_corners_refined.squeeze())

    # Estimate the intrinsic matrix and extrinsic parameters from the homography matrix
    A, R, t = decompose_homography(homography_matrix)

    # Reprojection error
    reprojection_error = calculate_reprojection_error(world_corners, found_corners_refined.squeeze(), homography_matrix)

    # Visualize reprojection error in the image
    for i, corner in enumerate(found_corners_refined):
        cv2.circle(image, tuple(corner.ravel().astype(int)), 5, (0, 0, 255), -1)
        world_point = np.array([world_corners[i][0], world_corners[i][1], 1]).reshape(-1, 1)
        estimated_point = np.dot(homography_matrix, world_point)
        estimated_point /= estimated_point[2]
        cv2.circle(image, tuple(estimated_point[0:2].ravel().astype(int)), 5, (255, 0, 0), 2)

    # Display the image with the visualization of the reprojection error
    cv2.imshow('Chessboard Corners with Reprojection Error', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Homography Matrix:")
    print(homography_matrix)

    print("Intrinsic Camera Matrix A:")
    print(A)

    print("Rotation Matrix R:")
    print(R)

    print("Translation Vector t:")
    print(t)

    print("Reprojection Error:")
    print(reprojection_error)
else:
    print("Corners not found")