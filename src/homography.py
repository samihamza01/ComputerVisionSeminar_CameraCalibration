import cv2
import numpy as np
import os

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


def estimate_overall_camera_matrix(homographies, image_size):
    def vij(h, i, j):
        return np.array([
            h[0, i] * h[0, j], h[0, i] * h[1, j] + h[1, i] * h[0, j],
            h[1, i] * h[1, j], h[2, i] * h[0, j] + h[0, i] * h[2, j],
            h[2, i] * h[1, j] + h[1, i] * h[2, j], h[2, i] * h[2, j]
        ])

    n = len(homographies)  # Anzahl der Bilder

    V = []
    for H in homographies:
        V.append(vij(H, 0, 1))
        V.append(vij(H, 0, 0) - vij(H, 1, 1))

    if n == 2:
        # Zusätzliche Gleichung für den Fall n = 2 hinzufügen
        V.append([0, 1, 0, 0, 0, 0])

    V = np.array(V)
    U, S, Vt = np.linalg.svd(V)
    b = Vt[-1]

    if n == 1:
        # Nur zwei Parameter lösbar, wenn n = 1
        # Annahme: u0 und v0 sind bekannt (z.B. im Bildzentrum)
        v0, u0 = image_size[0] / 2, image_size[1] / 2
        K = np.array([[b[0], 0, u0], [0, b[1], v0], [0, 0, 1]])
    else:
        # Matrix B aus b berechnen
        B = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])

        v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)
        lambda_ = B[2, 2] - ((B[0, 2] ** 2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0])
        alpha = np.sqrt(lambda_ / B[0, 0])
        beta = np.sqrt(lambda_ * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1] ** 2))
        gamma = -B[0, 1] * alpha ** 2 * beta / lambda_
        u0 = gamma * v0 / beta - B[0, 2] * alpha ** 2 / lambda_

        K = np.array([[alpha, 0, u0], [0, beta, v0], [0, 0, 1]])

    return K


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


def estimate_radial_distortion(imgpoints, objpoints, mtx, rvecs, tvecs):
    num_points = sum([len(p) for p in imgpoints])
    A = np.zeros((2 * num_points, 2))  # 2 equations for each point
    b = np.zeros((2 * num_points, 1))

    index = 0
    # Convert objpoints, rvecs, tvecs, and mtx to np.float64

    for i in range(len(objpoints)):
        # Convert objpoints, rvecs, tvecs, and mtx to np.float64
        objpoints_i = np.array(objpoints[i], dtype=np.float64).reshape(-1, 1, 2)
        rvecs_i = np.array(rvecs[i], dtype=np.float64).reshape(-1, 1, 3)
        tvecs_i = np.array(tvecs[i], dtype=np.float64).reshape(-1, 1, 3)
        mtx_i = np.array(mtx, dtype=np.float64).reshape(-1, 1, 3).reshape(3, 3)
        print(f"Parameters before using cv2.projectPoints:")
        print(f"objpoints: {objpoints_i}")
        print(f"rvecs: {rvecs_i}")
        print(f"tvecs: {tvecs_i}")
        print(f"mtx: {mtx_i}")
        print("===================================================")
        try:
            projected_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, None)
        except cv2.error as e:
            error_type = "OpenCV Error"
            context = "Error occurred during cv2.projectPoints operation"
            input_parameters = f"objpoints: {objpoints_i}, rvecs: {rvecs_i}, tvecs: {tvecs_i}, mtx: {mtx_i}"
            additional_details = f"Error message: {e}, Error code: {e.code}"  # If applicable
            error_message = f"{error_type}\n{context}\n{input_parameters}\n{additional_details}"
            print(error_message)
        for j in range(len(projected_points)):
            u, v = imgpoints[i][j][0][0], imgpoints[i][j][0][1]
            u0, v0 = mtx[0, 2], mtx[1, 2]
            x, y = projected_points[j][0][0], projected_points[j][0][1]

            r2 = x**2 + y**2
            r4 = r2**2

            # Aufstellen der Gleichungen
            A[2*index] = [(u - u0) * r2, (u - u0) * r4]
            b[2*index] = u - x

            A[2*index + 1] = [(v - v0) * r2, (v - v0) * r4]
            b[2*index + 1] = v - y

            index += 1

    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return k[0][0], k[1][0]


def extract_rotation_translation(A, homographies):
    """
    Extracts rotation vectors in Rodriguez form and translation vectors from a list of homography matrices,
    given the overall intrinsic camera matrix A.
    :param A: Intrinsic camera matrix
    :param homographies: List of homography matrices
    :return: List of R_vec (Rodriguez form of rotation vectors), list of t (translation vectors)
    """

    rotation_vecs = []
    translations = []

    for H in homographies:

        # Extract rotation and translation
        inv_A = np.linalg.inv(A)
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = np.cross(h1, h2)

        lambda_ = 1 / np.linalg.norm(inv_A.dot(h1))
        r1 = lambda_ * inv_A.dot(h1)
        r2 = lambda_ * inv_A.dot(h2)
        r3 = np.cross(r1, r2)
        t = lambda_ * inv_A.dot(H[:, 2])

        R = np.column_stack((r1, r2, r3))
        if np.linalg.det(R) < 0:
            R = -R

        # Konvertieren in Rodriguez-Form
        R_vec, _ = cv2.Rodrigues(R)
        rotation_vecs.append(R_vec)
        translations.append(t)

    return rotation_vecs, translations



if __name__ == "__main__":
	# Liste der Bilddateien
	image_files = os.listdir("./calibration_images/")
	print(image_files)

	# Schachbrett Parameter
	width, height = 6, 9

	# Weltkoordinaten für die Ecken definieren
	world_corners = np.zeros((width * height, 2), np.float32)
	for i in range(height):
		for j in range(width):
			world_corners[i * width + j] = [j, i]

	# Listen zur Speicherung der Homographie-Matrizen und Bildgrößen
	homographies_my_function = []
	homographies_opencv = []

	# Listen für 3D-Punkte in der Welt und 2D-Punkte im Bild
	object_points = []  # 3D-Punkte in der Welt
	image_points = []   # 2D-Punkte im Bild

	# Weltkoordinaten für 3D-Punkte
	objp = np.zeros((height*width, 3), np.float32)
	objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

	for image_idx, image_file in enumerate(image_files):
		if image_idx ==3:
			break
		image = cv2.imread("./calibration_images/"+image_file)
		if image_idx == 0:
			image_size = image.shape[0:2]  # Breite und Höhe hinzufügen
		else:
			if image_size != image.shape[0:2]:
				raise ValueError("Image shapes must be equal for calibration.")

		# Finden der Ecken auf dem Schachbrett
		success, found_corners = cv2.findChessboardCorners(image, (width, height))

		if success:
			# Eckenpositionen verfeinern
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
			found_corners_refined = cv2.cornerSubPix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
													found_corners, (11, 11), (-1, -1), criteria)

			# Anpassen der Formatierung der Punkte
			found_corners_refined = found_corners_refined.squeeze()

			# Homographie mit Ihrer Funktion berechnen
			H_my_function = compute_homography(world_corners, found_corners_refined)
			homographies_my_function.append(H_my_function)

			# Homographie mit OpenCV berechnen
			H_opencv, _ = cv2.findHomography(world_corners, found_corners_refined)
			homographies_opencv.append(H_opencv)

			# Punkte für Kamerakalibrierung hinzufügen
			object_points.append(objp)
			image_points.append(found_corners_refined)

	# Gesamtkameramatrix mit Ihren Homographien schätzen
	A_my_homographies = estimate_overall_camera_matrix(homographies_my_function, image_size)

	# Gesamtkameramatrix mit OpenCV Homographien schätzen
	A_opencv_homographies = estimate_overall_camera_matrix(homographies_opencv, image_size)

	# Kamerakalibrierung mit OpenCV
	ret, K_cv, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size[::-1], None, None)

	# Ausgabe der Ergebnisse
	print("Meine Gesamtkameramatrix aus meinen Homographien:")
	print(A_my_homographies)
	print("Gesamtkameramatrix aus OpenCV Homographien:")
	print(A_opencv_homographies)
	print("OpenCV Kameramatrix (K) mit calibrateCamera:")
	print(K_cv)

	# Vergleich der Homographie-Matrizen
	for i, (H_my, H_cv) in enumerate(zip(homographies_my_function, homographies_opencv)):
		print(f"Homographie für Bild {i+1}:")
		print("Meine Homographie:")
		print(H_my)
		print("OpenCV Homographie:")
		print(H_cv)

