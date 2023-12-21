import cv2
import numpy as np

# Realistische intrinsische Kamera-Parameter
f = 800  # angenommene Brennweite in Pixel
cx, cy = 640, 480  # angenommene Koordinaten des Hauptpunkts (z.B. Bildmitte bei einer 1280x960 Kamera)
chessboard_size = (6, 8)  # Größe des Schachbretts (Anzahl der inneren Ecken)


def compute_homography_3d_to_2d(world_points, image_points):
    if len(world_points) != len(image_points):
        raise ValueError("Die Anzahl der Welt- und Bild-Punkte muss gleich sein")

    # Erstelle Matrix A
    A = []
    for (X, Y, Z), (x, y) in zip(world_points, image_points):
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])

    A = np.array(A)

    # SVD von A
    U, S, Vh = np.linalg.svd(A)

    # Die Homographie ist die letzte Zeile von Vh und muss in eine 3x4 Matrix umgeformt werden
    H = Vh[-1].reshape((3, 4))
    # Füge eine Fehlerbehandlung für den Fall hinzu, dass H[2, 3] Null ist
    if H[2, 3] == 0:
        raise ValueError("Homographie-Matrix ist nicht normalisierbar: H[2, 3] ist Null")

    return H / H[2, 3]  # Normalisiere, so dass H[2, 3] = 1


def extractCameraParameters(H):
    # Annahme: Die Kamera-Kalibrierungsmatrix K ist bekannt
    # In der Praxis muss diese aus der Kamerakalibrierung bekannt sein
    # Hier ein Beispiel für eine generische Kamera-Kalibrierungsmatrix
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])  # f ist die Brennweite, cx und cy sind die Koordinaten des Hauptpunkts

    # Normalisiere die Homographie-Matrix
    H = H / np.linalg.norm(H[:,0])

    # Berechne die inverse der Kalibrierungsmatrix
    K_inv = np.linalg.inv(K)

    # Berechne die Rotations- und Translationsmatrix
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    lambda_ = 1 / np.linalg.norm(np.dot(K_inv, h1))

    r1 = lambda_ * np.dot(K_inv, h1)
    r2 = lambda_ * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)
    t = lambda_ * np.dot(K_inv, h3)

    # Stelle sicher, dass die Rotationsmatrix R eine gültige Rotationsmatrix ist
    R = np.column_stack((r1, r2, r3))
    U, S, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)

    return R, t

# Beispielanwendung mit realistischeren Punktkorrespondenzen

# Schachbrett-Muster als Bild laden
image = cv2.imread('../images/image6x9.jpg')
try:
    # Finde die Ecken des Schachbretts
    ret, corners = cv2.findChessboardCorners(image, chessboard_size)

    # Überprüfe, ob Ecken gefunden wurden
    if not ret:
        raise ValueError("Ecken des Schachbretts konnten nicht gefunden werden")

    # Definiere Weltkoordinaten für das Schachbrett
    world_points = np.zeros((np.prod(chessboard_size), 3), np.float32)
    world_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)


    # Bildpunkte anpassen (flatten)
    image_points = corners.reshape(-1, 2)

    # Berechne die Homographie
    homography = compute_homography_3d_to_2d(world_points, image_points)
    print("Homographie-Matrix:")
    print(homography)

    # Extrahiere Kameraparameter
    R, t = extractCameraParameters(homography)
    # Ausgabe der wichtigen Kameraparameter
    print("Wichtige Kameraparameter:")
    print("Brennweite (f):", f)
    print("Hauptpunktkoordinaten (cx, cy): ({}, {})".format(cx, cy))
    print("\nRotationsmatrix R:\n", R)
    print("\nTranslationsvektor t:\n", t)
except ValueError as e:
    print(e)
