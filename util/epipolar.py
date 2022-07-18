import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib import colors as mcolors


# F: 3x3 fundamental matrix maps points in image I1 to epipolar lines in image I2
# F = (K2^T)^-1 * E * K1^-1
# E = R [t]_x
# R: 3x3 relative rotation matrix between cameras
# t: 3x1 relative translation between cameras
# [t]_x = [0, -t(3), t(2); t(3), 0, -t(1); -t(2), t(1), 0]; cross-product matrix


def plot_epipolar(pixels1, pixels2, P1, P2, K1, K2, I1, I2):
    # pixels1: nx2 matrix of pixels in image 1 to plot epipolar lines for
    # pixels2: nx2 matrix of corresponding pixels in image 2 (can leave empty)
    # P1: 3x4 transformation matrix [R1|t1] of camera 1 (x_world = R1 * x_camera1 + t1)
    # P2: 3x4 transformation matrix [R2|t2] of camera 2 (x_world = R2 * x_camera2 + t2)
    # K1: 3x3 intrinsic matrix [fx, 0, px; 0, fy, py; 0, 0, 1] of camera 1
    # K2: 3x3 intrinsic matrix [fx, 0, px; 0, fy, py; 0, 0, 1] of camera 2
    # I1: HxWx3 image captured by camera 1
    # I2: HxWx3 image captured by camera 2

    num_points = pixels1.shape[0]

    F = fundamental_matrix(P1, P2, K1, K2)
    pix_h = np.concatenate([pixels1, np.ones((num_points, 1))], 1)
    epiLines = pix_h @ F.T
    pts = lineToBorderPoints(epiLines, I2.shape)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Epipolar lines')
    colors = get_n_colors(num_points)
    axs[0].imshow(I1)
    axs[0].scatter(pixels1[:, 0], pixels1[:, 1], c=colors)
    axs[1].imshow(I2)
    lines = np.stack([pts[:, :2], pts[:, 2:]], axis=1)
    collection=collections.LineCollection(lines, colors=colors)
    axs[1].add_collection(collection)
    plt.show()


def fundamental_matrix(P1, P2, K1, K2):
    R1 = P1[:, 0:3]
    R2 = P2[:, 0:3]
    t1 = P1[:, 3]
    t2 = P2[:, 3]

    # Convert to relative rotation and translation:
    R = R2.T @ R1
    t = R1.T @ (t2 - t1)

    # Compute fundamental matrix:
    tx = [[0, -t[2], t[1]],
          [t[2], 0, -t[0]],
          [-t[1], t[0], 0]]
    tx = np.array(tx)
    Y = np.linalg.solve(K2.T, R) @ tx
    F = np.linalg.solve(K1.T, Y.T).T
    return F


def lineToBorderPoints(lines, imageSize):
    nPts = lines.shape[0]
    pts = -np.ones((nPts, 4))
    firstRow = 0.5
    firstCol = 0.5

    # The border of the image is defined as
    #   row = -0.5
    #   row = imageSize(1) - 0.5
    #   col = -0.5
    #   col = imageSize(2) - 0.5
    lastRow = firstRow + imageSize[0]
    lastCol = firstCol + imageSize[1]

    eps = np.finfo(np.float32).eps

    # Loop through all lines and compute the intersection points of the lines
    # and the image border.
    for iLine in range(nPts):
        a = lines[iLine, 1]
        b = lines[iLine, 0]
        c = lines[iLine, 2]

        endPoints = np.zeros((4))
        iPoint = 0
        # Check for the intersections with the left and right image borders
        # unless the line is vertical.
        if abs(a) > eps:
            # Compute and check the intersection of the line and the left image
            # border. 
            row = - (b * firstCol + c) / a
            if row>=firstRow and row<=lastRow:
                endPoints[iPoint:iPoint+2] = np.array([row, firstCol])
                iPoint = iPoint + 2

            # Compute and check the intersection of the line and the right image
            # border. 
            row = - (b * lastCol + c) / a
            if row>=firstRow and row<=lastRow:
                endPoints[iPoint:iPoint+2] = np.array([row, lastCol])
                iPoint = iPoint + 2

        # Check for the intersections with the top and bottom image borders
        # unless the line is horizontal.
        if abs(b) > eps:
            # If we have not found two intersection points, compute and check the
            # intersection of the line and the top image border. 
            if iPoint < 3:
                col = - (a * firstRow + c) / b
                if col>=firstCol and col<=lastCol:
                    endPoints[iPoint:iPoint+2] = np.array([firstRow, col])
                    iPoint = iPoint + 2

            # If we have not found two intersection points, compute and check the
            # intersection of the line and the bottom image border. 
            if iPoint < 3:
                col = - (a * lastRow + c) / b
                if col>=firstCol and col<=lastCol:
                    endPoints[iPoint:iPoint+2] = np.array([lastRow, col])
                    iPoint = iPoint + 2

        # If the line does not intersect with the image border, set the
        # intersection to -1; 
        for i in range(iPoint, 4):
            endPoints[i] = -1

        pts[iLine, :] = endPoints[[1,0,3,2]]
      
    return pts


def get_n_colors(n):
    colors = [mcolors.to_rgba(c)
                  for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    colors = colors * math.ceil(n / len(colors))
    return colors[:n]
