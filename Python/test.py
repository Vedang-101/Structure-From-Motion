import cv2
import numpy as np
import math
import sys
from PLY import PLY

def detect_and_match_feature(img1, img2):
    """
    img1, img2: are input images
    The following outputs are needed:
    kp1, kp2: keypoints (here sift keypoints) of the two images
    matches_good: matches which pass the ratio test
    p1, p2: only the 2d points in the respective images
    pass ratio test. These points should correspond to each other.

    Steps:
    1. Compute sift descriptors.
    2. Match sift across two images.
    3. Use ratio test to get good matches.
    4. Store points retrieved from the good matches.

    : See SIFT_create
    For feature matching you could use
    - BruteForceMatcher
    (https://docs.opencv.org/3.4/d3/da1/classcv_1_1BFMatcher.html)
    - FLANN Matcher:
    (https://docs.opencv.org/3.4/dc/de2/classcv_1_1FlannBasedMatcher.html)
    """
    #Creating keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    #Brute force matching with k=2
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    #Ratio test and retrieval of indices
    matches_good = [m1 for m1, m2 in matches if m1.distance < 0.75*m2.distance]
    query_ind = [match.queryIdx for match in matches_good]
    train_ind = [match.trainIdx for match in matches_good]

    #Getting float based points from good matches
    p1 = np.float32([kp1[ind].pt for ind in query_ind])
    p2 = np.float32([kp2[ind].pt for ind in train_ind])

    return p1, p2, matches_good, kp1, kp2

def compute_essential(p1, p2, K):
    """
    p1, p2: only the 2d points in the respective images
    pass ratio test. These points should correspond to each other.
    Outputs:
    Essential Matrix (E), and corresponding (mask)
    used in its computation. The mask contains the inlier_matches
    to compute E

    Hint: findEssentialMat
    """
    #Computing essential matrix using global matching RANSAC approach
    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask

def compute_pose(p1, p2, E):
    """
    p1, p2: only the 2d points in the respective images
    pass ratio test. These points should correspond to each other.
    E: Essential matrix
    Outputs:
    R, trans: Rotation, Translation vectors

    Hint: recoverPose
    """
    #Decomposing essential matrix into rotational and translation vectors
    points, R, trans, mask = cv2.recoverPose(E, p1, p2)

    P = np.float32([[R[0][0], R[0][1], R[0][2], trans[0]],
                    [R[1][0], R[1][1], R[1][2], trans[1]],
                    [R[2][0], R[2][1], R[2][2], trans[2]]])
    return P

def LinearLSTriangulation(u, P, u1, P1):
    A = np.float32([[u[0]*P[2][0]-P[0][0], u[0]*P[2][1]-P[0][1], u[0]*P[2][2]-P[0][2]],
                    [u[1]*P[2][0]-P[1][0], u[1]*P[2][1]-P[1][1], u[1]*P[2][2]-P[1][2]],
                    [u1[0]*P1[2][0]-P1[0][0], u1[0]*P1[2][1]-P1[0][1], u1[0]*P1[2][2]-P[0][2]],
                    [u1[1]*P1[2][0]-P1[1][0], u1[1]*P1[2][1]-P1[1][1], u1[1]*P1[2][2]-P1[1][2]]])
    A = np.reshape(A, [4, 3])
    B = np.float32([[-(u[0]*P[2][3]-P[0][3])],
                    [-(u[1]*P[2][3]-P[1][3])],
                    [-(u1[0]*P1[2][3]-P1[0][3])],
                    [-(u1[1]*P1[2][3]-P1[1][3])]])
    B = np.reshape(B, [4, 1])
    _, X = cv2.solve(A,B,flags=cv2.DECOMP_SVD)
    return X

def TraingulatePoints(pt_set1, pt_set2, matches, K, P, P1, img1, ply, _current = None):
    Kinv = np.linalg.inv(K)
    reproj_error = []
    for i in range(0, len(matches)):
        kp = pt_set1[matches[i].queryIdx].pt
        u = np.float32([[[kp[0]], [kp[1]], [1]]])
        um = np.matmul(Kinv, u)
        u = um[0]

        kp1 = pt_set2[matches[i].trainIdx].pt
        u1 = np.float32([[[kp1[0]], [kp1[1]], [1]]])
        um1 = np.matmul(Kinv, u1)
        u1 = um1[0]

        #Triangulate
        X = LinearLSTriangulation(u, P, u1, P1)

        #Calculate reprojection error
        X1 = [[X[0][0]],
             [X[1][0]],
             [X[2][0]],
             [1]]
        xPt_img = np.matmul(np.matmul(K, P1), X1)
        xPt_img_ = np.float32([[xPt_img[0]/xPt_img[2], xPt_img[1]/xPt_img[2]]])
        reproj_error.append(np.linalg.norm(xPt_img_-kp1))
        
        #print(kp[0], kp[1])
        bgr = img1[int(kp[1]),int(kp[0])]

        if _current is not None:
            _current.add_entry((X[0],X[1],X[2]), (kp1[0], kp1[1]))

        #x = X[0] y = Y[1] z = X[2]
        ply.append([X[0],X[1],X[2],bgr[0],bgr[1],bgr[2]])
    me = np.mean(reproj_error)
    return me, ply

def triangulate(p1, p2, Rt1, Rt2, mask, K):
    """
    p1,p2: Points in the two images which correspond to each other
    R, trans: Rotation and translation matrix.
    mask: is obtained during computation of Essential matrix

    Outputs:
    point_3d: should be of shape (NumPoints, 3). The last dimension
    refers to (x,y,z) co-ordinates

    Hint: triangulatePoints
    """
    #Creating 3x4 matrices for the two cameras by aligning world coordinates with first camera and stacking the R, T vectors for the second camera
    
    #Creating Projection Matrices by multiplying with intrinsic matrix
    M1 = np.dot(K, Rt1)
    M2 = np.dot(K, Rt2)

    print("\nProjection Matrix:\n")
    print(M2)

    #Applying mask to the points to filter the outlier points and get inlier points
    p1_masked = p1[mask.ravel() == 1]
    p2_masked = p2[mask.ravel() == 1]

    #Converting image coordinates to normalized coordinates
    p1_norm = cv2.undistortPoints(p1_masked.reshape(-1, 1, 2), K, None)
    p2_norm = cv2.undistortPoints(p2_masked.reshape(-1, 1, 2), K, None)

    #Triangulating points
    point_4d = cv2.triangulatePoints(M1, M2, np.squeeze(p1_norm).T, np.squeeze(p2_norm).T)

    #Converting homogeneous coordinates to regular coordinates
    point_3d = (point_4d / np.tile(point_4d[-1,:], (4, 1)))[:3,:].T
    
    return point_3d

def findCalibrationMat(img):
    px = img.shape[1] / 2
    py = img.shape[0] / 2

    # For Fountain dataset
    K = np.float32([[2759.48, 0, 1520.69],
                    [0, 12764.16, 1006.81],
                    [0, 0, 1]])

    #For Palace dataset
    # K = np.float32([[2780.1700000000000728, 0, 1539.25],
    #                 [0, 2773.5399999999999636, 1001.2699999999999818],
    #                 [0, 0, 1]])
    
    # K = np.float32([[1000, 0, px],
    #                 [0, 1000, py],
    #                 [0, 0, 1]])
    return K

def PairStructureFromMotion():
    img1 = cv2.imread("../Resources/Fountain/im1.jpg")
    img2 = cv2.imread("../Resources/Fountain/im2.jpg")
    
    p1, p2, matches_good, kp1, kp2 = detect_and_match_feature(img1, img2)
    K = findCalibrationMat(img1)

    E, mask = compute_essential(p1, p2, K)

    P0 = np.float32([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0]])

    P1 = compute_pose(p1, p2, E)


    ply = triangulate(p1,p2,P0,P1,mask,K)
    #me, ply = TraingulatePoints(kp1, kp2, matches_good, K, P0, P1, img1, ply)

    out = PLY("Output/")
    out.insert_header(len(ply), "Fountain")
    for i in range(0,len(ply)):
        out.insert_point([ply[i][0]],[ply[i][1]],[ply[i][2]],255,255,255)

PairStructureFromMotion()