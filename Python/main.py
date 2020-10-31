import cv2
import numpy as np
import math
from PLY import PLY

def KeyPointsToPoints(keypoints):
    out = []
    for kp in keypoints:
        out.append([[kp.pt[0], kp.pt[1]]])
    res = np.array(out, dtype=np.float32)
    return res

def PointMatchingSURF(img1, img2):
    #Matches using SURF
    surf = cv2.xfeatures2d.SURF_create()
    keypoints1, descriptors1 = surf.detectAndCompute(img1, None)
    keypoints2, descriptors2 = surf.detectAndCompute(img2, None)
    img4 = cv2.drawKeypoints(img1, keypoints1, None)
    cv2.imwrite("KP.jpg", img4)
    bf = cv2.BFMatcher_create()
    matches = bf.match(descriptors1, descriptors2)
    img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:200], None, flags=2)
    cv2.imwrite("Out2.jpg", img3)
    return keypoints1, keypoints2,matches

def PointMatchingOpticalFlow(img1, img2):
    #matches using optical flow
    ffd = cv2.FastFeatureDetector_create()
    left_keypoints = ffd.detect(img1, None)
    right_keypoints = ffd.detect(img2, None)

    left_points = KeyPointsToPoints(left_keypoints)
    right_points = np.zeros_like(left_points)

    #Checking if images are in greyscale
    prevgray = img1
    gray = img2
    if len(img1.shape) == 3:
        prevgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    right_points, vstatus, verror = cv2.calcOpticalFlowPyrLK(prevgray,gray,left_points,right_points)

    #Filterout points with high error
    right_points_to_find = []
    right_points_to_find_back_index = []
    for i in range(0, len(vstatus)):
        if(vstatus[i] and verror[i] < 12.0):
            right_points_to_find_back_index.append(i)
            right_points_to_find.append(right_points[i])
        else:
            vstatus[i] = 0

    right_points_to_find_flat = np.array(right_points_to_find).reshape(1, len(right_points_to_find), 2)

    right_features = KeyPointsToPoints(right_keypoints)
    right_features_flat = right_features.reshape(1, len(right_features), 2)

    #Look around each of point in right image for any features that were detected in its area and make a match
    #matcher = cv2.BFMatcher_create()
    nearest_neighbours = cv2.BFMatcher().radiusMatch(right_features_flat, right_points_to_find_flat, 2.0)

def findCalibrationMat(img):
    px = img.shape[1] / 2
    py = img.shape[0] / 2

    # For Fountain dataset
    # 2759.48 0 1520.69 
    # 0 2764.16 1006.81 
    # 0 0 1
    K = np.float32([[2759.48, 0, 1520.69],
                    [0, 12764.16, 1006.81],
                    [0, 0, 1]])
    return K

def FindEssentialMat(kp1, kp2, matches, K):
    imgpts1 = []
    imgpts2 = []
    for i in range(0, len(matches)):
        imgpts1.append([[kp1[matches[i].queryIdx].pt[0], kp1[matches[i].queryIdx].pt[1]]])
        imgpts2.append([[kp2[matches[i].trainIdx].pt[0], kp2[matches[i].trainIdx].pt[1]]])

    F = cv2.findFundamentalMat(np.array(imgpts1, dtype=np.float32), np.array(imgpts2, dtype=np.float32), method=cv2.FM_RANSAC, ransacReprojThreshold=0.1, confidence=0.99)
    E = np.matmul(np.matmul(np.transpose(K), F[0]), K)
    return E

def checkCoherentRotation(R):
    if(math.fabs(np.linalg.det(R))-1.0 > 1e-07):
        print("Not a coherent rotational Matrix")
        return False
    else:
        print("Coherent Rotational Matrix found")
        return True

def FindPMat(E):
    _, u, vt = cv2.SVDecomp(E, flags=cv2.SVD_MODIFY_A)
    W = np.float32([[0,-1,0],
                    [1,0,0],
                    [0,0,1]])
    R = np.matmul(np.matmul(u, W), vt)
    t = u[:, 2]
    
    P = np.float32([])
    if checkCoherentRotation(R):
        P = np.float32([[R[0][0], R[0][1], R[0][2], t[0]],
                        [R[1][0], R[1][1], R[1][2], t[1]],
                        [R[2][0], R[2][1], R[2][2], t[2]]])
    else:
        P = None
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

def TraingulatePoints(pt_set1, pt_set2, matches, K, P, P1, img1, ply):
    Kinv = np.linalg.inv(K)
    #reproj_error = []
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

        # #Calculate reprojection error
        # xPt_img = np.matmul(np.matmul(K, P1), X)
        # xPt_img_ = np.float32([[xPt_img[0]/xPt_img[2], xPt_img[1]/xPt_img[2]]])
        # reproj_error.append(np.linalg.norm(xPt_img_-kp1))
        
        #print(kp[0], kp[1])
        bgr = img1[int(kp[1]),int(kp[0])]

        #x = X[0] y = Y[1] z = X[2]
        ply.append([X[0],X[1],X[2],bgr[0],bgr[1],bgr[2]])
    #me = np.mean(reproj_error)
    return ply

def main():
    img1 = cv2.imread("../Resources/Images/0000.jpg")
    img2 = cv2.imread("../Resources/Images/0000.jpg")

    kp1, kp2, matches = PointMatchingSURF(img1, img2)
    K = findCalibrationMat(img1)
    E = FindEssentialMat(kp1, kp2, matches, K)

    P0 = np.float32([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0]])

    P1 = FindPMat(E)
    print(P1)

    ply = []
    ply = TraingulatePoints(kp1, kp2, matches, K, P0, P1, img1, ply)
    #print("Mean Error = ", error)

    out = PLY("Output/")
    out.insert_header(len(ply), "Result7")
    for i in range(0,len(ply)):
        out.insert_point(ply[i][0],ply[i][1],ply[i][2],ply[i][3],ply[i][4],ply[i][5])

    #cv2.waitKey()
    cv2.destroyAllWindows()

main()