import cv2
import numpy as np
import math
import os
import glob
from PLY import PLY
from PointCloudTable import PointCloudTable

input_dir = "../Resources/Fountain/"
output_dir = "F:/Projects/Github/Major-Project-2020/Output/abc"
format_img = ".jpg"

def KeyPointsToPoints(keypoints):
    out = []
    for kp in keypoints:
        out.append([[kp.pt[0], kp.pt[1]]])
    res = np.array(out, dtype=np.float32)
    return res

def PointMatchingSURF(img1, img2, save = False, filename1 = None, filename2 = None):
    #Matches using SURF
    surf = cv2.xfeatures2d.SURF_create()
    keypoints1, descriptors1 = surf.detectAndCompute(img1, None)
    keypoints2, descriptors2 = surf.detectAndCompute(img2, None)
    img4 = cv2.drawKeypoints(img1, keypoints1, None)
    cv2.imwrite("KP.jpg", img4)
    bf = cv2.BFMatcher_create()
    matches = bf.match(descriptors1, descriptors2)
    img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:200], None, flags=2)
    if save:
        cv2.imwrite(img1.filename+"_x_"+img2.filename+".jpg", img3)
    return keypoints1, keypoints2,matches

def PointMatchingOpticalFlow(img1, img2, save=False, filename1 = None, filename2 = None):
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

    found_in_imgpts_j = []
    right_points_to_find_flat = np.array(right_points_to_find).reshape(len(right_points_to_find), 2)
    right_features = KeyPointsToPoints(right_keypoints)
    right_features_flat = right_features.reshape(len(right_features), 2)

    #Look around each of point in right image for any features that were detected in its area and make a match
    matcher = cv2.BFMatcher_create(cv2.NORM_L2)
    nearest_neighbours = matcher.radiusMatch(right_points_to_find_flat, right_features_flat, 2.0)#THIS IS THE NEW LINE added(Sarthak)
    #nearest_neighbours = cv2.BFMatcher().radiusMatch(right_features_flat, right_points_to_find_flat, 2.0)
    matches = []
    # print(len(nearest_neighbours))

    for i in range(0, len(nearest_neighbours)):      
        _m = None
        if len(nearest_neighbours[i]) == 1:
            _m = nearest_neighbours[i][0]
        elif len(nearest_neighbours[i]) > 1:
            if (nearest_neighbours[i][0].distance / nearest_neighbours[i][1].distance) < 0.7:
                _m = nearest_neighbours[i][0]
            else:
                #did not pass ratio test
                pass
        else:
            #no match
            pass

        #prevent duplicates
        if _m != None:
            if found_in_imgpts_j.count(_m.trainIdx) == 0:
                #back to original indexing of points for <i_idx>
                _m.queryIdx = right_points_to_find_back_index[_m.queryIdx]
                matches.append(_m)	
                right_points_to_find_back_index.append(_m.trainIdx) #Added this LINE(Sarthak)

    img3 = cv2.drawMatches(img1, left_keypoints, img2, right_keypoints, matches, None)
    if save:
        cv2.imwrite(filename1+"_x_"+filename2+".jpg", img3)
    return left_keypoints, right_keypoints, matches

def findCalibrationMat():
    with open(input_dir+'intrinsic.txt') as f:
        lines = f.readlines()
    return np.array(
        [l.strip().split(' ') for l in lines],
        dtype=np.float32
    )
    # # For Fountain dataset
    # # K = np.float32([[2759.48, 0, 1520.69],
    # #                 [0, 12764.16, 1006.81],
    # #                 [0, 0, 1]])

    # #For Palace dataset
    # # K = np.float32([[2780.1700000000000728, 0, 1539.25],
    # #                 [0, 2773.5399999999999636, 1001.2699999999999818],
    # #                 [0, 0, 1]])
    
    # #For Rubics and Rubics_Vertices
    # K = np.float32([[2666.6667, 0, 960],
    #                 [0, 2250.0000, 540],
    #                 [0, 0, 1]])
    #return K

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
        reproj_error.append(1.0)
        
        #print(kp[0], kp[1])
        bgr = img1[int(kp[1]),int(kp[0])]

        if _current is not None:
            _current.add_entry((X[0],X[1],X[2]), (kp1[0], kp1[1]))

        #x = X[0] y = Y[1] z = X[2]
        ply.append([X[0],X[1],X[2],bgr[0],bgr[1],bgr[2]])
    me = np.mean(reproj_error)
    return me, ply

def find_second_camera_matrix(p1, new_kp, old_kp, matches, current, prev, K):
    found_points2D = []
    found_points3D = []

    kp1 = []
    kp2 = []

    for i in range(0, len(matches)):
        kp1.append(new_kp[matches[i].trainIdx])
        kp2.append(old_kp[matches[i].queryIdx])

    for i in range(len(kp2)):
        found = prev.find_3d(kp2[i].pt)
        if found is not None:
            new_point = (found[0], found[1], found[2])
            new_point2 = (kp1[i].pt[0], kp1[i].pt[1])

            found_points3D.append(new_point)
            found_points2D.append(new_point2)
            #current.add_entry(new_point, new_point2)

    print('Matches found in table: ' + str(len(found_points2D)))

    size = len(found_points3D)

    found3d_points = np.zeros([size, 3], dtype=np.float32)
    found2d_points = np.zeros([size, 2], dtype=np.float32)

    for i in range(size):
        found3d_points[i, 0] = found_points3D[i][0]
        found3d_points[i, 1] = found_points3D[i][1]
        found3d_points[i, 2] = found_points3D[i][2]

        found2d_points[i, 0] = found_points2D[i][0]
        found2d_points[i, 1] = found_points2D[i][1]

    p_tmp = p1.copy()

    r = np.float32(p_tmp[0:3, 0:3])
    t = np.float32(p_tmp[0:3, 3:4])

    r_rog, _ = cv2.Rodrigues(r)

    _dc = np.float32([0, 0, 0, 0])

    _, r_rog, t = cv2.solvePnP(found3d_points, found2d_points, K, _dc, r_rog, t)
    t1 = np.float32(t)

    R1, _ = cv2.Rodrigues(r_rog)

    camera = np.float32([
        [R1[0, 0], R1[0, 1], R1[0, 2], t1[0]],
        [R1[1, 0], R1[1, 1], R1[1, 2], t1[1]],
        [R1[2, 0], R1[2, 1], R1[2, 2], t1[2]]
    ])

    return camera

def PairStructureFromMotion():
    img1 = cv2.imread("../Resources/Fountain/im7.jpg")
    img2 = cv2.imread("../Resources/Fountain/im8.jpg")
    #kp1, kp2, matches = PointMatchingSURF(img1, img2)
    kp1, kp2, matches = PointMatchingOpticalFlow(img1, img2)
    K = findCalibrationMat()
    E = FindEssentialMat(kp1, kp2, matches, K)

    P0 = np.float32([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0]])

    P1 = FindPMat(E)
    print(P1)

    ply = []
    error, ply = TraingulatePoints(kp1, kp2, matches, K, P0, P1, img1, ply)
    print("Mean Error = ", error)

    out = PLY("Pair_Output/")
    out.insert_header(len(ply), "Palace")
    for i in range(0,len(ply)):
        out.insert_point(ply[i][0],ply[i][1],ply[i][2],ply[i][3],ply[i][4],ply[i][5])

    
    cv2.destroyAllWindows()

def MultiViewStructureFromMotion():
    #Create Output directory if does not exists
    try:
	    os.mkdir(output_dir)
    except FileExistsError:
        pass
    out = PLY(output_dir+'/')

    #Number of frames present in the input directory
    number_of_images = len(glob.glob1(input_dir, '*' + format_img))

    current = PointCloudTable()
    prev = PointCloudTable()

    file_number = 0

    picture_number1 = 0
    picture_number2 = 1

    image_name1 = input_dir + 'im0' + format_img
    image_name2 = input_dir + 'im1' + format_img

    frame1 = cv2.imread(image_name1)
    frame2 = cv2.imread(image_name2)
    
    point_cloud = []
    p1 = np.zeros([3, 4], dtype=np.float32)
    p2 = np.zeros([3, 4], dtype=np.float32)

    initial_3d_model = True

    factor = 1
    count = 0

    K = findCalibrationMat()

    while file_number < number_of_images - 1:
        print('Using ' + str(image_name1) + ' and ' + str(image_name2))
        print('Matching...')
        kp1, kp2, matches = PointMatchingOpticalFlow(frame1, frame2, True, "im" + str(picture_number1), "im" + str(picture_number2))

        if len(matches) >= 8:
            if initial_3d_model:
                E = FindEssentialMat(kp1, kp2, matches, K)
                p1 = np.float32([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,0]])
                p2 = FindPMat(E)

                print('Creating initial 3D model...')
                error, point_cloud = TraingulatePoints(kp1, kp2, matches, K, p1, p2, frame1, point_cloud, current)
                print("Mean Error = ", error)
                print('Initial lookup table size is: ' + str(current.table_size()))
                initial_3d_model = False
            else:
                prev.init()
                prev = current.copy()

                current.init()

                print('LookupTable size is: ' + str(prev.table_size()))

                p1 = p2.copy()
                p2 = find_second_camera_matrix(p2, kp2, kp1, matches, current, prev, K)
                
                #print('New table size after adding known 3D points: ' + str(current.table_size()))
                print('Triangulating...')
                error, point_cloud = TraingulatePoints(kp1, kp2, matches, K, p1, p2, frame1, point_cloud, current)
                print("Mean Error = ", error)

            print('Start writing points to file...')
            out.insert_header(len(point_cloud), str(file_number))
            for i in range(len(point_cloud)):
                out.insert_point(point_cloud[i][0], point_cloud[i][1], point_cloud[i][2],
                                 point_cloud[i][3], point_cloud[i][4], point_cloud[i][5])
            point_cloud = []
            file_number += 1

        else:
            print("Not enough matches...")

        picture_number1 = picture_number2 % number_of_images
        picture_number2 = (picture_number2 + factor) % number_of_images
        
        count += 1
        if count % number_of_images == number_of_images - 1:
            picture_number2 += 1
            factor += 1

        image_name1 = input_dir + 'im' + str(picture_number1) + format_img
        image_name2 = input_dir + 'im' + str(picture_number2) + format_img
        frame1 = cv2.imread(image_name1)
        frame2 = cv2.imread(image_name2)

        print("\n\n")
    print("Done")

PairStructureFromMotion()
#MultiViewStructureFromMotion()