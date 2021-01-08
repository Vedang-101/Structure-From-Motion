from FeatureMatch import *
from bundleAdjustments import *

point_cloud = np.zeros((0,3))

out_cloud_dir = os.path.join(out_dir, 'point-clouds')
out_err_dir = os.path.join(out_dir, 'errors')

image_data, matches_data, errors = {}, {}, {}

def GetAlignedMatches(kp1,desc1,kp2,desc2,matches):
    img1idx = np.array([m.queryIdx for m in matches])
    img2idx = np.array([m.trainIdx for m in matches])

    #filtering out the keypoints that were matched. 
    kp1_ = (np.array(kp1))[img1idx]
    kp2_ = (np.array(kp2))[img2idx]

    #retreiving the image coordinates of matched keypoints
    img1pts = np.array([kp.pt for kp in kp1_])
    img2pts = np.array([kp.pt for kp in kp2_])

    return img1pts, img2pts, img1idx, img2idx

def InitialPoseEstimation(name1, name2):
    kp1, desc1 = LoadFeatures(name1)
    kp2, desc2 = LoadFeatures(name2)

    matches = LoadMatches(name1, name2)
    matches = sorted(matches, key = lambda x:x.distance)

    img1pts, img2pts, img1idx, img2idx = GetAlignedMatches(kp1,desc1,kp2,desc2,matches)
    F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=cv2.FM_RANSAC,ransacReprojThreshold=.9,confidence=.9)
    mask = mask.astype(bool).flatten()
    
    E = K.T.dot(F.dot(K))
    _,R,t,_ = cv2.recoverPose(E,img1pts[mask],img2pts[mask],K)

    image_data[name1] = [np.eye(3,3), np.zeros((3,1)), np.ones((len(kp1),))*-1]
    image_data[name2] = [R,t,np.ones((len(kp2),))*-1]

    matches_data[(name1,name2)] = [matches, img1pts[mask], img2pts[mask], img1idx[mask], img2idx[mask]]

    return R,t

def TriangulateLinear(img1pts, img2pts, R1, t1, R2, t2):
    img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:,0,:]
    img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:,0,:]
    
    img1ptsNorm = (np.linalg.inv(K).dot(img1ptsHom.T)).T
    img2ptsNorm = (np.linalg.inv(K).dot(img2ptsHom.T)).T

    img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
    img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]
    
    pts4d = cv2.triangulatePoints(np.hstack((R1,t1)),np.hstack((R2,t2)),img1ptsNorm.T,img2ptsNorm.T)
    pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]

    return pts3d

def Update3DReference(ref1, ref2, img1idx, img2idx, upp_limit, low_limit=0):
    ref1[img1idx] = np.arange(upp_limit) + low_limit
    ref2[img2idx] = np.arange(upp_limit) + low_limit

    return ref1, ref2

def TriangulateTwoViews(name1, name2):
    R1, t1, ref1 = image_data[name1]
    R2, t2, ref2 = image_data[name2]
    
    _, img1pts, img2pts, img1idx, img2idx = matches_data[(name1,name2)]

    new_point_cloud = TriangulateLinear(img1pts, img2pts, R1, t1, R2, t2)
    global point_cloud
    point_cloud = np.concatenate((point_cloud, new_point_cloud), axis=0)

    ref1, ref2 = Update3DReference(ref1, ref2, img1idx, img2idx,new_point_cloud.shape[0],point_cloud.shape[0]-new_point_cloud.shape[0])

    image_data[name1][-1] = ref1
    image_data[name2][-1] = ref2

def pts2ply(pts,colors,filename='out.ply'): 
    #Saves an ndarray of 3D coordinates

    with open(filename,'w') as f: 
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(pts.shape[0]))
        
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        
        f.write('end_header\n')
        
        colors = colors.astype(int)
        for pt, cl in zip(pts,colors): 
            f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],
                                                cl[0],cl[1],cl[2]))

def ToPly(filename):
    colors = np.zeros_like(point_cloud)

    for k in image_data.keys():
        _, _, ref = image_data[k]
        kp, desc = LoadFeatures(k)
        kp = np.array(kp)[ref>=0]
        image_pts = np.array([_kp.pt for _kp in kp])
        
        image = cv2.imread(os.path.join(data_dir, k+'.jpg'))[:,:,::-1]

        colors[ref[ref>=0].astype(int)] = image[image_pts[:,1].astype(int),image_pts[:,0].astype(int)]
    
    pts2ply(point_cloud, colors, filename)

def DrawCorrespondences(img, ptsTrue, ptsReproj, ax, drawOnly=50): 
    ax.imshow(img)
    
    randidx = np.random.choice(ptsTrue.shape[0],size=(drawOnly,),replace=False)
    ptsTrue_, ptsReproj_ = ptsTrue[randidx], ptsReproj[randidx]
    
    colors = colors=np.random.rand(drawOnly,3)
    
    ax.scatter(ptsTrue_[:,0],ptsTrue_[:,1],marker='x',c='r',linewidths=.1, label='Ground Truths')
    ax.scatter(ptsReproj_[:,0],ptsReproj_[:,1],marker='x',c='b',linewidths=.1, label='Reprojected')
    ax.legend()

    return ax

def ComputeReprojectionError(name):
    R, t, ref = image_data[name]
    X = point_cloud[ref[ref>0].astype(int)]
    outh = K.dot(R.dot(X.T) + t )
    reproj_pts = cv2.convertPointsFromHomogeneous(outh.T)[:,0,:]

    kp, desc = LoadFeatures(name)
    img_pts = np.array([kp_.pt for i, kp_ in enumerate(kp) if ref[i] > 0])
    err = np.mean(np.sqrt(np.sum((img_pts-reproj_pts)**2,axis=-1)))

    #Save reprojection visual file
    fig,ax = plt.subplots()
    image = cv2.imread(os.path.join(data_dir, name+'.jpg'))[:,:,::-1]
    ax = DrawCorrespondences(image, img_pts, reproj_pts, ax)
    ax.set_title('reprojection error = {}'.format(err))

    fig.savefig(os.path.join(out_err_dir, '{}.png'.format(name)))
    plt.close(fig)
    
    return err

def Find2D3DMatches(name):         
    matcher_temp = cv2.BFMatcher()
    image_names = [x.split('.')[0] for x in sorted(os.listdir(data_dir)) if x.split('.')[-1] in extension]
    kps, descs = [], []
    for n in image_names: 
        if n in image_data.keys():
            kp, desc = LoadFeatures(n)

            kps.append(kp)
            descs.append(desc)
    
    matcher_temp.add(descs)
    matcher_temp.train()

    kp, desc = LoadFeatures(name)

    matches_2d3d = matcher_temp.match(queryDescriptors=desc)

    #retrieving 2d and 3d points
    pts3d, pts2d = np.zeros((0,3)), np.zeros((0,2))
    for m in matches_2d3d: 
        train_img_idx, desc_idx, new_img_idx = m.imgIdx, m.trainIdx, m.queryIdx
        point_cloud_idx = image_data[image_names[train_img_idx]][-1][desc_idx]
        
        #if the match corresponds to a point in 3d point cloud
        if point_cloud_idx >= 0: 
            new_pt = point_cloud[int(point_cloud_idx)]
            pts3d = np.concatenate((pts3d, new_pt[np.newaxis]),axis=0)

            new_pt = np.array(kp[int(new_img_idx)].pt)
            pts2d = np.concatenate((pts2d, new_pt[np.newaxis]),axis=0)

    return pts3d, pts2d, len(kp)

def NewViewPoseEstimation(name):
    pts3d, pts2d, ref_len = Find2D3DMatches(name)
    _, R, t, _ = cv2.solvePnPRansac(pts3d[:,np.newaxis],pts2d[:,np.newaxis],K,None,confidence=.99,flags=cv2.SOLVEPNP_DLS,reprojectionError=8.)
    R,_=cv2.Rodrigues(R)
    image_data[name] = [R,t,np.ones((ref_len,))*-1]

def TriangulateNewView(name):
    for prev_name in image_data.keys(): 
        if prev_name != name: 
            kp1, desc1 = LoadFeatures(prev_name)
            kp2, desc2 = LoadFeatures(name)  

            prev_name_ref = image_data[prev_name][-1]
            matches = LoadMatches(prev_name,name)
            matches = [match for match in matches if prev_name_ref[match.queryIdx] < 0]

            if len(matches) > 0: 
                matches = sorted(matches, key = lambda x:x.distance)

                img1pts, img2pts, img1idx, img2idx = GetAlignedMatches(kp1,desc1,kp2,desc2,matches)
                
                F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=cv2.FM_RANSAC,ransacReprojThreshold=.9,confidence=.9)
                mask = mask.astype(bool).flatten()

                matches_data[(prev_name,name)] = [matches, img1pts[mask], img2pts[mask],img1idx[mask],img2idx[mask]]
                TriangulateTwoViews(prev_name, name)

                BundleAdjusment(prev_name, name)

            else: 
                print('skipping {} and {}'.format(prev_name, name))

def BundleAdjusment(name1, name2):
    img = cv2.imread(name2, 0)
    _, _, img2pts, _, _ = matches_data[(name1,name2)]
    img2pts = np.transpose(img2pts)
    P2 = np.hstack((image_data[name2][0], image_data[name2][1]))
    bundle_adjustment(point_cloud, img2pts, img, P2)

def SFM(input_dir = 'data/fountain-P11/images/', output_dir = 'data/fountain-P11/'):
    global data_dir, out_dir
    data_dir = input_dir
    out_dir = output_dir
    
    global K
    K = findCalibrationMat()
    
    if not os.path.exists(out_cloud_dir): 
        os.makedirs(out_cloud_dir)

    if not os.path.exists(out_err_dir): 
        os.makedirs(out_err_dir)
        
    image_names = [x.split('.')[0] for x in sorted(os.listdir(data_dir)) if x.split('.')[-1] in extension]
    name1, name2 = image_names[0], image_names[1]

    total_time, errors = 0, []
    t1 = time()
    InitialPoseEstimation(name1, name2)
    t2 = time()
    this_time = t2-t1
    total_time += this_time
    print ('Initial Pair Cameras {0}, {1}: Pose Estimation [time={2:.3}s]'.format(name1, name2,this_time))
    
    TriangulateTwoViews(name1, name2)
    t1 = time()
    this_time = t1-t2
    total_time += this_time
    print ('Initial Pair Cameras {0}, {1}: Triangulation [time={2:.3}s]'.format(name1, name2, this_time))

    views_done = 2
    #3d point cloud generation and reprojection error evaluation
    ToPly(os.path.join(out_cloud_dir, 'cloud_{}_view.ply'.format(views_done)))

    BundleAdjusment(name1, name2)

    err1 = ComputeReprojectionError(name1)
    err2 = ComputeReprojectionError(name2)
    errors.append(err1)
    errors.append(err2)

    print ('Camera {}: Reprojection Error = {}'.format(name1, err1))
    print ('Camera {}: Reprojection Error = {}'.format(name2, err2))

    for new_name in image_names[2:]:
        #new camera registration
        t1 = time()
        NewViewPoseEstimation(new_name)
        t2 = time()
        this_time = t2-t1
        total_time += this_time
        print ('Camera {0}: Pose Estimation [time={1:.3}s]'.format(new_name, this_time))
        
        #triangulation for new registered camera
        TriangulateNewView(new_name)
        t1 = time()
        this_time = t1-t2
        total_time += this_time
        print ('Camera {0}: Triangulation [time={1:.3}s]'.format(new_name, this_time))

        #3d point cloud update and error for new camera
        views_done += 1
        ToPly(os.path.join(out_cloud_dir, 'cloud_{}_view.ply'.format(views_done)))

        new_err = ComputeReprojectionError(new_name)
        errors.append(new_err)
        print ('Camera {}: Reprojection Error = {}'.format(new_name, new_err))
    
    mean_error = sum(errors) / float(len(errors))
    print ('Reconstruction Completed: Mean Reprojection Error = {1} [t={0:.6}s]'.format(total_time, mean_error))

if __name__=='__main__':
    SFM()