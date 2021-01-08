import cv2 
import numpy as np 
import pickle 
import os 
from time import time
import matplotlib.pyplot as plt

data_dir = ''
out_dir = ''

extension = 'jpg,png'
features = 'SURF'
matcher = 'BFMatcher'

K = None

def findCalibrationMat():
    with open(os.path.join(data_dir,'K.txt')) as f:
        lines = f.readlines()
    return np.array(
        [l.strip().split(' ') for l in lines],
        dtype=np.float32
    )

def SerializeKeypoints(kp):
    out = []
    for kp_ in kp: 
        temp = (kp_.pt, kp_.size, kp_.angle, kp_.response, kp_.octave, kp_.class_id)
        out.append(temp)

    return out

def DeserializeKeypoints(kp):
    out = []
    for point in kp:
        temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2],
         _response=point[3], _octave=point[4], _class_id=point[5]) 
        out.append(temp)

    return out

def SerializeMatches(matches):
    out = []
    for match in matches: 
        matchTemp = (match.queryIdx, match.trainIdx, match.imgIdx, match.distance) 
        out.append(matchTemp)
    return out

def DeserializeMatches(matches):
    out = []
    for match in matches:
        out.append(cv2.DMatch(match[0],match[1],match[2],match[3])) 
    return out

def FeatureMatch(input_dir = 'data/fountain-P11/images/', output_dir = 'data/fountain-P11/'):
    global data_dir, out_dir
    data_dir = input_dir
    out_dir = output_dir
    
    global K
    K = findCalibrationMat()

    img_names = sorted(os.listdir(data_dir))
    img_paths = [os.path.join(data_dir, x) for x in img_names if x.split('.')[-1] in extension]

    feat_out_dir = os.path.join(out_dir,'features', 'SURF')
    matches_out_dir = os.path.join(out_dir,'matches', 'BFMatcher')

    if not os.path.exists(feat_out_dir): 
        os.makedirs(feat_out_dir)
    if not os.path.exists(matches_out_dir): 
        os.makedirs(matches_out_dir)

    data = []
    t1 = time()

    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        #Get image name to save future files
        img_name = img_names[i].split('.')[0]
        #Conver image from BGR to RGB
        img = img[:,:,::-1]

        #Get feature points and descriptors for image_name
        feat = cv2.xfeatures2d.SURF_create()
        kp, desc = feat.detectAndCompute(img,None)
        data.append((img_name, kp, desc))

        #Store keypoints and descriptors
        kp_ = SerializeKeypoints(kp)
        with open(os.path.join(feat_out_dir, 'kp_'+img_name+'.pkl'),'wb') as out:
            pickle.dump(kp_, out)

        with open(os.path.join(feat_out_dir, 'desc_'+img_name+'.pkl'),'wb') as out:
            pickle.dump(desc, out)
        
        t2 = time()
        print('FEATURES DONE: '+str(i+1)+'/'+str(len(img_paths))+' [time={0:.2f}s]'.format(t2-t1))
        t1 = time()

    num_done = 0 
    num_matches = ((len(img_paths)-1) * (len(img_paths))) / 2
    
    t1 = time()
    #Matche keypoints of image_i with image_i+1, image_i+2,...,image_n
    for i in range(len(data)): 
        for j in range(i+1, len(data)): 
            img_name1, kp1, desc1 = data[i]
            img_name2, kp2, desc2 = data[j]

            matcher = cv2.BFMatcher(crossCheck=True)
            matches = matcher.match(desc1,desc2)

            matches = sorted(matches, key = lambda x:x.distance)
            matches_ = SerializeMatches(matches)

            #Store Matches
            pickle_path = os.path.join(matches_out_dir, 'match_'+img_name1+'_'+img_name2+'.pkl')
            with open(pickle_path,'wb') as out:
                pickle.dump(matches_, out)

            num_done += 1 
            t2 = time()

            print('MATCHES DONE: '+str(num_done)+'/'+str(num_matches)+' [time={0:.2f}s]'.format(t2-t1))

            t1 = time()

def LoadFeatures(name):
    feat_out_dir = os.path.join(out_dir,'features', features)
    with open(os.path.join(feat_out_dir,'kp_{}.pkl'.format(name)),'rb') as f:
        kp = pickle.load(f)
    kp = DeserializeKeypoints(kp)

    with open(os.path.join(feat_out_dir,'desc_{}.pkl'.format(name)),'rb') as f:
        desc = pickle.load(f)

    return kp, desc

def LoadMatches(name1, name2):
    matches_out_dir = os.path.join(out_dir,'matches', matcher)
    with open(os.path.join(matches_out_dir,'match_{}_{}.pkl'.format(name1,name2)),'rb') as f: 
        matches = pickle.load(f)
    matches = DeserializeMatches(matches)
    return matches

if __name__=='__main__':
    FeatureMatch()