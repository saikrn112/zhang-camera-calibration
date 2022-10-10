import numpy as np
import cv2
import argparse
import glob
from scipy.spatial.transform import Rotation as scipyRot
from scipy.optimize import least_squares

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def convert_A_to_vector(A):
    return np.array([A[0,0],A[0,1],A[1,1],A[0,2],A[1,2]])

def convert_A_vector_to_matrix(a):
    alpha,gamma,beta,u0,v0 = a
    A1 = [alpha,gamma,u0]
    A2 = [  0  ,beta ,v0]
    A3 = [  0  ,  0  , 1]

    A  = np.vstack((A1,A2,A3)) # 3 x 3
    return A


def get_images(base_path,input_extn):
    img_files = glob.glob(f"{base_path}/*{input_extn}",recursive=False)
    img_names = [img_file.replace(f"{base_path}/",'').replace(f"{input_extn}",'') for img_file in img_files]
    imgs = [cv2.imread(img_file) for img_file in img_files]
    return imgs,img_names

def get_chessboard_corners(img_color,pattern_size,name,args):
    if args.debug:
        cv2.imshow(name,img_color)

    img_gray = cv2.cvtColor(img_color,cv2.COLOR_RGB2GRAY)
    if args.debug:
        cv2.imshow(f"{name}_gray",img_gray)

    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH     \
                        + cv2.CALIB_CB_NORMALIZE_IMAGE  \
                        + cv2.CALIB_CB_FAST_CHECK       
    ret, corners = cv2.findChessboardCorners(img_gray,pattern_size,flags=chessboard_flags)
    # TODO cv2.cornerSubPix
    if not ret:
        print(f"something went wrong while processing {name}")
        exit(1)

    if args.display:
        chessboard_img = cv2.drawChessboardCorners(img_color,pattern_size,corners,ret)
        cv2.imshow(f"{name}_chessboard",chessboard_img)

    corners = corners.reshape((corners.shape[0],-1))
    return corners

def get_world_corners(pattern_size,square_size):
    """
    description:
        returns world corners for a given pattern size and square size(mm)
    input:
        pattern_size - tuple (2)
        square_size - scalar (mm)
    output:
        world_corners - pattern_size[0]*pattern_size[1] x 2
    """
    x_lin = np.arange(0,pattern_size[0],1)
    y_lin = np.arange(0,pattern_size[1],1)
    x_grid, y_grid = np.meshgrid(x_lin,y_lin)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    world_corners = np.vstack((x_grid,y_grid)).T
    world_corners = world_corners*square_size
    return world_corners

def get_V_mat_element(H,i,j):
    """
    description:
        calculate element of v vector from homography
    input:
        H - homography matrix 3 x 3
        i - according the paper convention
        j - according the paper convention
    output:
        v - 6 x 1 according to the paper
    """
    # convering indices from paper convention to numpy convention
    i = i - 1
    j = j - 1

    # calculation vector v for a given homography
    v1 = H[0][i]*H[0][j]
    v2 = H[0][i]*H[1][j] + H[1][i]*H[0][j]
    v3 = H[1][i]*H[1][j]                  
    v4 = H[2][i]*H[0][j] + H[0][i]*H[2][j]
    v5 = H[2][i]*H[1][j] + H[1][i]*H[2][j]
    v6 = H[2][i]*H[2][j]
    v  = np.vstack((v1,v2,v3,v4,v5,v6))
    return v

def get_V_mat(H):
    """
    description:
        calculate V for a given homography
    input:
        H - homography matrix 3 x 3
    output:
        V - 2 x 6 V matrix according to the paper
    """
    V1  = get_V_mat_element(H,1,2) # 6 x 1
    V1  = V1.T # 1 x 6
    V20 = get_V_mat_element(H,1,1) # 6 x 1
    V20 = V20.T # 1 x 6
    V21 = get_V_mat_element(H,2,2) # 6 x 1
    V21 = V21.T # 1 x 6
    V2  = V20 - V21 # 1 x 6
    V   = np.vstack((V1,V2)) # 2 x 6

    return V
def get_L_mat(img_corner,world_corner):
    """
    description:
        calculate L for a given img_corner and world_corner
    input:
        image_corner  - 2,
        world_corners - 3,
    output:
        L - as per paper convention 2 x 9
    """
    L1 = np.hstack((world_corner, np.zeros((3)), -img_corner[0]*world_corner))
    L2 = np.hstack((np.zeros((3)), world_corner, -img_corner[1]*world_corner))
    L  = np.vstack((L1,L2))
    return L

def get_homography(img_corners,world_corners,name):
    world_corners = np.hstack((world_corners,np.ones((world_corners.shape[0],1))))

    L = tuple([get_L_mat(img_corner,world_corner) for img_corner,world_corner in zip(img_corners,world_corners)]) # 2 x 9
    L = np.vstack(L) # 2*N x 9

    eig_val,eig_vec = np.linalg.eig(L.T @ L)
    min_eig_vec_ind = np.argmin(eig_val) # 1 x 1
    min_eig_vec     = eig_vec[:,min_eig_vec_ind] # 6 x 1

    h1 = min_eig_vec[0:3]
    h2 = min_eig_vec[3:6]
    h3 = min_eig_vec[6:9]

    H = np.vstack((h1,h2,h3))
    H = H/H[2,2]
    # TODO optimize using LV MINPACK

#   H, _ = cv2.findHomography(img_corners,world_corners)
#    if H is None:
#        print(f"something went wrong while processing homography for {name}")
#        exit(1)
    return H

def get_camera_intrinsic_from_b(b):
    """
    description:
        return camera intrinsics given b vector from paper
    input:
        b - vector as per convention from paper
    output:
        camera intrinsic matrix 3 x 3
            | alpha gamma u0 |
        A - |   0   beta  v0 |
            |   0     0    1 |
    """
    # TODO #1 could the divisions below fail?

    B11 = b[0] 
    B12 = b[1] 
    B22 = b[2] 
    B13 = b[3] 
    B23 = b[4] 
    B33 = b[5] 

    v0_num = B12*B13 - B11*B23
    v0_den = B11*B22 - B12*B12
    v0     = v0_num/v0_den #TODO #1

    lamda1_num  =  B13*B13 + v0*(B12*B13 - B11*B23)
    lamda = B33 - lamda1_num/B11 #TODO #1

    alpha = (lamda/B11)**(0.5) #TODO #1

    beta_num = lamda*B11
    beta_den = B11*B22 - B12*B12
    beta  = (beta_num/beta_den)**(0.5) #TODO #1

    gamma = (-B12*alpha*alpha*beta)/lamda

    u00 = (gamma*v0)/beta
    u01 = (B13*alpha*alpha)/lamda
    u0  = u00 - u01

    A  = convert_A_vector_to_matrix([alpha,gamma,beta,u0,v0])
    return A, lamda
    

def get_camera_intrinsics(homography_list):
    """
    description:
        calculate camera intrinsics based on the paper 3.1
    input:
        homography_list - list of size N homography matrices 3 x 3
    output:
        A - camera intrinsic matrix 3 x 3
    """
    V = tuple([get_V_mat(H) for H in homography_list]) # N, 2 x 6
    V = np.vstack(V) # (2*N) x 6
    M = V.T @ V # 6 x 6
    U,sigma,R = np.linalg.svd(V)
    eig_val,eig_vec = np.linalg.eig(V.T @ V) # 6 x 6
    min_eig_vec_ind = np.argmin(eig_val) # 1 x 1
    min_eig_vec     = eig_vec[:,min_eig_vec_ind] # 6 x 1
    #print(f"eig_vec:{min_eig_vec}")
    A = get_camera_intrinsic_from_b(min_eig_vec) # 3 x 3
    return A


def test_homography(imgs,imgs_names,homography_list):
    for i,H in enumerate(homography_list):
        img1_warp = cv2.warpPerspective(imgs[i],H,[imgs[i].shape[1],imgs[i].shape[0]])
        cv2.imshow(f"warp1_{imgs_names[i]}",img1_warp)


def get_transformation_mat(A,lamda,H):
    """
    description:
        calculate rotation and translation matrices for each image
    input:
    output:
    """
    A_inv = np.linalg.inv(A) # should be perfectly invertible
    lamda1 = 1/np.linalg.norm(A_inv @ H[:,0],ord=2)
    lamda2 = 1/np.linalg.norm(A_inv @ H[:,1],ord=2)

    r1 = lamda1*A_inv @ H[:,0] # 3 x 1
    r2 = lamda1*A_inv @ H[:,1] # 3 x 1
    r3 = skew(r1) @ r2 # 3 x 1

    t  = lamda1*A_inv @ H[:,2] # 3 x 1

    R  = np.vstack((r1,r2,r3)).T
    r  = scipyRot.from_matrix(R).as_mrp() # 3,

    rt = np.concatenate((r,t.flatten()),axis=0).tolist() # 6,

    return rt

def projection_error(x,A,img_corners,world_corners):
    """
    description:
        computes projection error for an image
    input:
        x - 6, vector of all parameters
        img_corners - M x 2 
        world_corners - M x 2
    output:
        residuals - 14,1
    """
    R  = scipyRot.from_mrp(x[0:3]).as_matrix() # 3 x 3
    t  = x[3:6].reshape((3,1)) # 3 x 1
    T  = np.hstack((R,t)) # 3 x 4
    
    M = world_corners.shape[0]
    zeros = np.zeros((M,1))
    ones  = np.ones((M,1))
    world_corners = np.hstack((world_corners,zeros,ones)).T # 4 x M
    img_corners = np.hstack((img_corners,ones)).T # 3 x M

    m_hat = A @ T @ world_corners # 3 x 3 @ 3 x 4 @ 4 x M =  3 x M
    m_hat = m_hat/m_hat[2]

    error = img_corners - m_hat # 3 x M
    error = np.sum(error,axis=0) # M,
    return error 

def compute_residuals(x,imgs_corners,world_corners):
    """
    description: 
        callable functional to calcuate residuals
    input:
        if N - number of images
            M - number of features per image
            nP - number of parameters required for transformation
                = 3(rotation rodrigues) + 3(translation)
        x - 5(intrinsics) + N*nP
        imgs_corners - N x M x 2
        world_corners - M x 2
    output:
        residuals - N*M*nP
    """

    n_imgs = len(imgs_corners)
    n_feats = len(world_corners)

    A = convert_A_vector_to_matrix(x[0:5]) 

    transformation_params = x[5:] # N*M*nP

    x = transformation_params.reshape((n_imgs,6)) # N*M x 6

    errors= [] # N
    for i in range(x.shape[0]):
        error = projection_error(x[i,:],A,imgs_corners[i],world_corners) # M,
        errors.append(error)

    errors = np.concatenate(errors) # N*M,
    return errors

def main(args):
    # parameters
    base_path = args.basePath
    input_extn = ".jpg"
    square_size = 21.5 #mm
    pattern_size = (9,6)

    imgs,imgs_names = get_images(base_path,input_extn)

    world_corners = get_world_corners(pattern_size,square_size)

    imgs_corners = [get_chessboard_corners(img,pattern_size,name,args) for img,name in zip(imgs,imgs_names)]

    homography_list = [get_homography(img_corners,world_corners,name) for img_corners,name in zip(imgs_corners,imgs_names)]

    A_estimate, lamda_estimate = get_camera_intrinsics(homography_list)
    print(f"K/A:\n{A_estimate}")

    transformations = tuple([get_transformation_mat(A_estimate,lamda_estimate,H) for H in homography_list]) # 13*6,
    transformations = np.concatenate(transformations)

    a  = convert_A_to_vector(A_estimate) # 5,
    x0 = np.concatenate((a,transformations)) # 83,

    kwargs1 = {"imgs_corners":imgs_corners, "world_corners":world_corners}
    
    result = least_squares(compute_residuals,x0=x0,method='lm',kwargs=kwargs1)

    print(convert_A_vector_to_matrix(result.x[0:5]))


    if args.debug:
        test_homography(imgs,imgs_names,homography_list)

    if args.display:
        cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basePath',default='../data/Calibration_Imgs')
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display debug information")

    args = parser.parse_args()
    main(args)
