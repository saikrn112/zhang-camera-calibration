import numpy as np
import cv2
import argparse
import glob
def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

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
    if not ret:
        print(f"something went wrong while processing {name}")
        exit(1)

    if args.display:
        chessboard_img = cv2.drawChessboardCorners(img_color,pattern_size,corners,ret)
        cv2.imshow(f"{name}_chessboard",chessboard_img)

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
    v1 = H[i][0]*H[j][0]
    v2 = H[i][0]*H[j][1] + H[i][1]*H[j][0]
    v3 = H[i][1]*H[j][1]                  
    v4 = H[i][2]*H[j][0] + H[i][0]*H[j][2]
    v5 = H[i][2]*H[j][1] + H[i][1]*H[j][2]
    v6 = H[i][2]*H[j][2]                  
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

def get_homography(img_corners,world_corners,name):
    H, _ = cv2.findHomography(img_corners,world_corners)
    if H is None:
        print(f"something went wrong while processing homography for {name}")
        exit(1)
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

    A1 = [alpha,gamma,u0]
    A2 = [  0  ,beta ,v0]
    A3 = [  0  ,  0  , 1]
    A  = np.vstack((A1,A2,A3))
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
    print(f"V:{R}")
    eig_val,eig_vec = np.linalg.eig(V.T @ V) # 6 x 6
    min_eig_vec_ind = np.argmin(eig_val) # 1 x 1
    min_eig_vec     = eig_vec[:,min_eig_vec_ind] # 6 x 1
    print(f"eig_vec:{min_eig_vec}")
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
        homography_list - list of size N homography matrices 3 x 3
    output:
        A - camera intrinsic matrix 3 x 3
    """
    A_inv = np.linalg.inv(A) # should be perfectly invertible
    r1 = lamda*A_inv @ H[:,0] # 3 x 1
    r2 = lamda*A_inv @ H[:,1] # 3 x 1
    r3 = skew(r1) @ r2 # 3 x 1
    t  = lamda*A_inv @ H[:,2] # 3 x 1
    R  = np.vstack((r1,r2,r3)).T
    print(f"R_compute:\n{R}")

    lamda_check1 = 1/np.linalg.norm(A_inv @ H[:,0],ord=2)
    lamda_check2 = 1/np.linalg.norm(A_inv @ H[:,1],ord=2)
    print(f"{lamda_check1}=={lamda_check2}")
    return R,t

def projection_error_functional(x):
    """
    description:
        callable functional for optimizing intrinsics and extrinsincs
    input:
        x - 14, vector of all parameters
    """
    A_elems = x[0:5]
    alpha,gamma,beta,u0,v0 = A_elems
    A1 = [alpha,gamma,u0]
    A2 = [  0  ,beta ,v0]
    A3 = [  0  ,  0  , 1]
    A  = np.vstack((A1,A2,A3))

    r1 = x[5:8]
    r2 = x[8:11]
    t  = x[11:14]




def main(args):
    # parameters
    base_path = args.basePath
    input_extn = ".jpg"
    square_size = 21.5 #mm
    pattern_size = (9,6)

    imgs,imgs_names = get_images(base_path,input_extn)

    world_corners = get_world_corners(pattern_size,square_size)
    #print(f"world_corners:{world_corners}")

    imgs_corners = [get_chessboard_corners(img,pattern_size,name,args) for img,name in zip(imgs,imgs_names)]

    homography_list = [get_homography(img_corners,world_corners,name) for img_corners,name in zip(imgs_corners,imgs_names)]

    A_estimate, lamda_estimate = get_camera_intrinsics(homography_list)
    print(f"K/A:\n{A_estimate}")
    print(f"lamda:\n{lamda_estimate}")

    R,t = get_transformation_mat(A_estimate,lamda_estimate,homography_list[0])
    
    if args.debug:
        R1 = np.hstack((R,np.expand_dims(t,axis=1)))
        print(f"lamda:{lamda_estimate}")
        print(f"R1:{R1}")
        print(f"homo:\n{homography_list[0]}")
        print((A_estimate@R1)/lamda_estimate)

    #scipy.optimize.least_squares(projection_error_functional,


    if args.debug:
        print(f"{imgs_names[0]}")
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
