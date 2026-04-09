import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main() -> None:
    chessboard_size = (4, 7)
    square_size = 34.0 

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = [] 

    images_dir = PROJECT_ROOT / 'data/combined_calibration_images'
    images = list(images_dir.glob('*.png'))

    if not images:
        print(f"No calibration images found in {images_dir}")
        return

    print(f'Loaded {len(images)} images for calibration.')

    for img_path in images:
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(500)
            
            output_fname = str(img_path).replace('.png', '_corners_detected.png')
            cv2.imwrite(output_fname, img)
        else:
            print(f'Chessboard not found in {img_path.name}')

    cv2.destroyAllWindows()

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)

    calib_out_path = PROJECT_ROOT / 'data/calibration_data.npz'
    np.savez(str(calib_out_path), camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)
    print(f"Calibration saved to {calib_out_path}")

    # Test undistortion on the first available image
    test_img = cv2.imread(str(images[0]))
    h, w = test_img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    undistorted_img = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"Total error: {mean_error / len(objpoints)}")

if __name__ == "__main__":
    main()