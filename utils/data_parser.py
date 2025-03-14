#
# Helpers for parsing the data
# 
# SceneFun3D Toolkit
#

import numpy as np
import cv2
import os
import open3d as o3d
import glob
import imageio
import utils.homogenous as hm
from utils.rigid_interpolation import rigid_interp_split, rigid_interp_geodesic

from utils.data_parser_paths import data_asset_to_path
from pathlib import Path
import json


def decide_pose(pose):
    """
    Determines the orientation of a 3D pose based on the alignment of its z-vector with predefined orientations.

    Args:
        pose (np.ndarray): A 4x4 NumPy array representing a 3D pose transformation matrix.

    Returns:
        (int): Index representing the closest predefined orientation:
             0 for upright, 1 for left, 2 for upside-down, and 3 for right.
    """

    # pose style
    z_vec = pose[2, :3]
    z_orien = np.array(
        [
            [0.0, -1.0, 0.0], # upright
            [-1.0, 0.0, 0.0], # left
            [0.0, 1.0, 0.0], # upside-down
            [1.0, 0.0, 0.0], # right
        ]  
    )
    corr = np.matmul(z_orien, z_vec)
    corr_max = np.argmax(corr)
    return corr_max


def rotate_pose(im, rot_index):
    """
    Rotates an image by a specified angle based on the rotation index.

    Args:
        im (numpy.ndarray): The input image to be rotated. It should have shape (height, width, channels).
        rot_index (int): Index representing the rotation angle:
                         0 for no rotation, 1 for 90 degrees clockwise rotation,
                         2 for 180 degrees rotation, and 3 for 90 degrees counterclockwise rotation.

    Returns:
        (numpy.ndarray): The rotated image.
    """
    h, w, d = im.shape
    if d == 3:
        if rot_index == 0:
            new_im = im
        elif rot_index == 1:
            new_im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif rot_index == 2:
            new_im = cv2.rotate(im, cv2.ROTATE_180)
        elif rot_index == 3:
            new_im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return new_im


def convert_angle_axis_to_matrix3(angle_axis):
    """
    Converts a rotation from angle-axis representation to a 3x3 rotation matrix.

    Args:
        angle_axis (numpy.ndarray): A 3-element array representing the rotation in angle-axis form.

    Returns:
        (numpy.ndarray): A 3x3 rotation matrix representing the same rotation as the input angle-axis.

    Raises:
        ValueError: If the input is not a valid 3-element numpy array.
    """
    # Check if input is a numpy array
    if not isinstance(angle_axis, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    # Check if the input is of shape (3,)
    if angle_axis.shape != (3,):
        raise ValueError("Input must be a 3-element array representing the rotation in angle-axis representation.")
    
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

def convert_matrix3_to_angle_axis(matrix):
    """
    Converts a 3x3 rotation matrix to angle-axis representation (rotation vector).

    Args:
        matrix (numpy.ndarray): A 3x3 rotation matrix representing the rotation.

    Returns:
        (numpy.ndarray): A 3-element array representing the rotation in angle-axis form

    Raises:
        ValueError: If the input is not a valid 3x3 numpy array.
    """
    # Check if input is a numpy array
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    # Check if the input is of shape (3, 3)
    if matrix.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix representing the rotation.")
    
    # Convert the 3x3 rotation matrix to an angle-axis (rotation vector)
    angle_axis, jacobian = cv2.Rodrigues(matrix)
    
    return angle_axis.flatten()  # Return as a 1D array (rotation vector)

class DataParser:
    """
    A class for parsing data files in the SceneFun3D dataset.
    """

    def __init__(self, data_root_path):
        """
        Initialize the DataParser instance with the root path.

        Args:
            data_root_path (str): The root path where data is located.
        """
        self.data_root_path = os.path.join(data_root_path)

    def TrajStringToMatrix(self, traj_str):
        """ 
        Converts a line from the camera trajectory file into translation and rotation matrices.

        Args:
            traj_str (str): A space-delimited string where each line represents a camera pose at a particular timestamp. 
                            The line consists of seven columns:
                - Column 1: timestamp
                - Columns 2-4: rotation (axis-angle representation in radians)
                - Columns 5-7: translation (in meters)

        Returns:
            (tuple): A tuple containing:
                - ts (str): Timestamp.
                - Rt (numpy.ndarray): 4x4 transformation matrix representing rotation and translation.

        Raises:
            AssertionError: If the input string does not have exactly seven columns.
        """
        tokens = traj_str.split()
        assert len(tokens) == 7
        ts = tokens[0]

        # Rotation in angle axis
        angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
        r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))

        # Translation
        t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
        extrinsics = np.eye(4, 4)
        extrinsics[:3, :3] = r_w_to_p
        extrinsics[:3, -1] = t_w_to_p
        Rt = np.linalg.inv(extrinsics)

        return (ts, Rt)

    def get_camera_trajectory(self, visit_id, video_id, pose_source="colmap"):
        """
        Retrieve the camera trajectory from a file and convert it into a dictionary whose keys are timestamps and 
        values are the corresponding camera poses.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.
            pose_source (str, optional): Specifies the trajectory asset type, either "colmap" or "arkit". Defaults to "colmap".

        Returns:
            (dict): A dictionary where keys are timestamps (rounded to 3 decimal points) and values are 4x4 transformation matrices representing camera poses.

        Raises:
            AssertionError: If an unsupported trajectory asset type is provided.
        """
        assert pose_source in ["colmap", "arkit"], f"Unknown option {pose_source}"

        data_asset_identifier = "hires_poses" if pose_source == "colmap" else "lowres_poses"
        traj_file_path = self.get_data_asset_path(data_asset_identifier=f"{data_asset_identifier}", visit_id=visit_id, video_id=video_id)

        with open(traj_file_path) as f:
            traj = f.readlines()

        # Convert trajectory to a dictionary
        poses_from_traj = {}
        for line in traj:
            traj_timestamp = line.split(" ")[0]

            if pose_source == "colmap":
                poses_from_traj[f"{traj_timestamp}"] = np.array(self.TrajStringToMatrix(line)[1].tolist())
            elif pose_source == "arkit":
                poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = np.array(self.TrajStringToMatrix(line)[1].tolist())

        return poses_from_traj

    def get_laser_scan(self, visit_id):
        """
        Load a point cloud from a .ply file containing laser scan data.

        Args:
            visit_id (str): The identifier of the scene.

        Returns:
            (open3d.geometry.PointCloud): A point cloud object containing the laser scan data (i.e., XYZRGB point cloud).
        """
        laser_scan_path = self.get_data_asset_path(data_asset_identifier="laser_scan_5mm", visit_id=visit_id)

        pcd = o3d.io.read_point_cloud(laser_scan_path)

        return pcd

    def get_arkit_reconstruction(self, visit_id, video_id, format="point_cloud"):
        """
        Load ARKit mesh reconstruction data based on the iPad video sequence from a .ply file.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.
            format (str, optional): The format of the mesh reconstruction data to load. 
                                    Supported formats are "point_cloud" and "mesh". 
                                    Defaults to "point_cloud".

        Returns:
            (Union[open3d.geometry.PointCloud, open3d.geometry.TriangleMesh]): 
                The loaded mesh reconstruction data in the specified format.

        Raises:
            ValueError: If an unsupported 3D data format is specified.
        """
        mesh_path = self.get_data_asset_path(data_asset_identifier="arkit_mesh", visit_id=visit_id, video_id=video_id)

        mesh = None 

        if format == "point_cloud":
            mesh = o3d.io.read_point_cloud(mesh_path)
        elif format == "mesh":
            mesh = o3d.io.read_triangle_mesh(mesh_path)
        else: 
            raise ValueError(f"Unknown mesh format {format}")
        
        return mesh

    def get_rgb_frames(self, visit_id, video_id, data_asset_identifier="hires_wide"):
        """
        Retrieve the paths to the RGB frames for a given scene and video sequence.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.
            data_asset_identifier (str, optional): The data asset type for the RGB frames.
                                                   Can be either "hires_wide" or "lowres_wide". 
                                                   Defaults to "hires_wide".

        Returns:
            (dict): A dictionary mapping frame timestamps to their corresponding file paths.

        Raises:
            ValueError: If an unsupported data asset identifier is provided.
            FileNotFoundError: If no frames are found at the specified path.
        """
        frame_mapping = {}
        if data_asset_identifier == "hires_wide":
            rgb_frames_path = self.get_data_asset_path(data_asset_identifier="hires_wide", visit_id=visit_id, video_id=video_id)

            frames = sorted(glob.glob(os.path.join(rgb_frames_path, "*.jpg")))
            if not frames:
                raise FileNotFoundError(f"No RGB frames found in {rgb_frames_path}")
            frame_timestamps = [os.path.basename(x).split(".jpg")[0].split("_")[1] for x in frames]

        elif data_asset_identifier == "lowres_wide":
            rgb_frames_path = self.get_data_asset_path(data_asset_identifier="lowres_wide", visit_id=visit_id, video_id=video_id)

            frames = sorted(glob.glob(os.path.join(rgb_frames_path, "*.png")))
            if not frames:
                raise FileNotFoundError(f"No RGB frames found in {rgb_frames_path}")
            frame_timestamps = [os.path.basename(x).split(".png")[0].split("_")[1] for x in frames]
        else: 
            raise ValueError(f"Unknown data_asset_identifier {data_asset_identifier} for RGB frames")
        
        # Create mapping from timestamp to full path
        frame_mapping = {timestamp: frame for timestamp, frame in zip(frame_timestamps, frames)}

        return frame_mapping

    def get_depth_frames(self, visit_id, video_id, data_asset_identifier="hires_depth"):
        """
        Retrieve the paths to the depth frames for a given scene and video sequence.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.
            data_asset_identifier (str, optional): The data asset type for the depth frames.
                                                   Can be either "hires_depth" or "lowres_depth". 
                                                   Defaults to "hires_depth".

        Returns:
            (dict): A dictionary mapping frame timestamps to their corresponding file paths.

        Raises:
            ValueError: If an unsupported data asset identifier is provided.
            FileNotFoundError: If no depth frames are found at the specified path.
        """
        frame_mapping = {}
        if data_asset_identifier == "hires_depth":
            depth_frames_path = self.get_data_asset_path(data_asset_identifier="hires_depth", visit_id=visit_id, video_id=video_id)

        elif data_asset_identifier == "lowres_depth":
            depth_frames_path = self.get_data_asset_path(data_asset_identifier="lowres_depth", visit_id=visit_id, video_id=video_id)

        else: 
            raise ValueError(f"Unknown data_asset_identifier {data_asset_identifier} for depth frames")
        
        frames = sorted(glob.glob(os.path.join(depth_frames_path, "*.png")))
        if not frames:
            raise FileNotFoundError(f"No depth frames found in {depth_frames_path}")
        frame_timestamps = [os.path.basename(x).split(".png")[0].split("_")[1] for x in frames]

         # Create mapping from timestamp to full path
        frame_mapping = {timestamp: frame for timestamp, frame in zip(frame_timestamps, frames)}

        return frame_mapping

    def get_camera_intrinsics(self, visit_id, video_id, data_asset_identifier="hires_wide_intrinsics"):
        """
        Retrieve the camera intrinsics for a given scene and video sequence.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.
            data_asset_identifier (str, optional): The data asset type for camera intrinsics.
                                                   Can be either "hires_wide_intrinsics" or "lowres_wide_intrinsics". 
                                                   Defaults to "hires_wide_intrinsics".

        Returns:
            (dict): A dictionary mapping timestamps to file paths of camera intrinsics data.

        Raises:
            ValueError: If an unsupported data asset identifier is provided.
            FileNotFoundError: If no intrinsics files are found at the specified path.
        """
        intrinsics_mapping = {}
        if data_asset_identifier == "hires_wide_intrinsics":
            intrinsics_path = self.get_data_asset_path(data_asset_identifier="hires_wide_intrinsics", visit_id=visit_id, video_id=video_id)

        elif data_asset_identifier == "lowres_wide_intrinsics":
            intrinsics_path = self.get_data_asset_path(data_asset_identifier="lowres_wide_intrinsics", visit_id=visit_id, video_id=video_id)

        else: 
            raise ValueError(f"Unknown data_asset_identifier {data_asset_identifier} for camera intrinsics")

        intrinsics = sorted(glob.glob(os.path.join(intrinsics_path, "*.pincam")))
        
        if not intrinsics:
            raise FileNotFoundError(f"No camera intrinsics found in {intrinsics_path}")

        intrinsics_timestamps = [os.path.basename(x).split(".pincam")[0].split("_")[1] for x in intrinsics]

        # Create mapping from timestamp to full path
        intrinsics_mapping = {timestamp: cur_intrinsics for timestamp, cur_intrinsics in zip(intrinsics_timestamps, intrinsics)}

        return intrinsics_mapping

    def get_nearest_pose(self, 
                            desired_timestamp,
                            poses_from_traj, 
                            time_distance_threshold = np.inf):
        """
        Get the nearest pose to a desired timestamp from a dictionary of poses.

        Args:
            desired_timestamp (str): The timestamp of the desired pose.
            poses_from_traj (dict): A dictionary where keys are timestamps (as strings) 
                                    and values are 4x4 transformation matrices representing poses.
            time_distance_threshold (float, optional): The maximum allowable time difference 
                                                    between the desired timestamp and the nearest pose timestamp. Defaults to np.inf.

        Returns:
            (Union[numpy.ndarray, None]): The nearest pose as a 4x4 transformation matrix if found within the specified threshold, else None.

        Note:
            The function will return the pose closest to the desired timestamp if it exists in the provided poses.
            If the closest pose is further away than the specified `time_distance_threshold`, the function returns `None`.
        """
        max_pose_timestamp = max(float(key) for key in poses_from_traj.keys())
        min_pose_timestamp = min(float(key) for key in poses_from_traj.keys()) 

        if float(desired_timestamp) < min_pose_timestamp or \
            float(desired_timestamp) > max_pose_timestamp:
            return None

        if desired_timestamp in poses_from_traj.keys():
            H = poses_from_traj[desired_timestamp]
        else:
            closest_timestamp = min(
                poses_from_traj.keys(), 
                key=lambda x: abs(float(x) - float(desired_timestamp))
            )

            if abs(float(closest_timestamp) - float(desired_timestamp)) > time_distance_threshold:
                return None

            H = poses_from_traj[closest_timestamp]

        desired_pose = H

        assert desired_pose.shape == (4, 4)

        return desired_pose

    def get_interpolated_pose(self, 
                                desired_timestamp,
                                poses_from_traj, 
                                time_distance_threshold = np.inf,
                                interpolation_method = 'split',
                                frame_distance_threshold = np.inf):
        """
        Get the interpolated pose for a desired timestamp from a dictionary of poses.

        Args:
            desired_timestamp (str): The timestamp of the desired pose.
            poses_from_traj (dict): A dictionary where keys are timestamps (as strings) 
                                    and values are 4x4 transformation matrices representing poses.
            time_distance_threshold (float, optional): The maximum allowable time difference 
                                                    between the desired timestamp and the nearest pose timestamps. Defaults to np.inf.
            interpolation_method (str, optional): Method used for interpolation. Defaults to 'split'.
                - "split": Performs rigid body motion interpolation in SO(3) x R^3.
                - "geodesic_path": Performs rigid body motion interpolation in SE(3).
            frame_distance_threshold (float, optional): Maximum allowable frame distance between two consecutive poses. Defaults to np.inf.

        Returns:
            (Union[numpy.ndarray, None]): The interpolated pose as a 4x4 transformation matrix, or None if not found within thresholds.

        Raises:
            ValueError: If an unsupported interpolation method is specified.

        Note:
            This function uses interpolation between two nearest poses if `desired_timestamp` is not directly available.
            The interpolation method can be either "split" (for rigid body interpolation in SO(3) x R^3) or "geodesic_path" (for SE(3)).
            If the difference between the timestamps or poses is beyond the specified thresholds, the function will return None.
        """
        
        max_pose_timestamp = max(float(key) for key in poses_from_traj.keys())
        min_pose_timestamp = min(float(key) for key in poses_from_traj.keys()) 

        if float(desired_timestamp) < min_pose_timestamp or \
            float(desired_timestamp) > max_pose_timestamp:
            return None

        if desired_timestamp in poses_from_traj.keys():
            H = poses_from_traj[desired_timestamp]
        else:
            greater_closest_timestamp = min(
                [x for x in poses_from_traj.keys() if float(x) > float(desired_timestamp) ], 
                key=lambda x: abs(float(x) - float(desired_timestamp))
            )
            smaller_closest_timestamp = min(
                [x for x in poses_from_traj.keys() if float(x) < float(desired_timestamp) ], 
                key=lambda x: abs(float(x) - float(desired_timestamp))
            )

            if abs(float(greater_closest_timestamp) - float(desired_timestamp)) > time_distance_threshold or \
                abs(float(smaller_closest_timestamp) - float(desired_timestamp)) > time_distance_threshold:
                return None
            
            H0 = poses_from_traj[smaller_closest_timestamp]
            H1 = poses_from_traj[greater_closest_timestamp]
            H0_t = hm.trans(H0)
            H1_t = hm.trans(H1)

            if np.linalg.norm(H0_t - H1_t) > frame_distance_threshold:
                return None

            if interpolation_method == "split":
                H = rigid_interp_split(
                    float(desired_timestamp), 
                    poses_from_traj[smaller_closest_timestamp], 
                    float(smaller_closest_timestamp), 
                    poses_from_traj[greater_closest_timestamp], 
                    float(greater_closest_timestamp)
                )
            elif interpolation_method == "geodesic_path":
                H = rigid_interp_geodesic(
                    float(desired_timestamp), 
                    poses_from_traj[smaller_closest_timestamp], 
                    float(smaller_closest_timestamp), 
                    poses_from_traj[greater_closest_timestamp], 
                    float(greater_closest_timestamp)
                )
            else:
                raise ValueError(f"Unknown interpolation method {interpolation_method}")

        desired_pose = H

        assert desired_pose.shape == (4, 4)

        return desired_pose

    def get_transform(self, visit_id, video_id):
        """
        Load the transformation matrix from a .npy file. This transformation matrix converts coordinates from the Faro laser scan coordinate system to the ARKit coodinate system.

        Args:
            visit_id (str): The identifier of the scene.
            video_id (str): The identifier of the video sequence.

        Returns:
            (numpy.ndarray): The estimated transformation matrix loaded from the file.
        """
        transform_path = self.get_data_asset_path(data_asset_identifier="transform", visit_id=visit_id, video_id=video_id)
        transform = np.load(transform_path) 
        return transform

    def read_rgb_frame(self, rgb_frame_path, normalize=False):
        """
        Read an RGB frame from the specified path.

        Args:
            rgb_frame_path (str): The full path to the RGB frame file.
            normalize (bool, optional): Whether to normalize the pixel values to the range [0, 1]. Defaults to False.

        Returns:
            (numpy.ndarray): The RGB frame as a NumPy array with the RGB color values.

        """
        color = imageio.v2.imread(rgb_frame_path)

        if normalize:
            color = color / 255.

        return color

    def read_depth_frame(self, depth_frame_path, conversion_factor=1000):
        """
        Read a depth frame from the specified path and convert it to depth values.

        Args:
            depth_frame_path (str): The full path to the depth frame file.
            conversion_factor (float, optional): The conversion factor to convert pixel values to depth values. Defaults to 1000 to convert millimeters to meters.

        Returns:
            (numpy.ndarray): The depth frame as a NumPy array with the depth values.
        """

        depth = imageio.v2.imread(depth_frame_path) / conversion_factor

        return depth

    def read_camera_intrinsics(self, intrinsics_file_path, format="tuple"):
        """
        Parses a file containing camera intrinsic parameters and returns them in the specified format.

        Args:
            intrinsics_file_path (str): The path to the file containing camera intrinsic parameters.
            format (str, optional): The format in which to return the camera intrinsic parameters.
                                    Supported formats are "tuple" and "matrix". Defaults to "tuple".

        Returns:
            (Union[tuple, numpy.ndarray]): Camera intrinsic parameters in the specified format.

                - If format is "tuple", returns a tuple \\(w, h, fx, fy, hw, hh\\).
                - If format is "matrix", returns a 3x3 numpy array representing the camera matrix.
        
        Raises:
            ValueError: If an unsupported format is specified.
        """
        w, h, fx, fy, hw, hh = np.loadtxt(intrinsics_file_path)

        if format == "tuple":
            return (w, h, fx, fy, hw, hh)
        elif format == "matrix":
            return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])
        else:
            raise ValueError(f"Unknown format {format}")

    def get_crop_mask(self, visit_id, return_indices=False):
        """
        Load the crop mask from a .npy file.

        Args:
            visit_id (str): The identifier of the scene.
            return_indices (bool, optional): Whether to return the indices of the cropped points. Defaults to False.

        Returns:
            (numpy.ndarray): The crop mask loaded from the file. If `return_indices` is False, returns a Numpy array that is a binary mask of the indices to keep. If `return_indices` is True, returns a Numpy array containing the indices of the points to keep.
        """
        # crop_mask_path = os.path.join(self.data_root_path, visit_id, f"{visit_id}_crop_mask.npy")
        crop_mask_path = self.get_data_asset_path(data_asset_identifier="crop_mask", visit_id=visit_id)
        crop_mask = np.load(crop_mask_path)
        
        if return_indices:
            return np.where(crop_mask)[0]
        else:
            return crop_mask

    def get_cropped_laser_scan(self, visit_id, laser_scan):
        """
        Crop a laser scan using a crop mask.

        Args:
            visit_id (str): The identifier of the scene.
            laser_scan (open3d.geometry.PointCloud): The laser scan point cloud to be cropped.

        Returns:
            (open3d.geometry.PointCloud): The cropped laser scan point cloud.
        """
        filtered_idx_list = self.get_crop_mask(visit_id, return_indices=True)

        laser_scan_points = np.array(laser_scan.points)
        laser_scan_colors = np.array(laser_scan.colors)
        laser_scan_points = laser_scan_points[filtered_idx_list]
        laser_scan_colors = laser_scan_colors[filtered_idx_list]

        cropped_laser_scan = o3d.geometry.PointCloud()
        cropped_laser_scan.points = o3d.utility.Vector3dVector(laser_scan_points)
        cropped_laser_scan.colors = o3d.utility.Vector3dVector(laser_scan_colors)
        
        return cropped_laser_scan

    def get_data_asset_path(self, data_asset_identifier, visit_id, video_id=None):
        """
        Get the file path for a specified data asset.

        Args:
            data_asset_identifier (str): A string identifier for the data asset.
            visit_id (str or int): The identifier for the visit (scene).
            video_id (str or int, optional): The identifier for the video sequence. Required if specified data asset requires a video identifier.

        Returns:
            (Path): A Path object representing the file path to the specified data asset.

        Raises:
            AssertionError: If the `data_asset_identifier` is not valid or if `video_id` is required but not provided.
        """
        assert data_asset_identifier in data_asset_to_path, f"Data asset identifier '{data_asset_identifier}' is not valid"

        data_path = data_asset_to_path[data_asset_identifier]

        if ("<video_id>" in data_path) and (video_id is None):
            assert False, f"video_id must be specified for the data asset identifier '{data_asset_identifier}'"

        visit_id = str(visit_id)

        data_path = (
            data_path
                .replace("<data_dir>", self.data_root_path)
                .replace("<visit_id>", visit_id)
        )

        if "<video_id>" in data_path:
            video_id = str(video_id)
            data_path = data_path.replace("<video_id>", video_id)

        return data_path

    def get_annotations(self, visit_id, group_excluded_points=True):
        """
        Retrieve the functionality annotations for a specified scene.

        Args:
            visit_id (str or int): The identifier for the scene.
            group_excluded_points (bool, optional): If True, all annotations with the label "exclude" will be grouped together 
                                                    into a single annotation instance. Defaults to True.

        Returns:
            (list): A list of annotations, each represented as a dictionary.

        """
        annotations_path = self.get_data_asset_path(data_asset_identifier="annotations", visit_id=visit_id)

        annotations_data = None
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)["annotations"]

        if group_excluded_points:
            # group the excluded points into a single annotation instance
            exclude_indices_set = set()
            first_exclude_annotation = None
            filtered_annotation_data = []

            for annotation in annotations_data:
                if annotation["label"] == "exclude":
                    if first_exclude_annotation is None:
                        first_exclude_annotation = annotation
                    exclude_indices_set.update(annotation["indices"])
                else:
                    filtered_annotation_data.append(annotation)

            if first_exclude_annotation:
                first_exclude_annotation["indices"] = sorted(list(exclude_indices_set))
                filtered_annotation_data.append(first_exclude_annotation)

            annotations_data = filtered_annotation_data

        return annotations_data

    def get_descriptions(self, visit_id):
        """
        Retrieve the natural language task descriptions for a specified scene.

        Args:
            visit_id (str or int): The identifier for the scene.

        Returns:
            (list): A list of descriptions, each represented as a dictionary.
        """
        descriptions_path = self.get_data_asset_path(data_asset_identifier="descriptions", visit_id=visit_id)

        with open(descriptions_path, 'r') as f:
            descriptions_data = json.load(f)["descriptions"]

        return descriptions_data

    def get_motions(self, visit_id):
        """
        Retrieve the motion annotations for a specified scene.

        Args:
            visit_id (str or int): The identifier for the scene.

        Returns:
            (list): A list of motions, each represented as a dictionary.
        """
        motions_path = self.get_data_asset_path(data_asset_identifier="motions", visit_id=visit_id)

        with open(motions_path, 'r') as f:
            motions_data = json.load(f)["motions"]

        return motions_data