"""
Helpers for parsing the data

SceneFun3D Toolkit
"""

data_asset_to_path = {
    'lowres_wide': '<data_dir>/<visit_id>/<video_id>/lowres_wide/',
    'lowres_wide_intrinsics': '<data_dir>/<visit_id>/<video_id>/lowres_wide_intrinsics/',
    'lowres_depth': '<data_dir>/<visit_id>/<video_id>/lowres_depth/',
    'confidence': '<data_dir>/<visit_id>/<video_id>/confidence/',
    'hires_wide': '<data_dir>/<visit_id>/<video_id>/hires_wide/',
    'hires_wide_intrinsics': '<data_dir>/<visit_id>/<video_id>/hires_wide_intrinsics/',
    'hires_depth': '<data_dir>/<visit_id>/<video_id>/hires_depth/',
    # 'vga_wide',
    # 'vga_wide_intrinsics',
    # 'ultrawide',
    # 'ultrawide_intrinsics',
    'lowres_poses': '<data_dir>/<visit_id>/<video_id>/lowres_poses.traj',
    'hires_poses': '<data_dir>/<visit_id>/<video_id>/hires_poses.traj',
    'vid_mov': '<data_dir>/<visit_id>/<video_id>/<video_id>.mov',
    'vid_mp4': '<data_dir>/<visit_id>/<video_id>/<video_id>.mp4',
    'arkit_mesh': '<data_dir>/<visit_id>/<video_id>/<video_id>_arkit_mesh.ply',
    # '3dod_annotation',
    'laser_scan_5mm': '<data_dir>/<visit_id>/<visit_id>_laser_scan.ply',
    'crop_mask': '<data_dir>/<visit_id>/<visit_id>_crop_mask.npy',
    'transform': '<data_dir>/<visit_id>/<video_id>/<video_id>_transform.npy',
    'annotations': '<data_dir>/<visit_id>/<visit_id>_annotations.json',
    'descriptions': '<data_dir>/<visit_id>/<visit_id>_descriptions.json',
    'motions': '<data_dir>/<visit_id>/<visit_id>_motions.json',
}