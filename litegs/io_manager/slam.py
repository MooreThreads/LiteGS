
from ..data import VideoFrame,ImageFrame,PinHoleCameraInfo
import numpy as np
import pandas as pd
import numpy.typing as npt
import re
import os.path
import datetime

def __read_intrinsics_text(path)->dict[int,PinHoleCameraInfo]:
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = PinHoleCameraInfo(id=camera_id,width=width, height=height,parameters=params)
    return cameras

def __read_extrinsics_text(file_path:str,frame_path:str,frame_type="image")->list[VideoFrame]:
    pattern = re.compile(
        r"(\d+)\s+"              # IMAGE_ID
        r"([-0-9.eE]+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)\s+"  # QW, QX, QY, QZ
        r"([-0-9.eE]+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)\s+"                 # TX, TY, TZ
        r"(\d+)\s+(\S+)\s+"      # CAMERA_ID, IMAGE_NAME
        r"mat4x4\(\(([^)]+)\),\s*\(([^)]+)\),\s*\(([^)]+)\),\s*\(([^)]+)\)\)"  # 4x4矩阵的四行
    )

    frame_list = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = pattern.match(line)
            if match:
                groups = match.groups()
                image_id = int(groups[0])
                qw, qx, qy, qz = map(float, groups[1:5])
                tx, ty, tz = map(float, groups[5:8])
                camera_id = int(groups[8])
                image_name = groups[9]
                trans_mat = np.array([list(map(float, row.split(','))) for row in groups[10:14]])
                if frame_type=="image":
                    frame=ImageFrame(image_id,np.array([qw,qx,qy,qz]),np.array([tx,ty,tz]),camera_id,image_name,os.path.join(frame_path,image_name),None)
                elif frame_type=="video":
                    frame=VideoFrame(image_id,np.array([qw,qx,qy,qz]),np.array([tx,ty,tz]),camera_id,image_name,frame_path,None)
                frame_list.append(frame)
                #assert(frame.view_matrix==trans_mat)
            else:
                print(f"Warning: line could not be parsed:\n{line}")

    return frame_list

def __read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def load_slam_result(path:str)->tuple[dict[int,PinHoleCameraInfo],list[ImageFrame],npt.NDArray,npt.NDArray]:

    items = os.listdir(path)
    mp4_files = [item for item in items if os.path.isfile(os.path.join(path, item)) and item.endswith('.mp4')]
    assert(len(mp4_files)==1)

    frame_list=__read_extrinsics_text(os.path.join(path,"inputs","slam","images.txt"),os.path.join(path,mp4_files[0]),"video")
    camreas=__read_intrinsics_text(os.path.join(path,"inputs","slam","cameras.txt"))
    xyz, rgb, _=__read_points3D_text(os.path.join(path,"inputs","slam","points3D.txt"))

    #timestamp to frame_id
    video_info = pd.read_csv(os.path.join(path,"inputs/videoInfo.txt"), delim_whitespace=True, header=None,names=["FrameID", "Timestamp", "Rotation"])
    for frame in frame_list:
        timestamp=int(frame.name.split('.')[0])
        nearest_frame = video_info.iloc[(video_info.Timestamp - timestamp).abs().argmin()].FrameID
        frame.name=nearest_frame
    return camreas,frame_list,xyz,rgb/255