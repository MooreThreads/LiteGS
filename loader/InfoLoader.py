import os
import struct
import collections
import numpy as np
import typing
import numpy.typing as npt
import math
import PIL.Image
from training.arguments import ModelParams
from util import qvec2rotmat,getWorld2View


class CameraInfo:
    def __init__(self):
        self.id:int=0
        self.model:str=''
        self.width:int=0
        self.height:int=0
        return
    
    def __init__(self,id:int,model_name:str,width:int,height:int):
        self.id:int=id
        self.model:str=model_name
        self.width:int=width
        self.height:int=height
        return
    

    
class PinHoleCameraInfo(CameraInfo):
    def __init__(self,id:int,width:int,height:int,parameters:typing.List):
        super(PinHoleCameraInfo,self).__init__(id,"PINHOLE",width,height)
        focal_length_x=parameters[0]
        focal_length_y=parameters[1]

        def __focal2fov(focal,pixels):
            return 2*math.atan(pixels/(2*focal))
        
        self.fovX=__focal2fov(focal_length_x, width)
        self.fovY=__focal2fov(focal_length_y, height)
        return
    
WARNED = False

class ImageInfo:
    def __init__(self):
        self.id:int=0
        self.viewtransform_rotation:npt.NDArray=np.array((0,0,0,0))
        self.viewtransform_position:npt.NDArray=np.array((0,0,0))
        self.camera_id:int=0
        self.name:str=""
        self.xys=np.array((0,0,0,0))
        return
    
    def __init__(self,id:int,qvec:npt.ArrayLike,tvec:npt.ArrayLike,camera_id:int,name:str,xys:npt.ArrayLike):
        self.id:int=id
        self.viewtransform_rotation:npt.NDArray=np.transpose(qvec2rotmat(np.array(qvec)))
        self.viewtransform_position:npt.NDArray=np.array(tvec)
        self.camera_id:int=camera_id
        self.name:str=name
        self.xys:npt.NDArray=np.array(xys)
        self.image=None
        return
    
    def load_image(self,data_path:str,img_dir:str,arg_resolution:int):
        img_path=os.path.join(data_path,img_dir,self.name)
        self.image=PIL.Image.open(img_path)

        orig_w, orig_h = self.image.size
        if arg_resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/ arg_resolution), round(orig_h/ arg_resolution)
        else:  # should be a type that converts to float
            if arg_resolution == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / arg_resolution

            scale = float(global_down)
            resolution = (int(orig_w / scale), int(orig_h / scale))  
        self.image=self.image.resize(resolution)
        return
    
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "viewtransform_rotation", "tvec", "camera_id", "name", "xys", "point3D_ids"])


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def __read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def __read_intrinsics_text(path):
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
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def __read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = __read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = __read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            viewtransform_rotation = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = __read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = __read_next_bytes(fid, 1, "c")[0]
            num_points2D = __read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = __read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, viewtransform_rotation=viewtransform_rotation, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def __read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = __read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = __read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = __read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def __read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                viewtransform_rotation = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, viewtransform_rotation=viewtransform_rotation, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def getNerfppNorm(image_info_list:typing.List[ImageInfo]):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for imageInfo in image_info_list:
        W2C = getWorld2View(imageInfo.viewtransform_rotation, imageInfo.viewtransform_position)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return translate,radius


def load(path:str,image_dir:str,resolution:int)->tuple[dict[int,PinHoleCameraInfo],list[ImageInfo]]:
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = __read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = __read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = __read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = __read_intrinsics_text(cameras_intrinsic_file)

    CameraInfoDict:typing.Dict[int,PinHoleCameraInfo]={}
    ImageInfoList:typing.List[ImageInfo]=[]

    for CameraArg in cam_intrinsics.values():
        if(CameraArg.model=="PINHOLE"):
            CameraInfoDict[CameraArg.id]=PinHoleCameraInfo(CameraArg.id,CameraArg.width,CameraArg.height,CameraArg.params)

    H=None
    W=None
    for ImgArg in cam_extrinsics.values():
        if ImgArg.camera_id in CameraInfoDict.keys():
            imgInfo=ImageInfo(ImgArg.id,ImgArg.viewtransform_rotation,ImgArg.tvec,ImgArg.camera_id,ImgArg.name,ImgArg.xys)
            imgInfo.load_image(path,image_dir,resolution)
            if H is None and W is None:
                H,W=imgInfo.image.size
            else:
                assert(H==imgInfo.image.size[0])
                assert(W==imgInfo.image.size[1])
            ImageInfoList.append(imgInfo)
    ImageInfoListSorted = sorted(ImageInfoList.copy(), key = lambda x : x.name)
    NerfNormTrans,NerfNormRadius=getNerfppNorm(ImageInfoListSorted)

    return (CameraInfoDict,ImageInfoListSorted,NerfNormTrans,NerfNormRadius)