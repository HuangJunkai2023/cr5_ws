import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

class D435iCapture:
    def __init__(self, width=1280, height=720, fps=30):
        """
        初始化Intel RealSense D435i相机
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        # 配置流
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置深度和颜色流
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # 启动管道
        try:
            self.profile = self.pipeline.start(self.config)
            print(f"D435i相机启动成功，分辨率: {width}x{height}, FPS: {fps}")
            
            # 获取相机内参
            self.get_camera_intrinsics()
            
        except Exception as e:
            print(f"相机启动失败: {e}")
            raise
    
    def get_camera_intrinsics(self):
        """
        获取相机内参
        """
        # 获取深度传感器的内参
        depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        
        # 获取彩色传感器的内参
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        
        print(f"深度相机内参:")
        print(f"  宽度: {depth_intrinsics.width}, 高度: {depth_intrinsics.height}")
        print(f"  fx: {depth_intrinsics.fx:.2f}, fy: {depth_intrinsics.fy:.2f}")
        print(f"  cx: {depth_intrinsics.ppx:.2f}, cy: {depth_intrinsics.ppy:.2f}")
        
        # 更新全局内参
        self.depth_intrinsics = depth_intrinsics
        self.color_intrinsics = color_intrinsics
        
        return depth_intrinsics, color_intrinsics
    
    def capture_frame(self):
        """
        采集一帧数据
        返回: color_image, depth_image, aligned_depth_image
        """
        try:
            # 等待帧
            frames = self.pipeline.wait_for_frames()
            
            # 获取深度和颜色帧
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None, None
            
            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 对齐深度图到彩色图
            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
            
            return color_image, depth_image, aligned_depth_image
            
        except Exception as e:
            print(f"采集帧失败: {e}")
            return None, None, None
    
    def save_frame(self, color_image, depth_image, save_dir, prefix="capture"):
        """
        保存采集的帧到指定目录（固定文件名，每次覆盖）
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 使用固定文件名，每次覆盖
        color_path = os.path.join(save_dir, f"{prefix}_color.png")
        depth_path = os.path.join(save_dir, f"{prefix}_depth.png")
        
        # 保存彩色图像
        cv2.imwrite(color_path, color_image)
        
        # 保存深度图像
        cv2.imwrite(depth_path, depth_image)
        
        print(f"帧已保存（覆盖上次文件）:")
        print(f"  彩色图: {color_path}")
        print(f"  深度图: {depth_path}")
        
        return color_path, depth_path
    
    def create_workspace_mask(self, depth_image, min_depth=300, max_depth=1500):
        """
        创建工作空间掩码
        min_depth, max_depth: 深度范围(毫米)
        """
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        valid_depth = (depth_image > min_depth) & (depth_image < max_depth)
        mask[valid_depth] = 255
        return mask
    
    def stop(self):
        """
        停止相机
        """
        try:
            self.pipeline.stop()
            print("相机已停止")
        except Exception as e:
            print(f"停止相机失败: {e}")

def test_camera():
    """
    测试相机采集功能
    """
    try:
        # 初始化相机
        camera = D435iCapture()
        
        # 采集一帧
        color, depth, aligned_depth = camera.capture_frame()
        
        if color is not None and depth is not None:
            # 保存到capture文件夹
            save_dir = "/home/huang/learn_arm_robot/graspnet-baseline/doc/capture"
            color_path, depth_path = camera.save_frame(color, aligned_depth, save_dir)
            print("测试采集成功!")
            return color_path, depth_path
        else:
            print("采集失败!")
            return None, None
            
    except Exception as e:
        print(f"测试失败: {e}")
        return None
    finally:
        camera.stop()

if __name__ == "__main__":
    test_camera()
