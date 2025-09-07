"""
尝试运行原始的YOLO+SAM物体识别和掩码生成功能
如果依赖库不可用，将提供安装指导
"""

import os
import sys


def check_and_install_dependencies():
    """检查并尝试安装必要的依赖库"""
    required_packages = {
        'ultralytics': 'ultralytics',
        'torch': 'torch',
        'cv2': 'opencv-python',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    installed_packages = []
    
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'cv2':
                import cv2
                installed_packages.append(f"opencv-python: {cv2.__version__}")
            elif module_name == 'torch':
                import torch
                installed_packages.append(f"torch: {torch.__version__}")
            elif module_name == 'numpy':
                import numpy as np
                installed_packages.append(f"numpy: {np.__version__}")
            elif module_name == 'ultralytics':
                import ultralytics
                installed_packages.append(f"ultralytics: {ultralytics.__version__}")
        except ImportError:
            missing_packages.append(package_name)
    
    print("=== 依赖库检查结果 ===")
    print("已安装的库:")
    for pkg in installed_packages:
        print(f"  ✓ {pkg}")
    
    if missing_packages:
        print("\n缺失的库:")
        for pkg in missing_packages:
            print(f"  ✗ {pkg}")
        
        print(f"\n要安装缺失的库，请运行以下命令:")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        print(f"  {install_cmd}")
        return False
    else:
        print("\n所有依赖库已安装完毕！")
        return True


def run_advanced_demo():
    """运行高级的YOLO+SAM演示"""
    try:
        # 导入所需库
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        from ultralytics.models.sam import Predictor as SAMPredictor
        
        print("=== 高级物体识别和掩码生成演示 (YOLO + SAM) ===")
        
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        def detect_and_segment(image_path, output_dir=None):
            """使用YOLO检测 + SAM分割"""
            # 如果没有指定输出目录，则使用脚本所在目录下的子目录
            if output_dir is None:
                output_dir = os.path.join(script_dir, "advanced_test_results")
            
            # 清理并创建输出目录
            if os.path.exists(output_dir):
                import shutil
                print(f"清理旧的结果文件夹: {output_dir}")
                shutil.rmtree(output_dir)
            
            os.makedirs(output_dir)
            print(f"创建新的结果文件夹: {output_dir}")
            
            print(f"处理图像: {image_path}")
            
            # 检查图像是否存在
            if not os.path.exists(image_path):
                print(f"错误：图像文件不存在 - {image_path}")
                return False
            
            # 加载图像
            bgr_img = cv2.imread(image_path)
            if bgr_img is None:
                print(f"错误：无法读取图像 - {image_path}")
                return False
            
            print(f"图像尺寸: {bgr_img.shape}")
            
            # 1) YOLO检测
            print("正在加载 YOLO-World 模型...")
            print("注意：首次运行时会自动下载模型文件（约 100MB），请耐心等待...")
            try:
                # 指定YOLO模型保存在脚本目录下
                yolo_model_path = os.path.join(script_dir, "yolov8s-world.pt")
                model = YOLO(yolo_model_path)
                print("YOLO模型加载成功，开始检测物体...")
                results = model.predict(image_path)
                
                boxes = results[0].boxes
                vis_img = results[0].plot()
                
                # 保存检测结果
                basename = os.path.basename(image_path).split('.')[0]
                detection_output = os.path.join(output_dir, f"yolo_detection_{basename}.jpg")
                cv2.imwrite(detection_output, vis_img)
                print(f"YOLO检测结果已保存: {detection_output}")
                
                # 提取有效检测
                valid_detections = []
                if boxes is not None:
                    for box in boxes:
                        if box.conf.item() > 0.25:
                            valid_detections.append({
                                "xyxy": box.xyxy[0].tolist(),
                                "conf": box.conf.item(),
                                "cls": results[0].names[int(box.cls.item())]
                            })
                
                print(f"检测到 {len(valid_detections)} 个物体:")
                for i, det in enumerate(valid_detections):
                    print(f"  {i+1}. {det['cls']} (置信度: {det['conf']:.3f})")
                
            except Exception as e:
                print(f"YOLO检测失败: {e}")
                return False
            
            # 2) 用户选择目标物体
            if valid_detections:
                # 显示检测结果图像让用户参考
                print(f"\n请查看检测结果图像: {detection_output}")
                print("可以使用图片查看器打开查看检测到的物体")
                
                # 让用户选择目标物体
                while True:
                    try:
                        print(f"\n请选择要生成掩码的物体 (输入 1-{len(valid_detections)} 的数字):")
                        print("可选择的物体:")
                        for i, det in enumerate(valid_detections):
                            print(f"  {i+1}. {det['cls']} (置信度: {det['conf']:.3f})")
                        
                        user_input = input("\n请输入数字选择物体 (按回车退出): ").strip()
                        
                        if not user_input:  # 用户按回车退出
                            print("用户取消选择，退出程序")
                            return True
                        
                        selection = int(user_input)
                        if 1 <= selection <= len(valid_detections):
                            selected_det = valid_detections[selection - 1]
                            print(f"\n您选择了: {selected_det['cls']} (置信度: {selected_det['conf']:.3f})")
                            break
                        else:
                            print(f"请输入 1 到 {len(valid_detections)} 之间的数字")
                    
                    except ValueError:
                        print("请输入有效的数字")
                    except KeyboardInterrupt:
                        print("\n用户中断操作")
                        return True
                
                # 3) SAM分割选中的物体
                print("正在初始化 SAM 模型...")
                print("注意：首次运行时会自动下载SAM模型文件（约 300MB），请耐心等待...")
                try:
                    # 初始化SAM - 指定模型保存在脚本目录下
                    model_weight = os.path.join(script_dir, 'sam_b.pt')
                    overrides = dict(
                        task='segment',
                        mode='predict',
                        model=model_weight,
                        conf=0.01,
                        save=False
                    )
                    predictor = SAMPredictor(overrides=overrides)
                    print("SAM模型加载成功，开始生成掩码...")
                    
                    # 设置图像
                    image_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    predictor.set_image(image_rgb)
                    
                    # 为选中的物体生成掩码
                    print(f"正在为选中的 {selected_det['cls']} 生成SAM掩码...")
                    
                    try:
                        # 使用检测框生成掩码
                        sam_results = predictor(bboxes=[selected_det["xyxy"]])
                        
                        if sam_results and sam_results[0].masks:
                            # 获取掩码
                            mask = sam_results[0].masks.data[0].cpu().numpy()
                            mask = (mask > 0).astype(np.uint8) * 255
                            
                            # 保存掩码
                            mask_output = os.path.join(output_dir, f"sam_mask_{selected_det['cls']}_{basename}.png")
                            cv2.imwrite(mask_output, mask)
                            
                            # 创建叠加图像
                            colored_mask = np.zeros_like(bgr_img)
                            colored_mask[mask > 0] = [0, 255, 0]  # 绿色掩码
                            overlay = cv2.addWeighted(bgr_img, 0.7, colored_mask, 0.3, 0)
                            overlay_output = os.path.join(output_dir, f"sam_overlay_{selected_det['cls']}_{basename}.jpg")
                            cv2.imwrite(overlay_output, overlay)
                            
                            # 创建纯分割结果（只显示目标物体）
                            segmented = bgr_img.copy()
                            segmented[mask == 0] = [0, 0, 0]  # 背景变黑
                            segmented_output = os.path.join(output_dir, f"sam_segmented_{selected_det['cls']}_{basename}.jpg")
                            cv2.imwrite(segmented_output, segmented)
                            
                            print(f"\n✓ SAM掩码已保存: {mask_output}")
                            print(f"✓ 叠加图像已保存: {overlay_output}")
                            print(f"✓ 分割结果已保存: {segmented_output}")
                            
                            # 计算中心点
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                M = cv2.moments(contours[0])
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    print(f"✓ 物体中心点: ({cx}, {cy})")
                                
                                # 在原图上标记中心点
                                marked_img = bgr_img.copy()
                                cv2.circle(marked_img, (cx, cy), 10, (0, 255, 0), -1)
                                cv2.putText(marked_img, f"{selected_det['cls']}", (cx-50, cy-20), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                marked_output = os.path.join(output_dir, f"sam_center_{selected_det['cls']}_{basename}.jpg")
                                cv2.imwrite(marked_output, marked_img)
                                print(f"✓ 中心点标记图已保存: {marked_output}")
                            
                            return True
                        else:
                            print(f"✗ 无法为 {selected_det['cls']} 生成SAM掩码")
                            return False
                    
                    except Exception as e:
                        print(f"✗ SAM分割 {selected_det['cls']} 时出错: {e}")
                        return False
                
                except Exception as e:
                    print(f"SAM初始化失败: {e}")
                    return False
            else:
                print("没有检测到任何物体，无法进行分割")
                return False
            
            return True
        
        # 动态扫描目录中的所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        # 获取脚本所在目录作为工作目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = script_dir
        
        all_files = os.listdir(workspace_dir)
        test_images = []
        
        for file in all_files:
            file_lower = file.lower()
            for ext in image_extensions:
                if file_lower.endswith(ext):
                    # 检查文件是否存在且可读
                    file_path = os.path.join(workspace_dir, file)
                    if os.path.isfile(file_path):
                        test_images.append(file)
                    break
        
        if not test_images:
            print("错误：在工作目录中没有找到任何图片文件")
            print(f"搜索目录: {workspace_dir}")
            print(f"支持的格式: {', '.join(image_extensions)}")
            return False
        
        # 按文件名排序
        test_images.sort()
        
        print(f"\n在目录中找到 {len(test_images)} 个图片文件:")
        for i, img_name in enumerate(test_images):
            file_path = os.path.join(workspace_dir, img_name)
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            print(f"  {i+1}. {img_name} ({size_mb:.2f} MB)")
        
        while True:
            try:
                user_choice = input(f"\n请选择要处理的图像 (1-{len(test_images)}) 或按回车退出: ").strip()
                
                if not user_choice:
                    print("退出程序")
                    return True
                
                choice_idx = int(user_choice) - 1
                if 0 <= choice_idx < len(test_images):
                    selected_image = test_images[choice_idx]
                    image_path = os.path.join(workspace_dir, selected_image)
                    
                    print(f"\n{'='*60}")
                    print(f"处理图像: {selected_image}")
                    print('='*60)
                    
                    if detect_and_segment(image_path):
                        print(f"\n✓ 成功处理图像: {selected_image}")
                    else:
                        print(f"✗ 处理图像失败: {selected_image}")
                    break
                else:
                    print(f"请输入 1 到 {len(test_images)} 之间的数字")
            
            except ValueError:
                print("请输入有效的数字")
            except KeyboardInterrupt:
                print("\n用户中断操作")
                return True
        
    except ImportError as e:
        print(f"导入库失败: {e}")
        print("请确保已安装所有必要的依赖库")
        return False
    except Exception as e:
        print(f"运行高级演示时出错: {e}")
        return False


def clean_old_results():
    """清理所有旧的结果文件夹"""
    import shutil
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    result_dirs = [
        "advanced_test_results",
        "simple_test_results", 
        "test_results"
    ]
    
    cleaned_count = 0
    for dir_name in result_dirs:
        full_path = os.path.join(script_dir, dir_name)
        if os.path.exists(full_path):
            print(f"清理旧结果文件夹: {full_path}")
            shutil.rmtree(full_path)
            cleaned_count += 1
    
    if cleaned_count > 0:
        print(f"已清理 {cleaned_count} 个旧结果文件夹")
    else:
        print("没有找到需要清理的旧结果文件夹")


def main():
    """主函数"""
    print("交互式物体识别和掩码生成")
    print("="*50)
    
    # 首先清理旧的结果文件夹
    clean_old_results()
    
    # 检查依赖
    if check_and_install_dependencies():
        print("\n准备运行交互式演示...")
        print("程序将：")
        print("1. 让您选择要处理的图像")
        print("2. 使用YOLO检测图像中的物体")
        print("3. 让您选择要生成掩码的目标物体")
        print("4. 使用SAM为选中的物体生成精确掩码")
        print("\n重要提示：")
        print("- 首次运行时会自动下载AI模型文件")
        print("- YOLO模型约100MB，SAM模型约300MB")
        print("- 请确保网络连接正常，下载过程可能需要几分钟")
        print("- 模型下载完成后会缓存，后续运行会更快")
        run_advanced_demo()
    else:
        print("\n由于缺少依赖库，无法运行演示")
        print("请安装缺失的库后重新运行此脚本")
        print("\n安装命令:")
        print("pip install ultralytics torch opencv-python numpy")


if __name__ == '__main__':
    main()
