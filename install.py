import launch
import platform


if not launch.is_installed("cv2"):
    print("Installing requirements for easyphoto-webui")
    launch.run_pip("install opencv-python", "requirements for opencv")

if not launch.is_installed("tensorflow-cpu"):
    print("Installing requirements for easyphoto-webui")
    launch.run_pip("install tensorflow-cpu", "requirements for tensorflow")

if not launch.is_installed("onnx"):
    print("Installing requirements for easyphoto-webui")
    launch.run_pip("install onnx", "requirements for onnx")

if not launch.is_installed("onnxruntime"):
    print("Installing requirements for easyphoto-webui")
    launch.run_pip("install onnxruntime", "requirements for onnxruntime")

if not launch.is_installed("modelscope==1.9.3"):
    print("Installing requirements for easyphoto-webui")
    launch.run_pip("install modelscope==1.9.3", "requirements for modelscope")

if not launch.is_installed("diffusers==0.18.2"):
    print("Installing requirements for easyphoto-webui")
    launch.run_pip("install diffusers==0.18.2", "requirements for diffusers")

if not launch.is_installed("shapely"):
    print("Installing requirements for easyphoto-webui")
    launch.run_pip("install shapely", "requirements for shapely")


if platform.system() != "Windows":
    if not launch.is_installed("nvitop"):
        print("Installing requirements for easyphoto-webui")
        launch.run_pip("install nvitop==1.3.0", "requirements for tensorflow")
