import launch


if not launch.is_installed("cv2"):
    print('Installing requirements for paiya-webui')
    launch.run_pip("install opencv-python", "requirements for opencv")

if not launch.is_installed("tensorflow-cpu"):
    print('Installing requirements for paiya-webui')
    launch.run_pip("install tensorflow-cpu", "requirements for tensorflow")

if not launch.is_installed("onnxruntime-gpu"):
    print('Installing requirements for paiya-webui')
    launch.run_pip("install onnxruntime-gpu", "requirements for onnxruntime-gpu")

if not launch.is_installed("ifnude"):
    print('Installing requirements for paiya-webui')
    launch.run_pip("install ifnude", "requirements for ifnude")

if not launch.is_installed("insightface"):
    print('Installing requirements for paiya-webui')
    launch.run_pip("install insightface==0.7", "requirements for insightface")

if not launch.is_installed("modelscope"):
    print('Installing requirements for paiya-webui')
    launch.run_pip("install modelscope", "requirements for modelscope")

if not launch.is_installed("diffusers"):
    print('Installing requirements for paiya-webui')
    launch.run_pip("install diffusers==0.18.2", "requirements for diffusers")