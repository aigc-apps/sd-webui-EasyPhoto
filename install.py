import launch


if not launch.is_installed("cv2"):
    print('Installing requirements for paiya-webui')
    launch.run_pip("install opencv-python", "requirements for opencv")

if not launch.is_installed("tensorflow"):
    print('Installing requirements for paiya-webui')
    launch.run_pip("install tensorflow", "requirements for tensorflow")
