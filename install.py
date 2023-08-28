import launch


if not launch.is_installed("cv2"):
    print('Installing requirements for Mov2mov')
    launch.run_pip("install opencv-python", "requirements for opencv")
