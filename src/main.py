import argparse
from pathlib import Path

from bira_orchestration.manager import BiraManager
    
def main():
    models_dir = Path(__file__).resolve().parents[1] / "models"

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=str(models_dir / 'yolov8n_resna.pt'), help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')

    opt = parser.parse_args()
    bira_manager = BiraManager()
    bira_manager.run()
    
    
if __name__ == "__main__":
    main()
    