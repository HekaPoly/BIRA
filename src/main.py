import argparse
import os
from pathlib import Path

from bira_orchestration.manager import BiraManager
    
def main():
    models_dir = Path(__file__).resolve().parents[1] / "models"

    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default=str(models_dir / 'yolov8n_resna.pt'), help='model.pt path(s)')
    # parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    # parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    # parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Run with mocked Camera/ComputerVision/Micro/STT/TTS and keep only SLM active.',
    )
    parser.add_argument(
        '--SLM_DEBUG',
        '--slm-debug',
        dest='slm_debug',
        action='store_true',
        help='Enable verbose SLM diagnostics (prompt metadata and Ollama done_reason).',
    )

    opt = parser.parse_args()
    if opt.slm_debug:
        os.environ['SLM_DEBUG'] = '1'
    bira_manager = BiraManager(mock_mode=opt.mock)
    bira_manager.preload()
    bira_manager.run()
    
    
if __name__ == "__main__":
    main()
    
