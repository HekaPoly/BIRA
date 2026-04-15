import argparse
import os
import signal
import sys
from pathlib import Path

from bira_orchestration.manager import BiraManager


def setup_signal_handlers(bira_manager):
    """Set up graceful shutdown on CTRL-C (SIGINT)."""
    def signal_handler(signum, frame):
        print("\n\n[Main] CTRL-C received. Shutting down gracefully...")
        try:
            bira_manager.controller.destroy()
            print("[Main] All components destroyed. Exiting.")
        except Exception as e:
            print(f"[Main] Error during shutdown: {e}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)


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
        '--mode',
        choices=['local', 'cloud'],
        default='local',
        help='Select the SLM backend mode.',
    )
    parser.add_argument(
        '--api-key',
        dest='api_key',
        default=None,
        help='API key for cloud mode.',
    )
    parser.add_argument(
        '--slm-debug',
        action='store_true',
        help='Enable verbose SLM diagnostics.',
    )
    parser.add_argument(
        '--slm-stream',
        '--no-slm-stream',
        dest='slm_stream',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable streamed SLM output in the terminal.',
    )

    opt = parser.parse_args()
    if opt.slm_debug:
        os.environ['SLM_DEBUG'] = '1'
    if not opt.slm_stream:
        os.environ['SLM_STREAM'] = '0'

    bira_manager = BiraManager(
        mock_mode=opt.mock,
        slm_mode=opt.mode,
        slm_debug=opt.slm_debug,
        slm_stream=opt.slm_stream,
        api_key=opt.api_key,
    )
    setup_signal_handlers(bira_manager)
    bira_manager.preload()
    bira_manager.run()

    
if __name__ == "__main__":
    main()

