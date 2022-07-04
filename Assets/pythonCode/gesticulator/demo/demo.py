from argparse import ArgumentParser
import os
import subprocess

import torch
import librosa

from gesticulator.model.model import GesticulatorModel
from gesticulator.interface.gesture_predictor import GesturePredictor
from gesticulator.visualization.motion_visualizer.generate_videos import visualize

def main():
    # 0. Check feature type based on the model
    # cwd = os.getcwd()
    # print("cwd: ", cwd)
    # print("sys.path: ", sys.path)
    # return cwd
    # os.chdir("C:/Users/CY/Desktop/desktop/pythonRuntime/Assets/pythonCode/gesticulator/demo")

    args = parse_args()
    feature_type, audio_dim = check_feature_type(args.model_file)

    # 1. Load the model
    model = GesticulatorModel.load_from_checkpoint(
        args.model_file, inference_mode=True)
    
    # This interface is a wrapper around the model for predicting new gestures conveniently
    gp = GesturePredictor(model, feature_type)

    # 2. Predict the gestures with the loaded model

    """
        Generate a sequence of gestures based on a sequence of speech features (audio and text)

        Args:
            audio [N, T, D_a]:    a batch of sequences of audio features
            text  [N, T/2, D_t]:  a batch of sequences of text BERT embedding
            use_conditioning:     a flag indicating if we are using autoregressive conditioning
            motion: [N, T, D_m]   the true motion corresponding to the input (NOTE: it can be None during testing and validation)
            use_teacher_forcing:  a flag indicating if we use teacher forcing

        Returns:
            motion [N, T, D_m]:   a batch of corresponding motion sequences
        """
    motion = gp.predict_gestures(args.audio, args.text)

    # 3. Visualize the results
    motion_length_sec = int(motion.shape[1] / 20)

    

    visualize(motion.detach(), "temp.bvh", "temp.npy", "temp.mp4", 
                start_t = 0, end_t = motion_length_sec, 
                data_pipe_dir = 'C:/Users/CY/Desktop/desktop/pythonRuntime/Assets/pythonCode/gesticulator/gesticulator/utils/data_pipe.sav')

    # Add the audio to the video
    command = f"ffmpeg -y -i {args.audio} -i temp.mp4 -c:v libx264 -c:a libvorbis -loglevel quiet -shortest {args.video_out}"
    subprocess.call(command.split())

    print("\nGenerated video:", args.video_out)

    # Copy temporary files to Asset Folder
    # import os.path
    # import shutil
    # original_path = r'C:/Users/CY/Desktop/desktop/pythonRuntime/temp.bvh'
    # save_path = r'C:/Users/CY/Desktop/desktop/pythonRuntime/Assets/temp.bvh'
    # dest = shutil.copyfile("C:/Users/CY/Desktop/desktop/pythonRuntime/temp.bvh", "C:/Users/CY/Desktop/desktop/pythonRuntime/Assets/temp.bvh")
    



def check_feature_type(model_file):
    """
    Return the audio feature type and the corresponding dimensionality
    after inferring it from the given model file.
    """
    params = torch.load(model_file, map_location=torch.device('cpu'))

    # audio feature dim. + text feature dim.
    audio_plus_text_dim = params['state_dict']['encode_speech.0.weight'].shape[1]

    # This is a bit hacky, but we can rely on the fact that 
    # BERT has 768-dimensional vectors
    # We add 5 extra features on top of that in both cases.
    text_dim = 768 + 5

    audio_dim = audio_plus_text_dim - text_dim

    if audio_dim == 4:
        feature_type = "Pros"
    elif audio_dim == 64:
        feature_type = "Spectro"
    elif audio_dim == 68:
        feature_type = "Spectro+Pros"
    elif audio_dim == 26:
        feature_type = "MFCC"
    elif audio_dim == 30:
        feature_type = "MFCC+Pros"
    else:
        print("Error: Unknown audio feature type of dimension", audio_dim)
        exit(-1)

    return feature_type, audio_dim


def truncate_audio(input_path, target_duration_sec):
    """
    Load the given audio file and truncate it to 'target_duration_sec' seconds.
    The truncated file is saved in the same folder as the input.
    """
    audio, sr = librosa.load(input_path, duration = int(target_duration_sec))
    output_path = input_path.replace('.wav', f'_{target_duration_sec}s.wav')

    librosa.output.write_wav(output_path, audio, sr)

    return output_path

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--audio', type=str, default="C:/Users/CY/Desktop/desktop/pythonRuntime/Assets/pythonCode/gesticulator/demo/input/jeremy_howard.wav", help="path to the input speech recording")
    parser.add_argument('--text', type=str, default="C:/Users/CY/Desktop/desktop/pythonRuntime/Assets/pythonCode/gesticulator/demo/input/jeremy_howard.json",
                        help="one of the following: "
                             "1) path to a time-annotated JSON transcription (this is what the model was trained with) "
                             "2) path to a plaintext transcription, or " 
                             "3) the text transcription itself (as a string)")
    parser.add_argument('--video_out', '-video', type=str, default="C:/Users/CY/Desktop/desktop/pythonRuntime/Assets/pythonCode/gesticulator/demo/output/generated_motion.mp4",
                        help="the path where the generated video will be saved.")
    parser.add_argument('--model_file', '-model', type=str, default="C:/Users/CY/Desktop/desktop/pythonRuntime/Assets/pythonCode/gesticulator/demo/models/default.ckpt",
                        help="path to a pretrained model checkpoint")
    parser.add_argument('--mean_pose_file', '-mean_pose', type=str, default="C:/Users/CY/Desktop/desktop/pythonRuntime/Assets/pythonCode/gesticulator/gesticulator/utils/mean_pose.npy",
                        help="path to the mean pose in the dataset (saved as a .npy file)")
    
    return parser.parse_args()

if __name__ == "__main__":
    # args = parse_args()
    
    main()
