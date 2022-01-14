
def main():
    import os
    import glob
    import shutil

    video_path = '/hSDB_ai/VIDEO/Gastrectomy/Robot' 
    video_list = glob.glob(os.path.join(video_path, '*', '*'))

    for video in video_list:
        patient_no = '_'.join(video.split('/')[-1].split('_')[3:5])
        video_name = video.split('/')[-1]
        restore_path = os.path.join('/ai_shared/ViHUB-pro/input_video/trian_100', patient_no)
        os.makedirs(restore_path, exist_ok=True)

        # copy target_video
        print('COPY {} \n==========> {}\n'.format(video, os.path.join(restore_path, video_name))) # /data3/Public/ViHUB-pro/results/inference_json/train_100/Dataset1/R_2/01_G_01_R_2_ch1_01/01_G_01_R_2_ch1_01.mp4
        shutil.copy2(video, os.path.join(output_path, video_name))


if __name__ == '__main__':
    main()