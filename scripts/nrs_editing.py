
# input video path (임시 테스트용) : '/NAS/nas19/DATA/IMPORT/211220/12_14/gangbuksamsung_127case'
# output dir path (임시 테스트용) : '/raid/NRS_Editing'

def extract_target_video_path(target_video_path):
    import os
    '''
    target_video_path = /data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01.mp4
    base_path = /data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1
    target_path = /data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01
    '''
    
    base_path = '/'.join(target_video_path.split('/')[:-1])
    target_path = os.path.join(base_path, target_video_path.split('/')[-1].split('.')[0]) # 마지막 비디오 명에 '.' 이 들어가 있으면 에러 -> naming rule 지켜져야 함. 

    return target_path
            

def frame_cutting(target_video):
    import sys
    import os    

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    sys.path.append(base_path)
    sys.path.append(base_path+'core')

    from core.utils.ffmpegHelper import ffmpegHelper

    # save_path 
    frame_save_path = os.path.join(extract_target_video_path(target_video), 'frames')
    os.makedirs(frame_save_path, exist_ok=True)

    # frame cutting -> save to '/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01/frames/frame-000000000.jpg'
    ffmpeg_helper = ffmpegHelper(target_video, frame_save_path)
    ffmpeg_helper.cut_frame_total()


def get_experiment_args():
    from core.config.base_opts import parse_opts

    parser = parse_opts()

    args = parser.parse_args()

    ### model basic info opts
    args.pretrained = True
    # TODO 원하는대로 변경 하기
    # 전 그냥 save path와 동일하게 가져갔습니다. (bgpark)
    args.save_path = args.save_path + '-trial:{}-fold:{}'.format(args.trial, args.fold)
    # args.save_path = args.save_path + '-model:{}-IB_ratio:{}-WS_ratio:{}-hem_extract_mode:{}-top_ratio:{}-seed:{}'.format(args.model, args.IB_ratio, args.WS_ratio, args.hem_extract_mode, args.top_ratio, args.random_seed) # offline method별 top_ratio별 IB_ratio별 실험을 위해
    args.experiments_sheet_dir = args.save_path

    ### dataset opts
    args.data_base_path = '/raid/img_db'

    ### train args
    args.num_gpus = 1
    
    ### etc opts
    args.use_lightning_style_save = True # TO DO : use_lightning_style_save==False 일 경우 오류해결 (True일 경우 정상작동)

    return args


def inference(target_video):
    '''
    inference
    결과 저장은 /data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01/results/04_GS4_99_L_1_01.json
    추후 정량 평가 가능하면, results 내에 결과 저장. (CR, OR 등)
    '''
    import os
    import pandas as pd
    import glob
    import sys

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    sys.path.append(base_path)
    sys.path.append(base_path+'core')
    sys.path.append(base_path+'/core/accessory/RepVGG')

    from core.utils.ffmpegHelper import ffmpegHelper

    ### inference_main
    ### test inference module
    from core.api.trainer import CAMIO
    from core.api.inference import InferenceDB # inference module
    from core.api.evaluation import Evaluator # evaluation module
    from core.utils.metric import MetricHelper # metric helper (for calc CR, OR, mCR, mOR)
    from core.utils.logger import Report # report helper (for experiments reuslts and inference results)

    from core.api.visualization import VisualTool # visual module



    target_frame_path = os.path.join(extract_target_video_path(target_video), 'frames')
    result_save_path = os.path.join(extract_target_video_path(target_video), 'results')
    os.makedirs(result_save_path, exist_ok=True)

    # model_path = get_inference_model_path(os.path.join(args.restore_path, 'checkpoints'))
    model_path = '/NRS_EDITING/model_ckpt/ckpoint_0816-test-mobilenet_v3_large-model=mobilenet_v3_large-batch=32-lr=0.001-fold=1-ratio=3-epoch=24-last.ckpt'
    
    model = CAMIO.load_from_checkpoint(model_path, args=args) # ckpt
    model = model.cuda()














def video_editing():
    pass



def main():

    ## 1. input 비디오 목록 읽기
        ## 1-1. 중복 확인 (저장 디렉토리에 결과 존재하는지 확인: 비디오 파일 명 + 알파)
    base_target_path = '/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case'

    import os

    for (root, dirs, files) in os.walk(base_target_path):
        for file in files:
            # extention 예외 처리 ['mp4', 'mpg']
            if file.split('.')[-1] not in ['mp4', 'mpg']:
                continue

            print('\nProcessing in \t====>\t{}\n'.format(os.path.join(root, file)))

            ## 2. 비디오 전처리 (frmae 추출) -> 임시 디렉토리
            # frame_cutting(os.path.join(root, file))

            ## 3. inference (비디오 단위) -> 저장 디렉토리 & result json 생성
            inference(os.path.join(root, file))

            exit(0)




    
    ## 3. inference (비디오 단위) -> 저장 디렉토리 & result json 생성
    ## 4. 비디오 편집 (ffmpep script)
    ## 5. 임시 디렉토리 삭제
        ## 5-1. log 파일 생성


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        sys.path.append(base_path+'/core')
        print(base_path)

    main()